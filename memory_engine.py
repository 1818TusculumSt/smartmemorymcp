import json
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import asyncio
import time

from embeddings import EmbeddingProvider
from llm_client import LLMClient
from config import settings

logger = logging.getLogger(__name__)

class MemoryEngine:
    """
    Core memory management engine.
    
    Handles:
    - Memory extraction from text using LLM
    - Deduplication via embedding similarity
    - Storage in Pinecone vector database
    - Semantic search and retrieval
    - Automatic pruning when memory limit exceeded
    """
    
    def __init__(self):
        self.embedder = EmbeddingProvider()
        self.llm = LLMClient()
        self.embedding_dim = None
        
        logger.info("Initializing Pinecone connection")
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        index_name = settings.PINECONE_INDEX_NAME
        
        self._determine_embedding_dimension()
        
        if index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=self.embedding_dim,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=settings.PINECONE_ENVIRONMENT
                )
            )
            logger.info(f"Pinecone index created with dimension {self.embedding_dim}")
        
        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    def _determine_embedding_dimension(self):
        """Determine embedding dimension based on provider"""
        if settings.EMBEDDING_PROVIDER == "local":
            model_dims = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "all-MiniLM-L12-v2": 384,
            }
            self.embedding_dim = model_dims.get(settings.EMBEDDING_MODEL, 384)
            logger.info(f"Using embedding dimension {self.embedding_dim} for {settings.EMBEDDING_MODEL}")
        elif settings.EMBEDDING_PROVIDER == "pinecone":
            pinecone_dims = {
                "llama-text-embed-v2": 384,  # Configured to match existing index
                "multilingual-e5-large": 1024,
            }
            self.embedding_dim = pinecone_dims.get(settings.EMBEDDING_MODEL, 384)
            logger.info(f"Using Pinecone inference dimension {self.embedding_dim} for {settings.EMBEDDING_MODEL}")
        elif settings.EMBEDDING_PROVIDER == "api":
            # API provider dimensions for common models
            api_dims = {
                # Voyage AI models
                "voyage-3": 1024,
                "voyage-3.5": 1024,
                "voyage-3-lite": 512,
                "voyage-code-3": 1024,
                "voyage-finance-2": 1024,
                "voyage-multilingual-2": 1024,
                "voyage-law-2": 1024,
                # OpenAI models
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
                # Cohere models
                "embed-english-v3.0": 1024,
                "embed-multilingual-v3.0": 1024,
            }
            self.embedding_dim = api_dims.get(settings.EMBEDDING_MODEL, 1536)
            logger.info(f"Using API embedding dimension {self.embedding_dim} for {settings.EMBEDDING_MODEL}")
        else:
            self.embedding_dim = 1536
            logger.info(f"Using default embedding dimension {self.embedding_dim}")
    
    async def extract_and_store(
        self,
        user_message: str,
        recent_history: List[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract memories from message and store them"""
        logger.info(f"Starting memory extraction for message: {user_message[:100]}...")

        memories = await self._extract_memories(user_message, recent_history or [])

        if not memories:
            logger.info("No memories extracted from message")
            return []

        logger.info(f"Extracted {len(memories)} potential memories")

        filtered = [
            m for m in memories
            if m.get("confidence", 0) >= settings.MIN_CONFIDENCE
        ]

        discarded = len(memories) - len(filtered)
        if discarded > 0:
            logger.info(f"Filtered out {discarded} low-confidence memories (threshold: {settings.MIN_CONFIDENCE})")

        if not filtered:
            logger.info("No memories passed confidence filter")
            return []

        deduplicated = await self._deduplicate(filtered, user_id=user_id)

        duplicates = len(filtered) - len(deduplicated)
        if duplicates > 0:
            logger.info(f"Removed {duplicates} duplicate memories")

        if not deduplicated:
            logger.info("All memories were duplicates")
            return []

        stored = []
        for memory in deduplicated:
            result = await self._store_memory(
                memory,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id
            )
            if result:
                stored.append(result)

        logger.info(f"Successfully stored {len(stored)} new memories")

        await self._prune_if_needed()

        return stored
    
    async def get_recent(
        self,
        limit: int = 10,
        user_id: Optional[str] = None,
        since: Optional[str] = None,
        before: Optional[str] = None,
        include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """Get most recent memories with optional date filtering"""
        try:
            from datetime import datetime

            all_mems = await self._get_all_memories(user_id=user_id)

            # Filter by date range if specified
            filtered = []
            for mem in all_mems:
                # Skip archived unless explicitly included
                if not include_archived and mem["metadata"].get("archived", False):
                    continue

                timestamp = mem["metadata"].get("timestamp", "")

                # Apply date filters
                if since and timestamp < since:
                    continue
                if before and timestamp > before:
                    continue

                filtered.append(mem)

            # Sort by timestamp descending (newest first)
            filtered.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)

            recent = []
            for mem in filtered[:limit]:
                # Handle tags (backwards compatible with both string and array)
                tags_raw = mem["metadata"].get("tags", "")
                tags = tags_raw if isinstance(tags_raw, list) else (tags_raw.split(",") if tags_raw else [])

                recent.append({
                    "id": mem["id"],
                    "content": mem["metadata"].get("content", ""),
                    "tags": tags,
                    "category": mem["metadata"].get("category"),
                    "confidence": mem["metadata"].get("confidence", 0.5),
                    "importance": mem["metadata"].get("importance"),
                    "pinned": mem["metadata"].get("pinned", False),
                    "archived": mem["metadata"].get("archived", False),
                    "sentiment": mem["metadata"].get("sentiment"),
                    "word_count": mem["metadata"].get("word_count"),
                    "timestamp": timestamp,
                    "created_at": mem["metadata"].get("created_at"),
                    "updated_at": mem["metadata"].get("updated_at"),
                    "event_date": mem["metadata"].get("event_date"),
                    "user_id": mem["metadata"].get("user_id")
                })

            return recent
        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []

    async def get_all_tags(
        self,
        user_id: Optional[str] = None,
        min_count: int = 1,
        sort_by: str = "count"
    ) -> List[Dict[str, Any]]:
        """Get all unique tags with usage counts"""
        try:
            all_mems = await self._get_all_memories(user_id=user_id)

            # Aggregate tags
            tag_counts = {}
            for mem in all_mems:
                # Skip archived memories
                if mem["metadata"].get("archived", False):
                    continue

                tags_raw = mem["metadata"].get("tags", "")
                tags = tags_raw if isinstance(tags_raw, list) else (tags_raw.split(",") if tags_raw else [])

                for tag in tags:
                    tag = tag.strip()
                    if tag:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Filter by min_count and format
            result = [
                {"name": tag, "count": count}
                for tag, count in tag_counts.items()
                if count >= min_count
            ]

            # Sort by count (descending) or name (alphabetical)
            if sort_by == "name":
                result.sort(key=lambda x: x["name"])
            else:  # sort_by == "count"
                result.sort(key=lambda x: x["count"], reverse=True)

            return result
        except Exception as e:
            logger.error(f"Failed to get all tags: {e}")
            return []

    async def add_tags_to_memory(
        self,
        memory_id: str,
        tags: List[str],
        replace: bool = False
    ) -> Dict[str, Any]:
        """Add or replace tags on an existing memory"""
        try:
            # Fetch the memory
            result = self.index.fetch(ids=[memory_id])

            if not result or "vectors" not in result or memory_id not in result["vectors"]:
                logger.error(f"Memory {memory_id} not found")
                return {"success": False, "error": "Memory not found"}

            mem = result["vectors"][memory_id]
            metadata = mem["metadata"]

            # Get existing tags
            existing_tags_raw = metadata.get("tags", "")
            if isinstance(existing_tags_raw, list):
                existing_tags = existing_tags_raw
            else:
                existing_tags = existing_tags_raw.split(",") if existing_tags_raw else []

            # Update tags
            if replace:
                new_tags = tags
            else:
                # Merge with existing, remove duplicates
                new_tags = list(set(existing_tags + tags))

            # Update metadata
            metadata["tags"] = new_tags
            metadata["updated_at"] = datetime.now().isoformat()

            # Upsert back to Pinecone
            self.index.upsert(
                vectors=[(memory_id, mem["values"], metadata)]
            )

            logger.info(f"Updated tags for memory {memory_id}: {new_tags}")

            return {
                "success": True,
                "memory_id": memory_id,
                "tags": new_tags,
                "added": list(set(tags) - set(existing_tags)) if not replace else tags,
                "existing": list(set(tags) & set(existing_tags)) if not replace else []
            }

        except Exception as e:
            logger.error(f"Failed to add tags to memory: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def search_by_tag(
        self,
        tag: str,
        limit: int = 20,
        user_id: Optional[str] = None,
        sort_by: str = "created_at"
    ) -> List[Dict[str, Any]]:
        """Get all memories with a specific tag"""
        try:
            all_mems = await self._get_all_memories(user_id=user_id)

            # Filter by tag
            tagged = []
            for mem in all_mems:
                # Skip archived
                if mem["metadata"].get("archived", False):
                    continue

                tags_raw = mem["metadata"].get("tags", "")
                tags = tags_raw if isinstance(tags_raw, list) else (tags_raw.split(",") if tags_raw else [])
                tags = [t.strip().lower() for t in tags]

                if tag.lower() in tags:
                    tags_raw = mem["metadata"].get("tags", "")
                    tags_display = tags_raw if isinstance(tags_raw, list) else (tags_raw.split(",") if tags_raw else [])

                    tagged.append({
                        "id": mem["id"],
                        "content": mem["metadata"].get("content", ""),
                        "tags": tags_display,
                        "category": mem["metadata"].get("category"),
                        "importance": mem["metadata"].get("importance"),
                        "confidence": mem["metadata"].get("confidence", 0.5),
                        "sentiment": mem["metadata"].get("sentiment"),
                        "timestamp": mem["metadata"].get("timestamp", ""),
                        "created_at": mem["metadata"].get("created_at"),
                        "user_id": mem["metadata"].get("user_id")
                    })

            # Sort
            if sort_by == "importance":
                tagged.sort(key=lambda x: x.get("importance") or 0, reverse=True)
            else:  # created_at (default)
                tagged.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            return tagged[:limit]
        except Exception as e:
            logger.error(f"Failed to search by tag: {e}")
            return []

    async def update_memory_importance(
        self,
        memory_id: str,
        importance: int
    ) -> Dict[str, Any]:
        """Set importance score for a memory"""
        try:
            if not 1 <= importance <= 10:
                return {"success": False, "error": "Importance must be between 1 and 10"}

            # Fetch the memory
            result = self.index.fetch(ids=[memory_id])

            if not result or "vectors" not in result or memory_id not in result["vectors"]:
                logger.error(f"Memory {memory_id} not found")
                return {"success": False, "error": "Memory not found"}

            mem = result["vectors"][memory_id]
            metadata = mem["metadata"]

            # Update importance
            metadata["importance"] = importance
            metadata["updated_at"] = datetime.now().isoformat()

            # Upsert back to Pinecone
            self.index.upsert(
                vectors=[(memory_id, mem["values"], metadata)]
            )

            logger.info(f"Updated importance for memory {memory_id}: {importance}")

            return {
                "success": True,
                "memory_id": memory_id,
                "importance": importance
            }

        except Exception as e:
            logger.error(f"Failed to update importance: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def pin_memory(self, memory_id: str) -> Dict[str, Any]:
        """Pin a memory to always include it"""
        try:
            # Fetch the memory
            result = self.index.fetch(ids=[memory_id])

            if not result or "vectors" not in result or memory_id not in result["vectors"]:
                logger.error(f"Memory {memory_id} not found")
                return {"success": False, "error": "Memory not found"}

            mem = result["vectors"][memory_id]
            metadata = mem["metadata"]

            # Pin it
            metadata["pinned"] = True
            metadata["updated_at"] = datetime.now().isoformat()

            # Upsert back to Pinecone
            self.index.upsert(
                vectors=[(memory_id, mem["values"], metadata)]
            )

            logger.info(f"Pinned memory {memory_id}")

            return {
                "success": True,
                "memory_id": memory_id,
                "pinned": True
            }

        except Exception as e:
            logger.error(f"Failed to pin memory: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def archive_memory(self, memory_id: str) -> Dict[str, Any]:
        """Archive a memory (soft delete)"""
        try:
            # Fetch the memory
            result = self.index.fetch(ids=[memory_id])

            if not result or "vectors" not in result or memory_id not in result["vectors"]:
                logger.error(f"Memory {memory_id} not found")
                return {"success": False, "error": "Memory not found"}

            mem = result["vectors"][memory_id]
            metadata = mem["metadata"]

            # Archive it
            metadata["archived"] = True
            metadata["updated_at"] = datetime.now().isoformat()

            # Upsert back to Pinecone
            self.index.upsert(
                vectors=[(memory_id, mem["values"], metadata)]
            )

            logger.info(f"Archived memory {memory_id}")

            return {
                "success": True,
                "memory_id": memory_id,
                "archived": True
            }

        except Exception as e:
            logger.error(f"Failed to archive memory: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _extract_memories(
        self,
        user_message: str,
        recent_history: List[str]
    ) -> List[Dict[str, Any]]:
        """Use LLM to extract memories from text"""

        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")

        system_prompt = f"""You are a memory extraction system. Extract ONLY user-specific facts, preferences, goals, relationships, and persistent information.

CURRENT DATE: {current_date}

Include temporal context when relevant (e.g., "As of {current_date}, user prefers X").

CRITICAL OUTPUT REQUIREMENTS:
1. Your ENTIRE response MUST be ONLY a valid JSON array: [...]
2. NO text before or after the JSON array
3. NO markdown code blocks (no ```json)
4. NO explanations or notes
5. Return [] if no user-specific memories found

Each memory object MUST have:
- "content": the extracted user-specific fact (string)
- "tags": array of relevant tags (e.g., ["homelab", "technical", "family"])
- "confidence": score from 0.0 to 1.0 indicating certainty
- "category": ONE category from the list below (REQUIRED)
- "importance": integer from 1-10 indicating significance (REQUIRED)
- "sentiment": ONE of: positive, negative, neutral, mixed (REQUIRED)

CATEGORIZATION (choose ONE):
- "achievement": Completed tasks, successes, victories, solved problems
- "frustration": Problems, failures, issues, roadblocks
- "idea": Thoughts, plans, possibilities, future considerations
- "fact": Factual information about user (name, location, possessions)
- "event": Things that happened, meetings, activities
- "conversation": Discussion topics, things mentioned
- "relationship": Information about people, connections, family
- "technical": Code, systems, infrastructure, tools, technology
- "personal": Family, hobbies, interests, daily life
- "misc": Everything else that doesn't fit above

IMPORTANCE SCORING (1-10):
- 1-3: Low - Minor details, trivial facts, passing mentions
- 4-6: Medium - Useful information, regular preferences, typical activities
- 7-8: High - Important facts, strong preferences, key relationships, significant events
- 9-10: Critical - Core identity, mission-critical information, deeply held values

SENTIMENT ANALYSIS:
- "positive": Achievements, good news, preferences, successes, joy
- "negative": Frustrations, problems, dislikes, failures, anger
- "neutral": Facts, observations, routine information
- "mixed": Complex feelings, both good and bad aspects

EXTRACT:
- Explicit user preferences ("I love X", "My favorite is Y")
- Identity details (name, location, profession, age)
- Goals and aspirations
- Relationships (family, friends, colleagues)
- Possessions (things owned or desired)
- Behavioral patterns and interests
- Achievements and victories
- Frustrations and problems

DO NOT EXTRACT:
- General knowledge or trivia
- Temporary thoughts or questions
- Information about the AI
- Meta-commentary about remembering

EXAMPLE OUTPUT:
[
  {{
    "content": "User fixed BADBUNNY PSU issue after 2 hours of troubleshooting",
    "tags": ["homelab", "technical", "hardware", "BADBUNNY"],
    "category": "achievement",
    "importance": 8,
    "sentiment": "positive",
    "confidence": 0.95
  }},
  {{
    "content": "User has a cat named Whiskers",
    "tags": ["relationship", "possession", "pets"],
    "category": "fact",
    "importance": 6,
    "sentiment": "neutral",
    "confidence": 0.9
  }},
  {{
    "content": "User frustrated with network connectivity issues",
    "tags": ["technical", "network", "problem"],
    "category": "frustration",
    "importance": 7,
    "sentiment": "negative",
    "confidence": 0.85
  }}
]

If no user-specific memories are found, return: []"""

        context = ""
        if recent_history:
            context = "Recent conversation:\n"
            for msg in recent_history[-3:]:
                context += f"- {msg}\n"
            context += "\n"
        
        user_prompt = f"""{context}Analyze this user message and extract memories:

"{user_message}"

Return ONLY the JSON array. No other text."""

        logger.debug("Calling LLM for memory extraction")
        response = await self.llm.query(system_prompt, user_prompt, temperature=0.1)
        
        if not response:
            logger.error("LLM extraction returned no response")
            return []
        
        logger.debug(f"LLM response length: {len(response)} chars")
        
        try:
            cleaned = re.sub(r'```(?:json)?\s*', '', response)
            cleaned = re.sub(r'\s*```', '', cleaned)
            cleaned = cleaned.strip()
            
            logger.debug(f"Cleaned response: {cleaned[:200]}...")
            
            memories = json.loads(cleaned)
            
            if isinstance(memories, dict):
                if "memories" in memories:
                    memories = memories["memories"]
                elif len(memories) == 1:
                    memories = list(memories.values())[0]
            
            if not isinstance(memories, list):
                logger.error(f"LLM returned invalid format: {type(memories)}")
                return []
            
            valid_memories = []
            for mem in memories:
                if not isinstance(mem, dict):
                    logger.warning(f"Skipping non-dict memory: {mem}")
                    continue

                if "content" not in mem or not mem["content"]:
                    logger.warning(f"Skipping memory without content: {mem}")
                    continue

                # Ensure tags is a list
                if "tags" not in mem:
                    mem["tags"] = []
                elif not isinstance(mem["tags"], list):
                    mem["tags"] = [str(mem["tags"])]

                # Set defaults for existing fields
                if "confidence" not in mem:
                    mem["confidence"] = 0.5

                # Set defaults for v2.0 fields
                if "category" not in mem:
                    mem["category"] = "misc"  # Default category

                if "importance" not in mem:
                    mem["importance"] = 5  # Default to medium importance

                if "sentiment" not in mem:
                    mem["sentiment"] = "neutral"  # Default sentiment

                # Calculate word count
                mem["word_count"] = len(mem["content"].split())

                valid_memories.append(mem)
            
            logger.info(f"Validated {len(valid_memories)}/{len(memories)} extracted memories")
            return valid_memories
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Failed to parse: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing memories: {e}", exc_info=True)
            return []
    
    async def _deduplicate(
        self,
        memories: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Smart deduplication with memory updating"""
        if not memories:
            return []

        logger.debug(f"Processing {len(memories)} memories for deduplication/update")

        existing = await self._get_all_memories(user_id=user_id)
        logger.debug(f"Checking against {len(existing)} existing memories")

        unique = []
        update_threshold = 0.85  # Lower than dedup threshold - allows for updates

        for new_mem in memories:
            content = new_mem.get("content", "").strip()
            if not content:
                continue

            new_emb = await self.embedder.get_embedding(content)
            if new_emb is None:
                logger.warning(f"Failed to generate embedding for: {content[:50]}...")
                continue

            is_duplicate = False
            should_update = False
            similar_memory = None
            max_similarity = 0.0

            for exist_mem in existing:
                exist_emb = np.array(exist_mem["values"], dtype=np.float32)
                similarity = float(np.dot(new_emb, exist_emb))

                if similarity > max_similarity:
                    max_similarity = similarity
                    similar_memory = exist_mem

                # Exact duplicate - skip completely
                if similarity >= settings.DEDUP_THRESHOLD:
                    logger.debug(
                        f"Duplicate detected (sim={similarity:.3f}): "
                        f"{content[:50]}... == {exist_mem['metadata'].get('content', '')[:50]}..."
                    )
                    is_duplicate = True
                    break

                # Similar but different - might be an update
                elif similarity >= update_threshold:
                    logger.info(
                        f"Similar memory found (sim={similarity:.3f}), treating as update: "
                        f"NEW: {content[:50]}... | OLD: {exist_mem['metadata'].get('content', '')[:50]}..."
                    )
                    should_update = True
                    similar_memory = exist_mem
                    break

            if is_duplicate:
                continue  # Skip exact duplicates
            elif should_update and similar_memory:
                # Delete old version and store new one
                logger.info(f"Updating memory {similar_memory['id']}")
                await self.delete(similar_memory['id'])
                new_mem["embedding"] = new_emb
                new_mem["updated_from"] = similar_memory['id']
                unique.append(new_mem)
            else:
                # Completely new memory
                logger.debug(f"New memory (max_sim={max_similarity:.3f}): {content[:50]}...")
                new_mem["embedding"] = new_emb
                unique.append(new_mem)

        logger.info(f"Deduplication complete: {len(unique)}/{len(memories)} to store (includes updates)")
        return unique
    
    async def _store_memory(
        self,
        memory: Dict[str, Any],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Store memory in Pinecone"""
        content = memory.get("content", "").strip()
        embedding = memory.get("embedding")

        if not content:
            logger.error("Cannot store memory without content")
            return None

        if embedding is None:
            embedding = await self.embedder.get_embedding(content)

        if embedding is None:
            logger.error(f"Failed to generate embedding for storage: {content[:50]}...")
            return None

        timestamp = int(time.time() * 1000)
        content_hash = abs(hash(content)) % 10000
        memory_id = f"mem_{timestamp}_{content_hash}"

        # Build enhanced metadata for v2.0
        now = datetime.now().isoformat()
        tags = memory.get("tags", [])

        metadata = {
            # Core fields
            "content": content,
            "confidence": float(memory.get("confidence", 0.5)),
            "timestamp": now,
            "created_at": now,
            "updated_at": now,

            # V2.0 additions - Categorization
            "tags": tags if isinstance(tags, list) else [],  # Store as array
            "category": memory.get("category"),  # Optional: achievement, frustration, idea, fact, event, conversation, relationship, technical, personal, misc

            # V2.0 additions - Quality/Importance
            "importance": memory.get("importance"),  # Optional: 1-10 scale
            "pinned": memory.get("pinned", False),
            "archived": memory.get("archived", False),

            # V2.0 additions - Content analysis
            "word_count": memory.get("word_count"),
            "sentiment": memory.get("sentiment"),  # Optional: positive, negative, neutral, mixed

            # V2.0 additions - Temporal
            "event_date": memory.get("event_date")  # Optional: When event actually occurred
        }

        # Remove None values to keep metadata clean
        metadata = {k: v for k, v in metadata.items() if v is not None}

        # Add optional identifiers
        if user_id:
            metadata["user_id"] = user_id
        if agent_id:
            metadata["agent_id"] = agent_id
        if run_id:
            metadata["run_id"] = run_id

        try:
            self.index.upsert(
                vectors=[(memory_id, embedding.tolist(), metadata)]
            )

            logger.info(f"Stored memory [{memory_id}]: {content[:80]}...")

            return {
                "id": memory_id,
                "content": content,
                "tags": metadata.get("tags", []),
                "category": metadata.get("category"),
                "confidence": metadata.get("confidence", 0.5),
                "importance": metadata.get("importance"),
                "pinned": metadata.get("pinned", False),
                "archived": metadata.get("archived", False),
                "sentiment": metadata.get("sentiment"),
                "word_count": metadata.get("word_count"),
                "timestamp": metadata["timestamp"],
                "created_at": metadata["created_at"],
                "updated_at": metadata["updated_at"],
                "event_date": metadata.get("event_date"),
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id
            }

        except Exception as e:
            logger.error(f"Failed to store memory in Pinecone: {e}", exc_info=True)
            return None
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity with optional hybrid keyword boost"""
        logger.debug(f"Searching for: {query[:100]}... (hybrid={use_hybrid})")

        query_emb = await self.embedder.get_embedding(query)

        if query_emb is None:
            logger.error("Failed to generate query embedding")
            return []

        try:
            # Build filter
            filter_dict = {}
            if user_id:
                filter_dict["user_id"] = {"$eq": user_id}
            if agent_id:
                filter_dict["agent_id"] = {"$eq": agent_id}

            # Get more results for hybrid re-ranking
            top_k = limit * 3 if use_hybrid else limit

            results = self.index.query(
                vector=query_emb.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )

            memories = []
            query_terms = query.lower().split()

            for match in results.matches:
                tags_raw = match.metadata.get("tags", "")
                tags = tags_raw if isinstance(tags_raw, list) else (tags_raw.split(",") if tags_raw else [])

                # Filter by categories if specified
                if categories:
                    if not any(cat in tags for cat in categories):
                        continue

                content = match.metadata.get("content", "")
                semantic_score = float(match.score)

                # Hybrid scoring: boost results with keyword matches
                if use_hybrid:
                    content_lower = content.lower()
                    keyword_matches = sum(1 for term in query_terms if term in content_lower)
                    keyword_boost = 1.0 + (0.15 * keyword_matches)  # 15% boost per keyword match
                    final_score = semantic_score * keyword_boost
                else:
                    final_score = semantic_score

                memories.append({
                    "id": match.id,
                    "content": content,
                    "relevance": final_score,
                    "semantic_score": semantic_score,
                    "tags": tags,
                    "confidence": match.metadata.get("confidence", 0.5),
                    "timestamp": match.metadata.get("timestamp", ""),
                    "user_id": match.metadata.get("user_id"),
                    "agent_id": match.metadata.get("agent_id")
                })

            # Re-sort by final score if using hybrid
            if use_hybrid:
                memories.sort(key=lambda x: x["relevance"], reverse=True)

            # Return top results
            final_results = memories[:limit]
            logger.info(f"Search returned {len(final_results)} results (hybrid={use_hybrid})")
            return final_results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    async def get_relevant(
        self,
        current_message: str,
        limit: int = 5,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to current context with threshold filtering"""
        results = await self.search(
            current_message,
            limit=limit * 2,
            user_id=user_id
        )

        filtered = [
            r for r in results
            if r["relevance"] >= settings.RELEVANCE_THRESHOLD
        ]

        relevant = filtered[:limit]

        logger.info(
            f"Relevant memories: {len(relevant)}/{len(results)} "
            f"(threshold: {settings.RELEVANCE_THRESHOLD})"
        )

        return relevant
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        try:
            self.index.delete(ids=[memory_id])
            logger.info(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Delete failed for {memory_id}: {e}")
            return False

    async def batch_delete(self, memory_ids: List[str]) -> int:
        """Delete multiple memories by IDs"""
        success_count = 0
        try:
            self.index.delete(ids=memory_ids)
            success_count = len(memory_ids)
            logger.info(f"Batch deleted {success_count} memories")
        except Exception as e:
            logger.error(f"Batch delete failed: {e}")
            # Fall back to individual deletes
            for memory_id in memory_ids:
                if await self.delete(memory_id):
                    success_count += 1
        return success_count

    async def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics including v2.0 metadata breakdown"""
        try:
            # Get basic Pinecone stats
            stats = self.index.describe_index_stats()
            total_count = stats.total_vector_count

            # Get detailed metadata statistics
            all_mems = await self._get_all_memories(user_id=user_id)

            # Count by category
            category_counts = {}
            importance_counts = {i: 0 for i in range(1, 11)}  # 1-10
            sentiment_counts = {}
            pinned_count = 0
            archived_count = 0
            tag_counts = {}

            for mem in all_mems:
                metadata = mem.get("metadata", {})

                # Category
                category = metadata.get("category")
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1

                # Importance
                importance = metadata.get("importance")
                if importance and 1 <= importance <= 10:
                    importance_counts[importance] += 1

                # Sentiment
                sentiment = metadata.get("sentiment")
                if sentiment:
                    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

                # Flags
                if metadata.get("pinned"):
                    pinned_count += 1
                if metadata.get("archived"):
                    archived_count += 1

                # Tags
                tags_raw = metadata.get("tags", "")
                tags = tags_raw if isinstance(tags_raw, list) else (tags_raw.split(",") if tags_raw else [])
                for tag in tags:
                    tag = tag.strip()
                    if tag:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Top 10 tags
            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                "count": total_count,
                "dimension": stats.dimension,
                "max_memories": settings.MAX_MEMORIES,
                "utilization_pct": round((total_count / settings.MAX_MEMORIES) * 100, 1) if settings.MAX_MEMORIES > 0 else 0,

                # V2.0 statistics
                "categories": category_counts,
                "importance_distribution": {k: v for k, v in importance_counts.items() if v > 0},
                "sentiment_distribution": sentiment_counts,
                "pinned_count": pinned_count,
                "archived_count": archived_count,
                "active_count": len([m for m in all_mems if not m.get("metadata", {}).get("archived")]),
                "top_tags": [{"name": tag, "count": count} for tag, count in top_tags],
                "unique_tag_count": len(tag_counts)
            }
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}", exc_info=True)
            return {
                "count": 0,
                "dimension": 0,
                "max_memories": settings.MAX_MEMORIES,
                "error": str(e)
            }

    async def _get_all_memories(
        self,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all memories for deduplication"""
        try:
            dummy = np.zeros(self.embedding_dim, dtype=np.float32)

            # Build filter
            filter_dict = {}
            if user_id:
                filter_dict["user_id"] = {"$eq": user_id}

            results = self.index.query(
                vector=dummy.tolist(),
                top_k=settings.MAX_MEMORIES,
                include_values=True,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )

            memories = []
            for match in results.matches:
                memories.append({
                    "id": match.id,
                    "values": match.values,
                    "metadata": match.metadata
                })

            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve all memories: {e}")
            return []
    
    async def _prune_if_needed(self):
        """Prune oldest memories if over limit"""
        stats = await self.get_stats()
        count = stats["count"]

        if count <= settings.MAX_MEMORIES:
            return

        to_delete = count - settings.MAX_MEMORIES
        logger.info(f"Memory limit exceeded ({count} > {settings.MAX_MEMORIES}), pruning {to_delete} memories")

        all_mems = await self._get_all_memories()
        all_mems.sort(key=lambda x: x["metadata"].get("timestamp", ""))

        for mem in all_mems[:to_delete]:
            await self.delete(mem["id"])

        logger.info(f"Pruned {to_delete} old memories")

    async def consolidate_memories(
        self,
        user_id: Optional[str] = None,
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Consolidate related memories into coherent summaries using LLM.
        Useful for merging fragmented information into unified facts.
        """
        logger.info(f"Starting memory consolidation for user_id={user_id}, tag={tag}")

        # Get all memories or filter by tag
        all_mems = await self._get_all_memories(user_id=user_id)

        if tag:
            all_mems = [
                m for m in all_mems
                if tag in (m["metadata"].get("tags", []) if isinstance(m["metadata"].get("tags"), list) else m["metadata"].get("tags", "").split(","))
            ]

        if len(all_mems) < 3:
            logger.info("Not enough memories to consolidate (need at least 3)")
            return {"consolidated": 0, "message": "Not enough memories to consolidate"}

        # Group memories by semantic similarity
        groups = await self._group_similar_memories(all_mems)

        consolidated_count = 0
        for group in groups:
            if len(group) < 2:
                continue  # Skip single memories

            # Use LLM to consolidate group
            consolidated = await self._consolidate_group(group, user_id=user_id)
            if consolidated:
                consolidated_count += 1

        logger.info(f"Consolidated {consolidated_count} memory groups")
        return {
            "consolidated": consolidated_count,
            "message": f"Successfully consolidated {consolidated_count} memory groups"
        }

    async def _group_similar_memories(
        self,
        memories: List[Dict[str, Any]],
        similarity_threshold: float = 0.75
    ) -> List[List[Dict[str, Any]]]:
        """Group memories by semantic similarity"""
        if not memories:
            return []

        groups = []
        used = set()

        for i, mem in enumerate(memories):
            if i in used:
                continue

            group = [mem]
            mem_emb = np.array(mem["values"], dtype=np.float32)

            # Find similar memories
            for j, other_mem in enumerate(memories):
                if j <= i or j in used:
                    continue

                other_emb = np.array(other_mem["values"], dtype=np.float32)
                similarity = float(np.dot(mem_emb, other_emb))

                if similarity >= similarity_threshold:
                    group.append(other_mem)
                    used.add(j)

            if len(group) >= 2:  # Only keep groups with multiple memories
                groups.append(group)
                used.add(i)

        logger.debug(f"Grouped {len(memories)} memories into {len(groups)} groups")
        return groups

    async def _consolidate_group(
        self,
        group: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """Use LLM to consolidate a group of related memories"""
        memory_texts = [m["metadata"].get("content", "") for m in group]

        system_prompt = """You are a memory consolidation system. Given multiple related memory fragments, create a single coherent, comprehensive memory that captures all the information.

RULES:
1. Combine all relevant information into one clear statement
2. Resolve any contradictions (prefer more recent information)
3. Keep the consolidated memory concise but complete
4. Maintain factual accuracy
5. Return ONLY the consolidated memory text, nothing else"""

        user_prompt = f"""Consolidate these related memories into one comprehensive memory:

{chr(10).join(f"{i+1}. {text}" for i, text in enumerate(memory_texts))}

Return only the consolidated memory:"""

        try:
            consolidated_text = await self.llm.query(system_prompt, user_prompt, temperature=0.2)

            if not consolidated_text:
                logger.warning("LLM returned no consolidation")
                return None

            # Delete old memories
            for mem in group:
                await self.delete(mem["id"])

            # Store consolidated memory
            consolidated_mem = {
                "content": consolidated_text.strip(),
                "tags": list(set(
                    tag for mem in group
                    for tag in (mem["metadata"].get("tags", []) if isinstance(mem["metadata"].get("tags"), list) else mem["metadata"].get("tags", "").split(","))
                    if tag
                )),
                "confidence": max(
                    mem["metadata"].get("confidence", 0.5)
                    for mem in group
                )
            }

            result = await self._store_memory(
                consolidated_mem,
                user_id=user_id
            )

            if result:
                logger.info(f"Consolidated {len(group)} memories into: {consolidated_text[:80]}...")
                return result["id"]

            return None

        except Exception as e:
            logger.error(f"Failed to consolidate group: {e}", exc_info=True)
            return None
