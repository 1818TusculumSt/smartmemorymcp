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
        else:
            self.embedding_dim = 1536
            logger.info(f"Using embedding dimension {self.embedding_dim} for API provider")
    
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
    
    async def get_recent(self, limit: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get most recent memories"""
        try:
            all_mems = await self._get_all_memories(user_id=user_id)
            # Sort by timestamp descending
            all_mems.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)

            recent = []
            for mem in all_mems[:limit]:
                tags = mem["metadata"].get("tags", "").split(",") if mem["metadata"].get("tags") else []
                recent.append({
                    "id": mem["id"],
                    "content": mem["metadata"].get("content", ""),
                    "tags": tags,
                    "confidence": mem["metadata"].get("confidence", 0.5),
                    "timestamp": mem["metadata"].get("timestamp", ""),
                    "user_id": mem["metadata"].get("user_id")
                })

            return recent
        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []

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
- "tags": array of relevant tags from: identity, preference, goal, relationship, possession, behavior
- "confidence": score from 0.0 to 1.0 indicating certainty

EXTRACT:
- Explicit user preferences ("I love X", "My favorite is Y")
- Identity details (name, location, profession, age)
- Goals and aspirations
- Relationships (family, friends, colleagues)
- Possessions (things owned or desired)
- Behavioral patterns and interests

DO NOT EXTRACT:
- General knowledge or trivia
- Temporary thoughts or questions
- Information about the AI
- Meta-commentary about remembering

EXAMPLE OUTPUT:
[
  {{
    "content": "User has a cat named Whiskers",
    "tags": ["relationship", "possession"],
    "confidence": 0.9
  }},
  {{
    "content": "User prefers working remotely",
    "tags": ["preference", "behavior"],
    "confidence": 0.75
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
                
                if "tags" not in mem:
                    mem["tags"] = ["behavior"]
                elif not isinstance(mem["tags"], list):
                    mem["tags"] = [str(mem["tags"])]
                
                if "confidence" not in mem:
                    mem["confidence"] = 0.5
                
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

        metadata = {
            "content": content,
            "tags": ",".join(memory.get("tags", [])),
            "confidence": float(memory.get("confidence", 0.5)),
            "timestamp": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }

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
                "tags": memory.get("tags", []),
                "confidence": memory.get("confidence", 0.5),
                "timestamp": metadata["timestamp"],
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
                tags = match.metadata.get("tags", "").split(",") if match.metadata.get("tags") else []

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

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "count": stats.total_vector_count,
                "dimension": stats.dimension,
                "max_memories": settings.MAX_MEMORIES
            }
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {
                "count": 0,
                "dimension": 0,
                "max_memories": settings.MAX_MEMORIES
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
                if tag in m["metadata"].get("tags", "").split(",")
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
                    for tag in mem["metadata"].get("tags", "").split(",")
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
