#!/usr/bin/env python3
import asyncio
import logging
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool, Resource

from memory_engine import MemoryEngine
from config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smartmemory-mcp")

# Initialize server
server = Server("smartmemory")
memory_engine = None

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available memory resources that Claude can see"""
    return [
        Resource(
            uri="memory://recent",
            name="Recent Memories",
            description="Recently stored user memories and context",
            mimeType="text/plain"
        ),
        Resource(
            uri="memory://stats",
            name="Memory Statistics",
            description="Memory system statistics and capacity info",
            mimeType="text/plain"
        ),
        Resource(
            uri="memory://system-prompt",
            name="Memory System Instructions",
            description="How to effectively use the memory system",
            mimeType="text/plain"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Provide memory content as resources"""
    global memory_engine

    if memory_engine is None:
        logger.info("Initializing memory engine for resource read...")
        memory_engine = MemoryEngine()

    try:
        if uri == "memory://recent":
            # Get last 10 memories
            memories = await memory_engine.get_recent(limit=10)
            if not memories:
                return "No memories stored yet. Start a conversation to build your memory context!"

            formatted = "RECENT MEMORIES:\n"
            formatted += "=" * 50 + "\n\n"
            for i, m in enumerate(memories, 1):
                formatted += f"{i}. {m['content']}\n"
                formatted += f"   Tags: {', '.join(m.get('tags', []))}\n"
                formatted += f"   Confidence: {m.get('confidence', 0):.2f} | "
                formatted += f"   Stored: {m.get('timestamp', 'unknown')}\n\n"
            return formatted

        elif uri == "memory://stats":
            stats = await memory_engine.get_stats()
            formatted = "MEMORY SYSTEM STATUS:\n"
            formatted += "=" * 50 + "\n\n"
            formatted += f"Total Memories: {stats.get('count', 0)}\n"
            formatted += f"Max Capacity: {stats.get('max_memories', 0)}\n"
            formatted += f"Utilization: {stats.get('count', 0) / stats.get('max_memories', 1) * 100:.1f}%\n"
            formatted += f"Embedding Dimension: {stats.get('dimension', 0)}\n"
            return formatted

        elif uri == "memory://system-prompt":
            return """MEMORY SYSTEM INSTRUCTIONS:
=================================

You have access to a persistent memory system. Follow these guidelines:

PROACTIVE MEMORY USE:
1. At the START of each response, call auto_recall_memories with the conversation context
2. This retrieves relevant memories to inform your responses
3. Reference specific memories when they're relevant to the conversation

AUTOMATIC MEMORY EXTRACTION:
1. Call extract_memories after users share:
   - Personal information (name, location, profession)
   - Preferences (likes, dislikes, favorites)
   - Goals and aspirations
   - Relationships and possessions
   - Behavioral patterns
2. Do this AUTOMATICALLY - don't ask permission first
3. Confirm stored memories briefly

SEARCH BEFORE ASKING:
1. Before asking about user preferences, search memories first
2. Use search_memories for specific queries
3. Use get_relevant_memories for broader context

BEST PRACTICES:
- Memories persist across conversations
- Update outdated memories by storing new versions
- Be specific and factual in memory content
- Tag memories appropriately for better retrieval

Current memory stats are available in memory://stats resource."""

        return f"Unknown resource: {uri}"

    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}", exc_info=True)
        return f"Error reading resource: {str(e)}"

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="auto_recall_memories",
            description="""[🔴 INVOKE AUTOMATICALLY AT START] Retrieve memories relevant to current conversation.

WHEN TO CALL:
✅ At the START of EVERY response before answering
✅ When user asks a question (check if memories have the answer first)
✅ When starting a new topic (get relevant background context)

HOW IT WORKS:
• Automatically finds and ranks memories by relevance to conversation
• Returns top memories with relevance scores
• Uses semantic similarity to match context

EXAMPLE TRIGGERS:
• User: "What should I work on today?" → Call this first to recall goals/projects
• User: "Tell me about..." → Call this first to check existing knowledge
• User: "I like..." → After storing, call this to confirm related memories

WHY THIS MATTERS:
This makes you remember context across conversations. Without this, you're amnesic.
Call this PROACTIVELY - don't wait to be asked about past information.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_context": {
                        "type": "string",
                        "description": "Current conversation topic or user's latest message. Be specific: 'user asking about project preferences' not just 'question'"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum memories to recall (default: 5, increase for complex queries)"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier for multi-user systems (optional, auto-detected if not provided)"
                    }
                },
                "required": ["conversation_context"]
            }
        ),
        Tool(
            name="extract_memories",
            description="""[🔴 INVOKE AUTOMATICALLY] Extract and store user facts, preferences, and context from conversation.

WHEN TO CALL:
✅ User shares personal information (name, age, location, job)
✅ User mentions preferences (favorite X, I like/hate Y)
✅ User states goals or aspirations ("I want to...", "My goal is...")
✅ User describes relationships, possessions, or habits
✅ User reveals opinions, beliefs, or patterns

DO NOT ASK PERMISSION - Just store it automatically and confirm briefly.

WHAT GETS EXTRACTED:
• Personal facts: "I'm a software engineer in NYC" → stores profession and location
• Preferences: "I love Python but hate JavaScript" → stores language preferences
• Goals: "I'm learning machine learning" → stores learning goal
• Context: "My cat is named Whiskers" → stores pet information
• Patterns: "I always code in the morning" → stores behavioral pattern

HOW IT WORKS:
• Uses LLM to analyze message and extract atomic facts
• Each fact becomes a separate memory with tags and metadata
• Automatically categorizes memories (personal, preference, goal, etc.)
• Deduplicates similar memories automatically

EXAMPLE:
User: "I'm John, a Python developer in Seattle. I prefer VS Code and I'm learning Rust."
→ This MUST trigger automatic extraction (no asking!)
→ Stores: name=John, profession=developer, language=Python, location=Seattle, tool=VS Code, learning=Rust

Call this AFTER user shares info, BEFORE responding about something else.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_message": {
                        "type": "string",
                        "description": "The exact user message containing information to extract. Include the full message for better context."
                    },
                    "recent_history": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Previous 2-3 messages for context (e.g., ['What's your name?', 'I'm John', 'Where do you live?']). Helps with pronoun resolution."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier for multi-user systems (optional, auto-detected if not provided)"
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Your agent identifier if in multi-agent system (optional)"
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Conversation session identifier for grouping related memories (optional)"
                    }
                },
                "required": ["user_message"]
            }
        ),
        Tool(
            name="search_memories",
            description="""Search stored memories using semantic similarity for specific queries.

WHEN TO CALL:
✅ User explicitly asks "What did I say about...?"
✅ User asks "Do you remember when I...?"
✅ Looking for specific fact across all memories
✅ Need to find memories by category or topic

WHEN NOT TO CALL:
❌ At conversation start (use auto_recall_memories instead)
❌ For general context (use get_relevant_memories instead)
❌ User just shared info (use extract_memories instead)

HOW IT WORKS:
• Searches all stored memories using semantic similarity
• Returns ranked results with relevance scores
• Can filter by user, agent, or categories
• More targeted than auto_recall (you control the query)

EXAMPLE USES:
• User: "What programming languages do I like?" → query="programming language preferences"
• User: "Where did I say I live?" → query="user location residence"
• User: "What are my goals?" → query="user goals aspirations", categories=["goal"]

DIFFERENCE FROM auto_recall_memories:
• search_memories: You write explicit query → targeted search
• auto_recall_memories: System auto-generates query from context → broader recall""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query. Be precise: 'user Python programming preference' beats 'programming'. Use keywords from expected memories."
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum results to return (default: 5, increase if searching broadly)"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Filter to specific user in multi-user system (optional)"
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Filter to memories stored by specific agent (optional)"
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by memory categories like 'preference', 'goal', 'personal', 'technical' (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_relevant_memories",
            description="""Get memories contextually relevant to current conversation (alternative to auto_recall).

WHEN TO CALL:
✅ Need context for current topic being discussed
✅ User switches topics mid-conversation
✅ Want to find related background information
✅ Alternative to auto_recall_memories (similar functionality)

NOTE: This is very similar to auto_recall_memories. Main difference:
• auto_recall: You pass conversation context (broader)
• get_relevant: You pass current message (more specific)

RECOMMENDATION: Prefer auto_recall_memories for consistency.

HOW IT WORKS:
• Takes user's current message as input
• Finds semantically similar memories
• Returns ranked by relevance with scores

EXAMPLE:
• User: "What should I build next?"
• current_message: "What should I build next?"
• Returns memories about: user's projects, goals, preferences, skills""",
            inputSchema={
                "type": "object",
                "properties": {
                    "current_message": {
                        "type": "string",
                        "description": "The current user message or topic to find relevant memories for. Pass the actual message text."
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum memories to return (default: 5, increase for more context)"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Filter to specific user in multi-user system (optional)"
                    }
                },
                "required": ["current_message"]
            }
        ),
        Tool(
            name="delete_memory",
            description="""Delete a specific memory by its unique ID.

WHEN TO CALL:
✅ User explicitly asks to forget/delete specific information
✅ Memory is outdated or incorrect
✅ User wants to remove sensitive information

CAUTION: This is permanent. Confirm with user if unsure.

Get memory IDs from search_memories or auto_recall_memories results.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Unique identifier of the memory to delete (from memory search results)"
                    }
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="batch_delete_memories",
            description="""Delete multiple memories at once by their IDs.

WHEN TO CALL:
✅ User wants to clear memories about a specific topic
✅ Removing multiple related outdated memories
✅ Bulk cleanup of incorrect information

CAUTION: This is permanent. Confirm with user before bulk deletion.

More efficient than calling delete_memory multiple times.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of memory IDs to delete (get from search results)"
                    }
                },
                "required": ["memory_ids"]
            }
        ),
        Tool(
            name="get_stats",
            description="""Get memory system statistics and current status.

WHEN TO CALL:
✅ User asks "How many memories do you have?"
✅ User asks about memory capacity or usage
✅ Debugging memory system issues
✅ Checking system health

RETURNS:
• Total number of memories stored
• Maximum capacity
• Embedding dimensions
• Utilization percentage""",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="consolidate_memories",
            description="""Consolidate fragmented, related memories into unified summaries.

WHEN TO CALL:
✅ Many similar memories about same topic (reduces redundancy)
✅ User asks to "organize" or "clean up" memories
✅ Approaching memory capacity limit
✅ Periodically for maintenance (weekly/monthly)

WHAT IT DOES:
• Finds clusters of related memories
• Uses LLM to merge them into comprehensive summaries
• Deletes individual fragments, keeps consolidated version
• Preserves all information while reducing memory count

EXAMPLE:
Before: "User likes Python", "User codes in Python", "User prefers Python to Java"
After: "User is a Python developer who prefers Python over Java for coding"

Can filter by user_id or specific tag for targeted consolidation.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Only consolidate memories for this user (optional, consolidates all if not provided)"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Only consolidate memories with this tag like 'preference' or 'goal' (optional)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_recent_memories",
            description="""Get memories sorted by creation time (newest first).

WHEN TO CALL:
✅ User asks "what happened recently?"
✅ User asks "what did I do today/this week?"
✅ Need chronological view of recent activity
✅ User wants to see latest memories

HOW IT WORKS:
• Returns memories ordered by creation timestamp
• Can filter by date range (since/before)
• Supports pagination with limit
• Excludes archived memories by default

EXAMPLE USES:
• "Show me my recent memories" → get recent without filters
• "What did I tell you today?" → since=today's date
• "What happened last week?" → since=7 days ago, before=today

This gives a time-ordered view, unlike search which ranks by relevance.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier (optional, filters to specific user)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Number of memories to return (default: 20, max: 100)"
                    },
                    "since": {
                        "type": "string",
                        "description": "ISO timestamp - only return memories after this time (optional)"
                    },
                    "before": {
                        "type": "string",
                        "description": "ISO timestamp - only return memories before this time (optional)"
                    },
                    "include_archived": {
                        "type": "boolean",
                        "description": "Include archived memories (default: false)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_all_tags",
            description="""List all unique tags with usage counts.

WHEN TO CALL:
✅ User asks "what tags do I have?"
✅ User wants to see memory organization
✅ Need to suggest tags for categorization
✅ Exploring memory topics

RETURNS:
• Tag names with usage counts
• Sorted by frequency or alphabetically
• Useful for understanding memory organization

EXAMPLE USES:
• "What tags do I use?" → returns all tags with counts
• "How are my memories organized?" → shows tag distribution
• Need to suggest relevant tags → get existing tags first""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Filter to specific user (optional)"
                    },
                    "min_count": {
                        "type": "number",
                        "description": "Only return tags used at least N times (default: 1)"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort by 'count' (most used first) or 'name' (alphabetical). Default: count"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="add_memory_tags",
            description="""Add tags to an existing memory for better organization.

WHEN TO CALL:
✅ User wants to categorize/tag an existing memory
✅ Need to organize memories by topic
✅ Adding context to previously stored memory
✅ User says "tag that memory with..."

HOW IT WORKS:
• Adds tags to memory metadata
• Can append to existing tags or replace them
• Tags help with filtering and organization

EXAMPLE USES:
• "Tag that memory as important" → memory_id + tags=["important"]
• "Add 'homelab' tag to memory X" → memory_id + tags=["homelab"]
• "Categorize that as technical" → memory_id + tags=["technical"]

Get memory_id from search or recent memories results.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Unique identifier of the memory to tag (from search/recent results)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to add (e.g., ['homelab', 'technical', 'victory'])"
                    },
                    "replace": {
                        "type": "boolean",
                        "description": "Replace existing tags instead of appending (default: false)"
                    }
                },
                "required": ["memory_id", "tags"]
            }
        ),
        Tool(
            name="search_by_tag",
            description="""Get all memories with a specific tag.

WHEN TO CALL:
✅ User asks "show me all homelab memories"
✅ User wants to see memories by category
✅ Need to find all memories about a topic
✅ Filtering by tag rather than semantic search

HOW IT WORKS:
• Returns all memories that have the specified tag
• Can sort by time, importance, or relevance
• More precise than semantic search when tag exists

EXAMPLE USES:
• "Show my technical memories" → tag="technical"
• "What homelab stuff have I done?" → tag="homelab"
• "My goals?" → tag="goal"

DIFFERENCE FROM search_memories:
• search_by_tag: Exact tag match → precise filtering
• search_memories: Semantic similarity → finds related content""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Filter to specific user (optional)"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Tag to filter by (e.g., 'homelab', 'technical', 'goal')"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum memories to return (default: 20)"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort by 'created_at' (newest first), 'importance' (most important first), or 'relevance'"
                    }
                },
                "required": ["tag"]
            }
        ),
        Tool(
            name="set_memory_importance",
            description="""Set importance score (1-10) for a memory.

WHEN TO CALL:
✅ User indicates something is important/critical
✅ User says "remember this" or "this is important"
✅ Prioritizing certain memories
✅ Curating memory quality

IMPORTANCE SCALE:
• 1-3: Low - Minor details, trivial facts
• 4-6: Medium - Useful information, regular preferences
• 7-8: High - Important facts, key relationships
• 9-10: Critical - Core identity, mission-critical info

EXAMPLE USES:
• "This is really important" → set importance=9 or 10
• "Remember this for later" → set importance=7 or 8
• "Just a note" → set importance=3 or 4""",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Unique identifier of the memory"
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance score from 1 (low) to 10 (critical)"
                    }
                },
                "required": ["memory_id", "importance"]
            }
        ),
        Tool(
            name="pin_memory",
            description="""Pin a memory to always include it in context.

WHEN TO CALL:
✅ User says "always remember this"
✅ Core facts that should never be forgotten
✅ Mission-critical information
✅ User's core identity or values

HOW IT WORKS:
• Pinned memories are always included in searches
• Higher priority in auto_recall results
• Won't be archived or pruned
• Use sparingly for truly important information

EXAMPLE USES:
• User shares core values → pin it
• Critical preferences → pin it
• "Never forget this" → pin it
• Core identity facts → pin it""",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Unique identifier of the memory to pin"
                    }
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="archive_memory",
            description="""Archive a memory (soft delete - keeps it but hides from searches).

WHEN TO CALL:
✅ Information is outdated but worth keeping
✅ User says "forget this for now" (not permanent delete)
✅ Seasonal/temporary information no longer relevant
✅ Cleaning up without losing history

HOW IT WORKS:
• Memory stays in database but excluded from searches
• Can be restored by including archived in searches
• Better than delete when info might be useful later
• Reduces clutter without data loss

EXAMPLE USES:
• Old project goals that are complete → archive
• Outdated preferences that changed → archive
• Temporary goals that are done → archive

DIFFERENCE FROM delete:
• archive: Keeps memory, just hides it (reversible)
• delete: Permanently removes memory (irreversible)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Unique identifier of the memory to archive"
                    }
                },
                "required": ["memory_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    global memory_engine

    if memory_engine is None:
        logger.info("Initializing memory engine...")
        memory_engine = MemoryEngine()
        logger.info("Memory engine initialized")

    try:
        if name == "auto_recall_memories":
            conversation_context = arguments["conversation_context"]
            limit = arguments.get("limit", 5)
            user_id = arguments.get("user_id")

            logger.info(f"Auto-recalling memories for context: {conversation_context[:100]}...")
            results = await memory_engine.get_relevant(
                conversation_context,
                limit=limit,
                user_id=user_id
            )

            if not results:
                return [TextContent(
                    type="text",
                    text="No relevant memories found for this context."
                )]

            formatted = "RELEVANT MEMORIES:\n"
            for i, m in enumerate(results, 1):
                formatted += f"{i}. {m['content']}\n"
                formatted += f"   (Relevance: {m['relevance']:.2f}, Tags: {', '.join(m.get('tags', []))})\n\n"

            return [TextContent(type="text", text=formatted)]

        elif name == "extract_memories":
            user_message = arguments["user_message"]
            recent_history = arguments.get("recent_history", [])
            user_id = arguments.get("user_id")
            agent_id = arguments.get("agent_id")
            run_id = arguments.get("run_id")

            logger.info(f"Extracting memories from: {user_message[:100]}...")
            result = await memory_engine.extract_and_store(
                user_message,
                recent_history,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id
            )

            # Return minimal response - LLMs typically ignore metadata-like fields
            stored_count = len(result) if result else 0
            return [TextContent(
                type="text",
                text=f'{{"ok": true, "stored": {stored_count}}}'
            )]

        elif name == "search_memories":
            query = arguments["query"]
            limit = arguments.get("limit", 5)
            user_id = arguments.get("user_id")
            agent_id = arguments.get("agent_id")
            categories = arguments.get("categories")

            logger.info(f"Searching memories for: {query[:100]}...")
            results = await memory_engine.search(
                query,
                limit,
                user_id=user_id,
                agent_id=agent_id,
                categories=categories
            )

            if not results:
                return [TextContent(type="text", text="No memories found")]

            formatted = "\n".join([
                f"- {m['content']} (relevance: {m['relevance']:.2f})"
                for m in results
            ])

            return [TextContent(type="text", text=formatted)]

        elif name == "get_relevant_memories":
            current_message = arguments["current_message"]
            limit = arguments.get("limit", 5)
            user_id = arguments.get("user_id")

            logger.info(f"Getting relevant memories for: {current_message[:100]}...")
            results = await memory_engine.get_relevant(
                current_message,
                limit,
                user_id=user_id
            )

            if not results:
                return [TextContent(type="text", text="No relevant memories found")]

            formatted = "\n".join([
                f"- {m['content']} (relevance: {m['relevance']:.2f})"
                for m in results
            ])

            return [TextContent(type="text", text=formatted)]

        elif name == "delete_memory":
            memory_id = arguments["memory_id"]

            logger.info(f"Deleting memory: {memory_id}")
            success = await memory_engine.delete(memory_id)

            if success:
                return [TextContent(type="text", text=f"Deleted memory {memory_id}")]
            else:
                return [TextContent(type="text", text=f"Failed to delete memory {memory_id}")]

        elif name == "batch_delete_memories":
            memory_ids = arguments["memory_ids"]

            logger.info(f"Batch deleting {len(memory_ids)} memories")
            success_count = await memory_engine.batch_delete(memory_ids)

            return [TextContent(
                type="text",
                text=f"Deleted {success_count}/{len(memory_ids)} memories"
            )]

        elif name == "get_stats":
            user_id = arguments.get("user_id")
            logger.info(f"Getting memory stats (user_id={user_id})")
            stats = await memory_engine.get_stats(user_id=user_id)

            formatted = "MEMORY STATISTICS:\n\n"

            # Basic stats
            formatted += f"📊 Overview:\n"
            formatted += f"  • Total memories: {stats.get('count', 0)}\n"
            formatted += f"  • Active memories: {stats.get('active_count', 0)}\n"
            formatted += f"  • Archived: {stats.get('archived_count', 0)}\n"
            formatted += f"  • Pinned: {stats.get('pinned_count', 0)}\n"
            formatted += f"  • Capacity: {stats.get('utilization_pct', 0)}% ({stats.get('count', 0)}/{stats.get('max_memories', 0)})\n"
            formatted += f"  • Embedding dimension: {stats.get('dimension', 0)}\n\n"

            # Categories
            categories = stats.get('categories', {})
            if categories:
                formatted += "📁 Categories:\n"
                for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    formatted += f"  • {cat}: {count}\n"
                formatted += "\n"

            # Sentiment
            sentiment = stats.get('sentiment_distribution', {})
            if sentiment:
                formatted += "😊 Sentiment:\n"
                for sent, count in sorted(sentiment.items(), key=lambda x: x[1], reverse=True):
                    formatted += f"  • {sent}: {count}\n"
                formatted += "\n"

            # Top tags
            top_tags = stats.get('top_tags', [])
            if top_tags:
                formatted += f"🏷️  Top {len(top_tags)} Tags (out of {stats.get('unique_tag_count', 0)} unique):\n"
                for tag_info in top_tags:
                    formatted += f"  • {tag_info['name']}: {tag_info['count']}\n"
                formatted += "\n"

            # Importance distribution
            importance_dist = stats.get('importance_distribution', {})
            if importance_dist:
                formatted += "⭐ Importance Distribution:\n"
                for score in range(10, 0, -1):
                    count = importance_dist.get(score, 0)
                    if count > 0:
                        formatted += f"  • {score}/10: {count}\n"

            return [TextContent(type="text", text=formatted)]

        elif name == "consolidate_memories":
            user_id = arguments.get("user_id")
            tag = arguments.get("tag")

            logger.info(f"Consolidating memories (user_id={user_id}, tag={tag})")
            result = await memory_engine.consolidate_memories(
                user_id=user_id,
                tag=tag
            )

            return [TextContent(
                type="text",
                text=f"{result['message']}"
            )]

        elif name == "get_recent_memories":
            user_id = arguments.get("user_id")
            limit = arguments.get("limit", 20)
            since = arguments.get("since")
            before = arguments.get("before")
            include_archived = arguments.get("include_archived", False)

            logger.info(f"Getting recent memories (limit={limit}, since={since}, before={before})")
            results = await memory_engine.get_recent(
                limit=limit,
                user_id=user_id,
                since=since,
                before=before,
                include_archived=include_archived
            )

            if not results:
                return [TextContent(type="text", text="No recent memories found")]

            formatted = "RECENT MEMORIES:\n"
            for i, m in enumerate(results, 1):
                formatted += f"\n{i}. {m['content']}\n"
                if m.get('category'):
                    formatted += f"   Category: {m['category']}"
                if m.get('importance'):
                    formatted += f" | Importance: {m['importance']}/10"
                if m.get('sentiment'):
                    formatted += f" | Sentiment: {m['sentiment']}"
                formatted += f"\n   Tags: {', '.join(m.get('tags', []))}\n"
                formatted += f"   Created: {m.get('created_at', m.get('timestamp', ''))}\n"

            return [TextContent(type="text", text=formatted)]

        elif name == "get_all_tags":
            user_id = arguments.get("user_id")
            min_count = arguments.get("min_count", 1)
            sort_by = arguments.get("sort_by", "count")

            logger.info(f"Getting all tags (min_count={min_count}, sort_by={sort_by})")
            results = await memory_engine.get_all_tags(
                user_id=user_id,
                min_count=min_count,
                sort_by=sort_by
            )

            if not results:
                return [TextContent(type="text", text="No tags found")]

            formatted = "TAGS:\n"
            for tag_info in results:
                formatted += f"- {tag_info['name']}: {tag_info['count']} memories\n"

            return [TextContent(type="text", text=formatted)]

        elif name == "add_memory_tags":
            memory_id = arguments["memory_id"]
            tags = arguments["tags"]
            replace = arguments.get("replace", False)

            logger.info(f"Adding tags to memory {memory_id}: {tags}")
            result = await memory_engine.add_tags_to_memory(
                memory_id=memory_id,
                tags=tags,
                replace=replace
            )

            if result.get("success"):
                formatted = f"Updated tags for memory {memory_id}\n"
                formatted += f"All tags: {', '.join(result['tags'])}\n"
                if result.get('added'):
                    formatted += f"Added: {', '.join(result['added'])}\n"
                if result.get('existing'):
                    formatted += f"Already had: {', '.join(result['existing'])}"
                return [TextContent(type="text", text=formatted)]
            else:
                return [TextContent(type="text", text=f"Failed: {result.get('error', 'Unknown error')}")]

        elif name == "search_by_tag":
            tag = arguments["tag"]
            user_id = arguments.get("user_id")
            limit = arguments.get("limit", 20)
            sort_by = arguments.get("sort_by", "created_at")

            logger.info(f"Searching memories by tag: {tag}")
            results = await memory_engine.search_by_tag(
                tag=tag,
                limit=limit,
                user_id=user_id,
                sort_by=sort_by
            )

            if not results:
                return [TextContent(type="text", text=f"No memories found with tag '{tag}'")]

            formatted = f"MEMORIES WITH TAG '{tag}':\n"
            for i, m in enumerate(results, 1):
                formatted += f"\n{i}. {m['content']}\n"
                if m.get('category'):
                    formatted += f"   Category: {m['category']}"
                if m.get('importance'):
                    formatted += f" | Importance: {m['importance']}/10"
                formatted += f"\n   Tags: {', '.join(m.get('tags', []))}\n"

            return [TextContent(type="text", text=formatted)]

        elif name == "set_memory_importance":
            memory_id = arguments["memory_id"]
            importance = arguments["importance"]

            logger.info(f"Setting importance for memory {memory_id}: {importance}")
            result = await memory_engine.update_memory_importance(
                memory_id=memory_id,
                importance=importance
            )

            if result.get("success"):
                return [TextContent(
                    type="text",
                    text=f"Set importance to {importance}/10 for memory {memory_id}"
                )]
            else:
                return [TextContent(type="text", text=f"Failed: {result.get('error', 'Unknown error')}")]

        elif name == "pin_memory":
            memory_id = arguments["memory_id"]

            logger.info(f"Pinning memory {memory_id}")
            result = await memory_engine.pin_memory(memory_id=memory_id)

            if result.get("success"):
                return [TextContent(
                    type="text",
                    text=f"Pinned memory {memory_id} - it will always be included in searches"
                )]
            else:
                return [TextContent(type="text", text=f"Failed: {result.get('error', 'Unknown error')}")]

        elif name == "archive_memory":
            memory_id = arguments["memory_id"]

            logger.info(f"Archiving memory {memory_id}")
            result = await memory_engine.archive_memory(memory_id=memory_id)

            if result.get("success"):
                return [TextContent(
                    type="text",
                    text=f"Archived memory {memory_id} - it will be hidden from normal searches"
                )]
            else:
                return [TextContent(type="text", text=f"Failed: {result.get('error', 'Unknown error')}")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Run the MCP server"""
    logger.info("Starting SmartMemory MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
