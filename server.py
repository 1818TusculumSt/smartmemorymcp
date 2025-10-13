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
            description="[CALL FIRST] Automatically retrieve memories relevant to the current conversation context. Call this at the START of your response to get contextual information about the user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_context": {
                        "type": "string",
                        "description": "Brief summary of current conversation topic or user's message"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum memories to recall (default: 5)"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier (optional)"
                    }
                },
                "required": ["conversation_context"]
            }
        ),
        Tool(
            name="extract_memories",
            description="[CALL AUTOMATICALLY] Extract and store user-specific facts, preferences, and context from the conversation. Call this AUTOMATICALLY when user shares personal information - don't ask permission first.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_message": {
                        "type": "string",
                        "description": "User message to extract memories from"
                    },
                    "recent_history": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recent conversation history for context (optional)"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier for memory organization (optional)"
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Agent identifier for memory organization (optional)"
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Conversation run identifier (optional)"
                    }
                },
                "required": ["user_message"]
            }
        ),
        Tool(
            name="search_memories",
            description="Search stored memories using semantic similarity. Use this for specific queries when user explicitly asks about past information.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum results to return (default: 5)"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Filter by user ID (optional)"
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Filter by agent ID (optional)"
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by memory categories (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_relevant_memories",
            description="Get memories relevant to current conversation context",
            inputSchema={
                "type": "object",
                "properties": {
                    "current_message": {
                        "type": "string",
                        "description": "Current message to find relevant memories for"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum memories to return (default: 5)"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Filter by user ID (optional)"
                    }
                },
                "required": ["current_message"]
            }
        ),
        Tool(
            name="delete_memory",
            description="Delete a specific memory by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of memory to delete"
                    }
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="batch_delete_memories",
            description="Delete multiple memories by IDs",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of memory IDs to delete"
                    }
                },
                "required": ["memory_ids"]
            }
        ),
        Tool(
            name="get_stats",
            description="Get memory system statistics and status",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="consolidate_memories",
            description="Consolidate fragmented related memories into unified summaries. Use this periodically to improve memory quality and reduce redundancy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier to consolidate memories for (optional)"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Specific tag to consolidate memories for (optional)"
                    }
                },
                "required": []
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

            if not result:
                return [TextContent(
                    type="text",
                    text="No new memories extracted"
                )]

            return [TextContent(
                type="text",
                text=f"Stored {len(result)} new memories"
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
            logger.info("Getting memory stats")
            stats = await memory_engine.get_stats()

            formatted = f"Memory Statistics:\n"
            formatted += f"- Total memories: {stats.get('count', 0)}\n"
            formatted += f"- Embedding dimension: {stats.get('dimension', 0)}\n"
            formatted += f"- Max memories: {stats.get('max_memories', 0)}"

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
