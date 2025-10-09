#!/usr/bin/env python3
import asyncio
import logging
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

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

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="extract_memories",
            description="Extract and store memories from user message. Automatically identifies user-specific facts, preferences, and context.",
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
            description="Search stored memories using semantic similarity",
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
        if name == "extract_memories":
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
