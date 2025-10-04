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
            
            logger.info(f"Extracting memories from: {user_message[:100]}...")
            result = await memory_engine.extract_and_store(user_message, recent_history)
            
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
            
            logger.info(f"Searching memories for: {query[:100]}...")
            results = await memory_engine.search(query, limit)
            
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
            
            logger.info(f"Getting relevant memories for: {current_message[:100]}...")
            results = await memory_engine.get_relevant(current_message, limit)
            
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
