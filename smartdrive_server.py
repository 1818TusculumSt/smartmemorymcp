import os
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Don't load .env - use environment vars from Claude config
# load_dotenv()  # REMOVE THIS LINE

# Initialize
app = Server("smartdrive-mcp")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(
    name=os.getenv("PINECONE_INDEX_NAME"),
    host=os.getenv("PINECONE_HOST")
)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="search_onedrive",
            description="Semantic search across OneDrive documents. Returns relevant file snippets based on your query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., '1099 tax forms', 'project proposal')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    if name == "search_onedrive":
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace="smartdrive",
            include_metadata=True
        )
        
        # Format results
        if not results.matches:
            return [TextContent(
                type="text",
                text="No matching documents found."
            )]
        
        output = f"🔍 Found {len(results.matches)} results for: '{query}'\n\n"
        
        for i, match in enumerate(results.matches, 1):
            meta = match.metadata
            score = match.score
            
            output += f"**Result {i}** (Score: {score:.3f})\n"
            output += f"📄 **File:** {meta.get('file_name', 'Unknown')}\n"
            output += f"📁 **Path:** {meta.get('file_path', 'Unknown')}\n"
            output += f"📅 **Modified:** {meta.get('modified', 'Unknown')}\n"
            output += f"📝 **Preview:**\n{meta.get('text_preview', 'No preview available')}\n\n"
            output += "---\n\n"
        
        return [TextContent(type="text", text=output)]
    
    raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    import asyncio
    import mcp.server.stdio
    
    async def main():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    
    asyncio.run(main())
