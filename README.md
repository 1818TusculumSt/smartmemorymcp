<div align="center">
  <img src="logo.png" alt="SmartMemory Logo" width="400"/>
</div>

# SmartMemory MCP Server

Auto-extracting memory system for Claude Desktop with semantic search powered by Pinecone and LLM analysis.

## Features

- **Automatic Memory Extraction**: Uses LLM to identify user-specific facts, preferences, and context
- **Semantic Search**: Vector-based similarity search using embeddings
- **Deduplication**: Prevents duplicate memories using embedding similarity
- **Configurable Providers**: Switch between local and API-based embeddings
- **Auto-Pruning**: Maintains memory limit by removing oldest entries
- **Multi-User Support**: Track memories per user with user_id filtering
- **Session Tracking**: Organize memories by conversation with agent_id and run_id
- **Batch Operations**: Delete multiple memories at once for efficiency
- **Category Filtering**: Search memories by specific categories/tags
- **Statistics & Monitoring**: Get real-time stats on memory usage

## Attribution

This project's memory management system was derived from [gramanoid's Adaptive Memory filter for Open WebUI](https://github.com/gramanoid/owui-adaptive-memory). The original filter provided the foundation for LLM-based memory extraction, embedding similarity, and semantic deduplication. This implementation refactors those concepts into a standalone MCP server / REST API with simplified architecture for broader platform compatibility.

## Prerequisites

- Python 3.11+
- Pinecone account and API key
- OpenAI API key (or compatible LLM API)
- Claude Desktop for Windows

## Installation

### 1. Clone or Download

Create the project directory and download all files:

```
C:\Users\YourUsername\mcp-smartmemory\
```

### 2. Install Python Dependencies

Open PowerShell or Command Prompt in the project directory:

```powershell
cd C:\Users\YourUsername\mcp-smartmemory
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

This will download the local embedding model (all-MiniLM-L6-v2) during installation.

### 3. Configure Claude Desktop

Edit your Claude Desktop config file:

**Windows Location:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

Add the SmartMemory server:

```json
{
  "mcpServers": {
    "smartmemory": {
      "command": "C:\\Users\\YourUsername\\mcp-smartmemory\\venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\YourUsername\\mcp-smartmemory\\server.py"],
      "env": {
        "PINECONE_API_KEY": "your-pinecone-api-key",
        "PINECONE_ENVIRONMENT": "us-east-1-aws",
        "PINECONE_INDEX_NAME": "adaptive-memory",
        "LLM_API_URL": "https://api.openai.com/v1/chat/completions",
        "LLM_API_KEY": "your-openai-api-key",
        "LLM_MODEL": "gpt-4o-mini",
        "EMBEDDING_PROVIDER": "local",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "MAX_MEMORIES": "200",
        "DEDUP_THRESHOLD": "0.95",
        "MIN_CONFIDENCE": "0.5",
        "RELEVANCE_THRESHOLD": "0.6"
      }
    }
  }
}
```

**Important:** Use double backslashes (`\\`) in Windows paths in JSON.

### 4. (Optional) Create .env File for Manual Testing

If you want to test the server manually (not through Claude Desktop), create a `.env` file in the project directory:

```
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=smartmemory
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=your-openai-api-key
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_MEMORIES=200
DEDUP_THRESHOLD=0.95
MIN_CONFIDENCE=0.5
RELEVANCE_THRESHOLD=0.6
```

**Note:** This is only needed for `python server.py` testing. Claude Desktop passes these values from the JSON config.

### 5. Restart Claude Desktop

Completely quit and restart Claude Desktop for the MCP server to load.

## Usage

The server provides 6 tools that Claude can use:

### 1. Extract Memories

Automatically extracts and stores memories from conversation:

```
User: "I just adopted a cat named Whiskers"
Claude uses extract_memories tool in background
→ Stores: "User has a cat named Whiskers"
```

**Optional Parameters:**
- `user_id`: Associate memory with specific user
- `agent_id`: Track which agent created the memory
- `run_id`: Link memory to conversation session

### 2. Search Memories

Search for specific memories with advanced filtering:

```
User: "What do you know about my pets?"
Claude uses search_memories with query="pets"
→ Returns relevant memories
```

**Optional Parameters:**
- `user_id`: Filter memories by user
- `agent_id`: Filter memories by agent
- `categories`: Filter by memory tags (e.g., ["preference", "identity"])

### 3. Get Relevant Memories

Retrieves contextually relevant memories above relevance threshold:

```
User: "Tell me about my hobbies"
Claude uses get_relevant_memories
→ Returns hobby-related memories above relevance threshold
```

**Optional Parameters:**
- `user_id`: Filter memories by user

### 4. Delete Memory

Remove a specific memory by ID:

```
User: "Delete that memory about my cat"
Claude uses delete_memory with memory_id
→ Removes the memory
```

### 5. Batch Delete Memories

Delete multiple memories at once:

```
Claude uses batch_delete_memories with memory_ids=["mem_123", "mem_456"]
→ Deletes both memories efficiently
```

### 6. Get Stats

View memory system statistics:

```
Claude uses get_stats
→ Returns: Total memories, embedding dimension, max capacity
```

## Configuration Options

All settings are configured via environment variables in Claude Desktop config:

### Embedding Providers

**Local (sentence-transformers) - Recommended**
```json
"EMBEDDING_PROVIDER": "local",
"EMBEDDING_MODEL": "all-MiniLM-L6-v2"
```

**API (OpenAI-compatible)**
```json
"EMBEDDING_PROVIDER": "api",
"EMBEDDING_API_URL": "https://api.openai.com/v1/embeddings",
"EMBEDDING_API_KEY": "your-key",
"EMBEDDING_MODEL": "text-embedding-3-small"
```

### Memory Settings

```json
"MAX_MEMORIES": "200",           // Max memories before pruning
"DEDUP_THRESHOLD": "0.95",       // Similarity threshold for duplicates (0-1)
"MIN_CONFIDENCE": "0.5",         // Min confidence to store memory (0-1)
"RELEVANCE_THRESHOLD": "0.6"     // Min relevance to return memory (0-1)
```

## Troubleshooting

### MCP Server Not Appearing in Claude

1. Check Claude Desktop logs:
   - Windows: `%APPDATA%\Claude\logs\mcp.log`

2. Verify Python path is correct:
   ```powershell
   C:\Users\YourUsername\mcp-smartmemory\venv\Scripts\python.exe --version
   ```

3. Test the server manually:
   ```powershell
   cd C:\Users\YourUsername\mcp-smartmemory
   .\venv\Scripts\activate
   python server.py
   ```
   
   It should start without errors. Press Ctrl+C to stop.

### Memory Extraction Not Working

- Check API keys are correct
- Verify Pinecone index exists and is accessible
- Check logs in Claude Desktop

### Embedding Errors

- If using local embeddings, ensure the model downloaded correctly
- Try deleting and recreating the virtual environment
- Check available disk space

## Architecture

```
Claude Desktop
      │
      ├─── MCP Protocol (stdio)
      │
SmartMemory Server (Python)
      │
      ├─── Memory Engine
      ├─── LLM Client (OpenAI API)
      ├─── Embedding Provider (Local/API)
      │
Pinecone Vector Database
```

## Files

- `server.py` - MCP server entry point
- `memory_engine.py` - Core memory extraction/storage logic
- `embeddings.py` - Embedding generation (local/API)
- `llm_client.py` - LLM API client
- `config.py` - Configuration management
- `requirements.txt` - Python dependencies

## Logs

Server logs are visible in Claude Desktop's MCP log file:

```
%APPDATA%\Claude\logs\mcp.log
```

Look for lines containing "smartmemory" to see what the server is doing.

## License

MIT
