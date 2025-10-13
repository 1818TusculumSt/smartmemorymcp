<img width="1024" height="576" alt="Adobe Express - file" src="https://github.com/user-attachments/assets/0d591de8-f907-4313-8233-13c89b59a2f3" />

# SmartMemory MCP Server

Auto-extracting memory system for Claude Desktop with semantic search powered by Pinecone and LLM analysis.

## Features

### Core Capabilities
- **Automatic Memory Extraction**: Uses LLM to identify user-specific facts, preferences, and context
- **Proactive Memory Recall**: Automatically injects relevant memories as context (NEW!)
- **Smart Memory Updates**: Detects and updates outdated memories instead of rejecting them (NEW!)
- **Hybrid Search**: Combines semantic similarity with keyword matching for better results (NEW!)
- **Memory Consolidation**: Merges fragmented memories into coherent summaries (NEW!)
- **Temporal Awareness**: Tracks when memories were created with date context (NEW!)

### Infrastructure
- **MCP Resources**: Exposes recent memories and stats that Claude can see automatically (NEW!)
- **Semantic Search**: Vector-based similarity search using embeddings
- **Intelligent Deduplication**: Prevents duplicate memories using embedding similarity
- **Configurable Providers**: Switch between Pinecone, local, and API-based embeddings
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
        "EMBEDDING_PROVIDER": "pinecone",
        "EMBEDDING_MODEL": "llama-text-embed-v2",
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
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=llama-text-embed-v2
MAX_MEMORIES=200
DEDUP_THRESHOLD=0.95
MIN_CONFIDENCE=0.5
RELEVANCE_THRESHOLD=0.6
```

**Note:** This is only needed for `python server.py` testing. Claude Desktop passes these values from the JSON config.

### 5. Restart Claude Desktop

Completely quit and restart Claude Desktop for the MCP server to load.

## What's New in v2.0

This version transforms SmartMemory from a **reactive** to a **proactive** memory system:

**Before**: Claude only accessed memories when explicitly asked
**After**: Claude automatically recalls relevant memories and sees recent context

### Key Improvements:
1. **MCP Resources** - Recent memories visible to Claude automatically
2. **Auto-Recall Tool** - Proactive memory injection at conversation start
3. **Smart Updates** - Memories evolve instead of being rejected as duplicates
4. **Hybrid Search** - Better results by combining semantic + keyword matching
5. **Consolidation** - Merge fragmented memories into coherent summaries
6. **Temporal Context** - Track when facts were learned

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed technical documentation.

## Usage

The server provides **8 tools** and **3 resources** that Claude can use:

### MCP Resources (Auto-Visible)

Claude sees these automatically without calling tools:

- **`memory://recent`** - Last 10 stored memories with tags
- **`memory://stats`** - System statistics and capacity
- **`memory://system-prompt`** - Usage instructions for Claude

### Tools

### 1. Auto-Recall Memories (NEW!)

Automatically retrieves relevant memories for current conversation:

```
Claude internally calls this at response start:
auto_recall_memories(conversation_context="user asking about pets")
→ Returns relevant memories about user's pets
```

**Parameters:**
- `conversation_context`: Brief topic summary
- `limit`: Max memories to return (default: 5)
- `user_id`: User filter (optional)

### 2. Extract Memories

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

### 3. Search Memories

Search for specific memories with hybrid semantic + keyword matching:

```
User: "What do you know about my pets?"
Claude uses search_memories with query="pets"
→ Returns relevant memories (now with keyword boost!)
```

**Parameters:**
- `query`: Search query
- `limit`: Max results (default: 5)
- `user_id`: Filter by user (optional)
- `agent_id`: Filter by agent (optional)
- `categories`: Filter by tags (optional)

**Improvement**: Now uses hybrid search for better precision!

### 4. Get Relevant Memories

Retrieves contextually relevant memories above relevance threshold:

```
User: "Tell me about my hobbies"
Claude uses get_relevant_memories
→ Returns hobby-related memories above threshold
```

**Parameters:**
- `current_message`: Message to find context for
- `limit`: Max results (default: 5)
- `user_id`: Filter by user (optional)

### 5. Consolidate Memories (NEW!)

Merge fragmented related memories into coherent summaries:

```
Before: ["User has a cat", "Cat is named Whiskers", "Cat is orange"]
After: ["User has an orange cat named Whiskers"]
```

**Parameters:**
- `user_id`: User to consolidate for (optional)
- `tag`: Specific category to consolidate (optional)

Use this periodically to improve memory quality and reduce redundancy.

### 6. Delete Memory

Remove a specific memory by ID:

```
User: "Delete that memory about my cat"
Claude uses delete_memory with memory_id
→ Removes the memory
```

### 7. Batch Delete Memories

Delete multiple memories at once:

```
Claude uses batch_delete_memories with memory_ids=["mem_123", "mem_456"]
→ Deletes both memories efficiently
```

### 8. Get Stats

View memory system statistics:

```
Claude uses get_stats
→ Returns: Total memories, embedding dimension, max capacity
```

## Configuration Options

All settings are configured via environment variables in Claude Desktop config:

### Embedding Providers

**Pinecone Inference (Recommended)**
```json
"EMBEDDING_PROVIDER": "pinecone",
"EMBEDDING_MODEL": "llama-text-embed-v2"
```
Supported models: `llama-text-embed-v2` (1024-dim), `multilingual-e5-large` (1024-dim)

**Local (sentence-transformers)**
```json
"EMBEDDING_PROVIDER": "local",
"EMBEDDING_MODEL": "all-MiniLM-L6-v2"
```
Supported models: `all-MiniLM-L6-v2` (384-dim), `all-mpnet-base-v2` (768-dim)

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
