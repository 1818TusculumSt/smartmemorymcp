# SmartMemory MCP Server

**Transform Claude Desktop into a personalized AI that remembers you across all conversations.**

An intelligent memory system for Claude Desktop that automatically extracts, stores, and recalls user-specific facts, preferences, and context using semantic search powered by Pinecone vector database and LLM analysis.

## What This Does

SmartMemory solves Claude's amnesia problem. Instead of starting from scratch every conversation, Claude now:
- **Automatically remembers** what you tell it about yourself
- **Proactively recalls** relevant context when you ask questions
- **Updates memories** as your preferences and information change
- **Searches semantically** to find exactly what it needs to know

## Key Features

### Memory Management (v2.0)
- **Automatic Extraction** - LLM analyzes conversations to identify and store user facts, preferences, goals, and context
- **Smart Categorization** - Automatically classifies memories into 10 categories (achievement, frustration, idea, fact, event, conversation, relationship, technical, personal, misc)
- **Importance Scoring** - Assigns 1-10 importance scores to prioritize critical information
- **Sentiment Analysis** - Tracks emotional tone (positive, negative, neutral, mixed) of memories
- **Proactive Recall** - Automatically retrieves relevant memories at conversation start without being asked
- **Smart Updates** - Detects and updates outdated memories instead of rejecting them as duplicates
- **Hybrid Search** - Combines semantic similarity (embeddings) with keyword matching for 3x better precision
- **Memory Consolidation** - Merges fragmented memories into coherent summaries
- **Temporal Awareness** - Tracks creation, update times, and optional event dates
- **Quality Curation** - Pin important memories, archive outdated ones

### Organization & Filtering
- **Tag-Based Organization** - Tag memories with custom labels for easy categorization
- **Search by Tag** - Find all memories with a specific tag
- **Time-Based Retrieval** - Get memories from specific date ranges
- **Importance Filtering** - Focus on high-priority information
- **Archive System** - Soft-delete outdated memories (reversible)
- **Pinned Memories** - Mark critical information to always include in context

### Infrastructure
- **MCP Resources** - Exposes recent memories and stats that Claude can see automatically
- **Semantic Search** - Vector-based similarity search using normalized embeddings
- **Intelligent Deduplication** - Prevents exact duplicates using 0.95 similarity threshold
- **Flexible Embedding Providers** - Switch between Pinecone Inference, local sentence-transformers, or API endpoints
- **Auto-Pruning** - Maintains memory limit by removing oldest entries when capacity reached
- **Multi-User Support** - Track memories per user with user_id filtering
- **Session Tracking** - Organize memories by conversation using agent_id and run_id
- **Batch Operations** - Delete multiple memories efficiently
- **Rich Statistics** - Category breakdown, sentiment analysis, tag distribution, importance metrics

### Data Portability & Privacy
- **Your Data, Your Control** - All memories stored in your own Pinecone account, not ours
- **Export Anytime** - Memories stored as simple JSON metadata in vector database
- **Provider Flexibility** - Currently supports Pinecone; local vector databases on the roadmap
- **No Vendor Lock-In** - Switch between embedding providers without losing memories
- **Local Embedding Option** - Run embeddings completely offline with sentence-transformers

## Attribution

The memory extraction and deduplication system in this project is derived from [gramanoid'''s Adaptive Memory filter for Open WebUI](https://github.com/gramanoid/owui-adaptive-memory). The original filter provided the foundation for LLM-based memory extraction, embedding similarity comparison, and semantic deduplication. This implementation adapts those concepts into a standalone MCP server with additional features like proactive recall, hybrid search, memory consolidation, and temporal awareness.

## Available MCP Tools

SmartMemory provides 15 MCP tools that Claude can use automatically:

### Core Memory Operations
| Tool | Description | When Claude Uses It |
|------|-------------|-------------------|
| `auto_recall_memories` | Retrieve memories relevant to conversation | 🔴 Automatically at start of every response |
| `extract_memories` | Extract and store facts from conversation | 🔴 Automatically when user shares information |
| `search_memories` | Search memories by semantic similarity | When user asks "What did I say about..." |
| `get_relevant_memories` | Get contextual memories for current topic | Alternative to auto_recall |

### Time-Based Retrieval (v2.0)
| Tool | Description | Use Case |
|------|-------------|----------|
| `get_recent_memories` | Get memories sorted by time (newest first) | "Show me recent memories", "What happened this week?" |

### Tag Management (v2.0)
| Tool | Description | Use Case |
|------|-------------|----------|
| `get_all_tags` | List all tags with usage counts | "What tags do I have?", "How are my memories organized?" |
| `add_memory_tags` | Add or replace tags on a memory | "Tag that memory as homelab", "Categorize that as technical" |
| `search_by_tag` | Find memories with a specific tag | "Show me all homelab memories", "My technical notes" |

### Quality Curation (v2.0)
| Tool | Description | Use Case |
|------|-------------|----------|
| `set_memory_importance` | Set importance score 1-10 | "Mark that as important", "This is critical" |
| `pin_memory` | Pin memory to always include | "Always remember this", "Never forget this" |
| `archive_memory` | Archive memory (soft delete) | "Forget that for now", "Archive old project notes" |

### Bulk Operations
| Tool | Description | Use Case |
|------|-------------|----------|
| `delete_memory` | Delete a specific memory | "Delete that memory", "Forget that" |
| `batch_delete_memories` | Delete multiple memories | "Clear all memories about X" |
| `consolidate_memories` | Merge similar memories | "Organize my memories", "Clean up duplicates" |

### System Information
| Tool | Description | Use Case |
|------|-------------|----------|
| `get_stats` | Get memory statistics (v2.0 enhanced) | "How many memories?", "Show memory stats" |

### What's New in v2.0

**Enhanced Memory Metadata:**
- **Categories**: achievement, frustration, idea, fact, event, conversation, relationship, technical, personal, misc
- **Importance**: 1-10 scoring (auto-assigned during extraction)
- **Sentiment**: positive, negative, neutral, mixed
- **Tags**: Array-based tagging for better organization
- **Flags**: pinned (always include), archived (hide from searches)
- **Timestamps**: created_at, updated_at, optional event_date

**Example Memory Structure:**
```json
{
  "content": "User completed BADBUNNY PSU migration after 3 hours",
  "tags": ["homelab", "hardware", "BADBUNNY", "victory"],
  "category": "achievement",
  "importance": 8,
  "sentiment": "positive",
  "confidence": 0.95,
  "word_count": 7,
  "pinned": false,
  "archived": false
}
```

**Enhanced Statistics:**
```
📊 Overview: Total, active, archived, pinned counts
📁 Categories: Breakdown by category
😊 Sentiment: Distribution of positive/negative/neutral
🏷️  Top Tags: Most-used tags
⭐ Importance: Distribution by score
```

## Prerequisites

- **Python 3.11+**
- **Pinecone account and API key** ([sign up free](https://www.pinecone.io/)) - *Your own account, your own data*
- **OpenAI API key** or compatible LLM API (OpenRouter, local models, etc.)
- **Claude Desktop** for Windows
## Installation

### 1. Clone or Download

```bash
git clone https://github.com/1818TusculumSt/smartmemorymcp.git
cd smartmemorymcp
```

Or download and extract to: `C:\Users\YourUsername\smartmemorymcp`

### 2. Install Python Dependencies

Open PowerShell or Command Prompt:

```powershell
cd C:\Users\YourUsername\smartmemorymcp
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies installed:**
- `mcp>=1.0.0` - Model Context Protocol server
- `pinecone>=5.0.0` - Vector database client
- `sentence-transformers>=2.3.1` - Local embedding models
- `numpy>=1.26.0` - Array operations
- `aiohttp>=3.9.0` - Async HTTP client
- `pydantic>=2.5.0` - Settings validation
- `python-dotenv>=1.0.0` - Environment variable loading

### 3. Configure Claude Desktop

Edit your Claude Desktop config file at `%APPDATA%\Claude\claude_desktop_config.json`

Add the SmartMemory server to the `mcpServers` section:

```json
{
  "mcpServers": {
    "smartmemory": {
      "command": "C:\Users\YourUsername\smartmemorymcp\venv\Scripts\python.exe",
      "args": ["C:\Users\YourUsername\smartmemorymcp\server.py"],
      "env": {
        "PINECONE_API_KEY": "your-pinecone-api-key",
        "PINECONE_ENVIRONMENT": "us-east-1-aws",
        "PINECONE_INDEX_NAME": "smartmemory",
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

**Important Notes:**
- Use **double backslashes** (`\`) in Windows paths
- Replace `YourUsername` with your actual Windows username
- Get your Pinecone API key from [console.pinecone.io](https://console.pinecone.io/)
- Get your OpenAI API key from [platform.openai.com](https://platform.openai.com/)

### 4. Restart Claude Desktop

1. Completely quit Claude Desktop (check system tray)
2. Restart Claude Desktop
3. Wait ~10 seconds for the MCP server to initialize

## Configuration Options

### Required Settings

| Variable | Description | Example |
|----------|-------------|---------|
| `PINECONE_API_KEY` | Your Pinecone API key | `pcsk_123abc...` |
| `PINECONE_ENVIRONMENT` | Pinecone cloud region | `us-east-1-aws` |
| `PINECONE_INDEX_NAME` | Name for your memory index | `smartmemory` |
| `LLM_API_URL` | OpenAI-compatible API endpoint | `https://api.openai.com/v1/chat/completions` |
| `LLM_API_KEY` | Your LLM provider API key | `sk-...` |
| `LLM_MODEL` | Model for memory extraction | `gpt-4o-mini` |

### Embedding Provider Options

**Option 1: Pinecone Inference (Recommended)**
```json
"EMBEDDING_PROVIDER": "pinecone",
"EMBEDDING_MODEL": "llama-text-embed-v2"
```
- Models: `llama-text-embed-v2` (384-dim), `multilingual-e5-large` (1024-dim)
- Pros: Fast, integrated, low cost
- Cons: Requires API key

**Option 2: Local (sentence-transformers)**
```json
"EMBEDDING_PROVIDER": "local",
"EMBEDDING_MODEL": "all-MiniLM-L6-v2"
```
- Models: `all-MiniLM-L6-v2` (384-dim), `all-mpnet-base-v2` (768-dim)
- Pros: Free, privacy-friendly, fully offline
- Cons: Slower first run, uses more RAM

**Option 3: API (OpenAI-compatible)**
```json
"EMBEDDING_PROVIDER": "api",
"EMBEDDING_API_URL": "https://api.openai.com/v1/embeddings",
"EMBEDDING_API_KEY": "your-key",
"EMBEDDING_MODEL": "text-embedding-3-small"
```
- Models: OpenAI, Voyage AI, Cohere, etc.
- Pros: High quality embeddings
- Cons: Additional API cost

### Memory Tuning Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `MAX_MEMORIES` | 200 | 1-10000 | Max memories before auto-pruning |
| `DEDUP_THRESHOLD` | 0.95 | 0.0-1.0 | Similarity threshold for duplicates |
| `MIN_CONFIDENCE` | 0.5 | 0.0-1.0 | Minimum confidence to store memory |
| `RELEVANCE_THRESHOLD` | 0.6 | 0.0-1.0 | Minimum relevance to return |

**How thresholds work:**
- Similarity ≥ 0.95 → Exact duplicate, skip
- Similarity 0.85-0.94 → Update existing memory
- Similarity < 0.85 → Store as new memory

See [ARCHITECTURE.md](ARCHITECTURE.md) for full technical documentation.

## Support

- **Issues:** [GitHub Issues](https://github.com/1818TusculumSt/smartmemorymcp/issues)
- **Logs:** `%APPDATA%\Claude\logs\mcp.log` for debugging
- **Documentation:** [ARCHITECTURE.md](ARCHITECTURE.md) for technical details

## Contributing

Contributions welcome! We especially need help with:

**🎯 Top Priority: Fully Local Operation**
- Support for local vector databases (ChromaDB, Qdrant, Weaviate)
- Local LLM integration (Ollama, llama.cpp, vLLM)
- Offline-first architecture with no cloud dependencies
- Memory export/import for portability

**How to contribute:**

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

---

**SmartMemory transforms Claude from a forgetful chatbot into a personalized AI assistant that truly knows you.**
