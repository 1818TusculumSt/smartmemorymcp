# SmartMemory MCP Server - Enhancement Summary

## Overview
This document outlines all the improvements made to transform SmartMemory from a reactive to a proactive memory system that automatically recalls and injects relevant context.

## Key Problem Solved
**Original Issue**: Claude only accessed memories when explicitly asked via tool calls. Memories existed but weren't automatically surfaced as context.

**Solution**: Multiple complementary improvements to make memories visible and automatically recalled.

---

## v2.0 Phase 1: Enhanced Metadata & Organization 🎯 NEW (2025)

### Overview
Phase 1 of v2.0 enhances the memory system with rich metadata, intelligent categorization, and powerful organization tools—all while maintaining full backward compatibility with existing Pinecone data.

### Enhanced Metadata Schema

**Previous (v1.x):**
```python
{
    "content": str,
    "tags": "comma,separated,string",  # String format
    "confidence": float,
    "timestamp": str,
    "user_id": str
}
```

**New (v2.0):**
```python
{
    # Core fields (existing)
    "content": str,
    "confidence": float,
    "timestamp": str,
    "created_at": str,
    "user_id": str,
    "agent_id": str,
    "run_id": str,

    # V2.0 additions - Categorization
    "tags": ["tag1", "tag2"],           # Now an array
    "category": str,                     # achievement, frustration, idea, fact, etc.

    # V2.0 additions - Quality/Importance
    "importance": int,                   # 1-10 scale
    "pinned": bool,                      # Always include in context
    "archived": bool,                    # Soft delete, hide from searches

    # V2.0 additions - Content analysis
    "word_count": int,
    "sentiment": str,                    # positive, negative, neutral, mixed

    # V2.0 additions - Temporal
    "updated_at": str,
    "event_date": str                    # Optional: when event occurred
}
```

### New Categories (Auto-Assigned)
LLM automatically classifies memories into one of 10 categories:

| Category | Description | Examples |
|----------|-------------|----------|
| `achievement` | Completed tasks, successes, victories | "Fixed BADBUNNY PSU issue", "Deployed new feature" |
| `frustration` | Problems, failures, roadblocks | "Network connectivity issues", "Failed migration" |
| `idea` | Thoughts, plans, future considerations | "Should implement caching", "Consider using Redis" |
| `fact` | Factual information about user | "Lives in Seattle", "Uses Python", "Has 4 kids" |
| `event` | Things that happened | "Team meeting today", "Conference next week" |
| `conversation` | Discussion topics | "Talked about AI ethics", "Discussed homelab setup" |
| `relationship` | Information about people | "Works with Jane", "Friend named Mike" |
| `technical` | Code, systems, infrastructure | "Using Docker", "Postgres on port 5432" |
| `personal` | Family, hobbies, interests | "Enjoys hiking", "Plays guitar" |
| `misc` | Everything else | - |

### Importance Scoring (1-10)
LLM assigns importance during extraction:

- **1-3 (Low)**: Minor details, trivial facts, passing mentions
- **4-6 (Medium)**: Useful information, regular preferences, typical activities
- **7-8 (High)**: Important facts, strong preferences, key relationships, significant events
- **9-10 (Critical)**: Core identity, mission-critical information, deeply held values

### Sentiment Analysis
Tracks emotional tone:
- **positive**: Achievements, good news, preferences, successes
- **negative**: Frustrations, problems, dislikes, failures
- **neutral**: Facts, observations, routine information
- **mixed**: Complex feelings, both good and bad aspects

### New MCP Tools (7 Added)

#### Time-Based Retrieval
**`get_recent_memories`** - Chronological view of memories
```python
get_recent_memories(
    user_id: Optional[str],
    limit: int = 20,
    since: Optional[str],        # ISO timestamp
    before: Optional[str],       # ISO timestamp
    include_archived: bool = False
) -> List[Memory]
```

#### Tag Management
**`get_all_tags`** - List all tags with usage counts
```python
get_all_tags(
    user_id: Optional[str],
    min_count: int = 1,
    sort_by: str = "count"  # "count" or "name"
) -> List[Dict[str, Any]]
```

**`add_memory_tags`** - Add/replace tags on memory
```python
add_memory_tags(
    memory_id: str,
    tags: List[str],
    replace: bool = False
) -> Dict[str, Any]
```

**`search_by_tag`** - Find memories with specific tag
```python
search_by_tag(
    tag: str,
    user_id: Optional[str],
    limit: int = 20,
    sort_by: str = "created_at"  # or "importance"
) -> List[Memory]
```

#### Quality Curation
**`set_memory_importance`** - Set 1-10 importance score
```python
set_memory_importance(
    memory_id: str,
    importance: int  # 1-10
) -> Dict[str, Any]
```

**`pin_memory`** - Pin memory for always-include
```python
pin_memory(
    memory_id: str
) -> Dict[str, Any]
```

**`archive_memory`** - Soft delete (reversible)
```python
archive_memory(
    memory_id: str
) -> Dict[str, Any]
```

### Enhanced Extraction Prompt

The LLM extraction prompt now includes:

1. **Category classification** with descriptions and examples
2. **Importance scoring** with clear 1-10 guidelines
3. **Sentiment analysis** instructions
4. **Word count** automatic calculation
5. **Comprehensive vs atomic** extraction guidance

Example output:
```json
{
  "content": "User completed BADBUNNY PSU migration after 3 hours",
  "tags": ["homelab", "hardware", "BADBUNNY", "victory"],
  "category": "achievement",
  "importance": 8,
  "sentiment": "positive",
  "confidence": 0.95,
  "word_count": 7
}
```

### Enhanced Statistics (`get_stats`)

Now returns rich metadata breakdown:

```python
{
    "count": 150,
    "active_count": 140,
    "archived_count": 10,
    "pinned_count": 5,
    "utilization_pct": 75.0,

    "categories": {
        "technical": 45,
        "achievement": 32,
        "fact": 28,
        ...
    },

    "sentiment_distribution": {
        "positive": 85,
        "neutral": 50,
        "negative": 15
    },

    "importance_distribution": {
        10: 3,
        9: 8,
        8: 15,
        ...
    },

    "top_tags": [
        {"name": "homelab", "count": 42},
        {"name": "technical", "count": 38},
        ...
    ],

    "unique_tag_count": 67
}
```

### Implementation Details

**Files Modified:**
- `memory_engine.py` (~500 lines added)
  - Enhanced metadata schema in `_store_memory()`
  - Updated extraction prompt in `_extract_memories()`
  - Added 6 new methods for v2.0 tools
  - Enhanced `get_stats()` with metadata aggregation
  - Updated `get_recent()` with date filtering and archived flag

- `server.py` (~480 lines added)
  - Added 7 new MCP tool definitions
  - Added 7 new tool handlers
  - Enhanced `get_stats` response formatting

**Backward Compatibility:**
- All new metadata fields are optional
- Tags support both string (old) and array (new) formats
- Existing memories work without modification
- No breaking changes to existing tools
- Pinecone metadata is flexible (supports nested JSON)

### Why It Matters

1. **Better Organization**: Tag-based filtering and categorization
2. **Quality Curation**: Pin important, archive outdated
3. **Temporal Queries**: Find memories by date ranges
4. **Rich Analytics**: Understand memory patterns and distribution
5. **Importance Prioritization**: Focus on what matters most
6. **Sentiment Tracking**: Understand emotional context
7. **No Migration Required**: Works with existing Pinecone data

### What's Next (Phase 2+)

Future phases will add:
- Advanced multi-filter search (category + importance + tags + dates)
- Pinned memory priority in auto_recall
- Archived memory exclusion from searches
- Migration script for backfilling v2.0 metadata on old memories
- Relationship detection and entity recognition
- Duplicate detection improvements

---

## 1. MCP Resources for Proactive Visibility ✨ MAJOR

### What Changed
Added three MCP Resources that Claude can see automatically:

1. **`memory://recent`** - Last 10 memories with tags and confidence scores
2. **`memory://stats`** - System statistics (count, capacity, utilization)
3. **`memory://system-prompt`** - Instructions for using the memory system effectively

### Why It Matters
- Claude now **sees** recent memories without asking
- Resources appear in Claude's context automatically
- Provides guidance on when and how to use memory tools

### Implementation
```python
@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available memory resources that Claude can see"""

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Provide memory content as resources"""
```

---

## 2. Auto-Recall Memories Tool ✨ MAJOR

### What Changed
New tool: `auto_recall_memories` that Claude is instructed to call at the START of each response.

### Tool Signature
```python
auto_recall_memories(
    conversation_context: str,  # Brief summary of current topic
    limit: int = 5,             # Max memories to recall
    user_id: Optional[str]      # User filter
)
```

### Why It Matters
- Enables **proactive** context injection
- Claude gets relevant memories BEFORE responding
- Works better than passive resources for dynamic context

### Usage Pattern
```
User: "What should I work on today?"
Claude internally: auto_recall_memories("user asking about work tasks")
Returns: Relevant memories about projects, deadlines, preferences
Claude: "Based on your current projects and preference for morning coding..."
```

---

## 3. Enhanced Tool Descriptions 🎯

### What Changed
Updated all tool descriptions to guide Claude's behavior:

**Before:**
```python
"Extract and store memories from user message"
```

**After:**
```python
"[CALL AUTOMATICALLY] Extract and store user-specific facts, preferences,
and context from the conversation. Call this AUTOMATICALLY when user shares
personal information - don't ask permission first."
```

### Why It Matters
- Explicit instructions = better Claude behavior
- Tags like `[CALL FIRST]` and `[CALL AUTOMATICALLY]` create execution order
- Reduces unnecessary permission-seeking

---

## 4. Smart Memory Updates (Not Just Deduplication) 🔄

### What Changed
Replaced simple "duplicate or unique" logic with intelligent update detection:

**Before:**
- Similarity ≥ 0.95 = duplicate, reject
- Similarity < 0.95 = unique, store

**After:**
- Similarity ≥ 0.95 = exact duplicate, skip
- Similarity 0.85-0.94 = **update old memory**
- Similarity < 0.85 = new memory, store

### Implementation
```python
update_threshold = 0.85
if similarity >= settings.DEDUP_THRESHOLD:
    is_duplicate = True  # Skip
elif similarity >= update_threshold:
    await self.delete(similar_memory['id'])  # Update
    store_new_version()
```

### Why It Matters
- Memories evolve over time (preferences change, new information)
- No longer stuck with outdated memories
- Reduces "I already told you that" frustration

### Example
```
Old: "User prefers Python"
New: "User now prefers Rust after learning it"
Result: Old memory deleted, new one stored
```

---

## 5. Hybrid Search (Semantic + Keyword) 🔍

### What Changed
Enhanced search to combine:
1. **Semantic similarity** (via embeddings)
2. **Keyword matching** (exact term matches)

### Algorithm
```python
semantic_score = cosine_similarity(query_emb, memory_emb)
keyword_matches = count_matching_terms(query, memory_content)
keyword_boost = 1.0 + (0.15 * keyword_matches)
final_score = semantic_score * keyword_boost
```

### Why It Matters
- Better handles specific queries ("my cat named Whiskers")
- Semantic search alone can miss exact name matches
- 15% boost per keyword match improves precision

### Results
- Retrieves 3x more candidates
- Re-ranks by hybrid score
- Returns top N results

---

## 6. Temporal Context in Extraction 📅

### What Changed
Memory extraction now includes current date and temporal awareness:

```python
current_date = datetime.now().strftime("%Y-%m-%d")
system_prompt = f"""
CURRENT DATE: {current_date}
Include temporal context when relevant (e.g., "As of {current_date}, user prefers X").
"""
```

### Why It Matters
- Tracks when preferences/facts were stated
- Helps with memory aging and updates
- Provides context for time-sensitive information

---

## 7. Memory Consolidation 🎯

### What Changed
New capability to merge related fragmented memories into coherent summaries.

### How It Works
1. **Group** memories by semantic similarity (≥0.75)
2. **LLM consolidates** each group into single memory
3. **Delete** old fragments
4. **Store** consolidated version

### Example
**Before:**
```
- User has a cat
- User's cat is named Whiskers
- User's cat is orange
```

**After:**
```
- User has an orange cat named Whiskers
```

### Tool
```python
consolidate_memories(
    user_id: Optional[str],
    tag: Optional[str]  # Consolidate specific category
)
```

### Why It Matters
- Reduces redundancy
- Creates more informative memories
- Better utilizes limited memory capacity

---

## 8. Helper Methods Added

### `get_recent(limit=10)`
- Retrieves most recent memories by timestamp
- Used by `memory://recent` resource
- Sorted descending (newest first)

### Memory Engine Enhancements
- All methods now support `user_id` filtering
- Better error handling and logging
- Async throughout for performance

---

## Comparison: Before vs After

### Before (Reactive)
```
User: "I love Python"
Claude: "That's great!"
[Memory stored but not used]

User: "What languages do I like?"
Claude: "I don't have that information" ❌
```

### After (Proactive)
```
User: "I love Python"
Claude: [auto-stores memory]
"Noted! I'll remember your preference for Python"

User: "What languages do I like?"
Claude: [auto_recall_memories("user asking about language preferences")]
"Based on our previous conversations, you love Python!" ✅
```

---

## Configuration Updates

No configuration changes needed! All improvements work with existing settings.

Optional: Adjust these thresholds in `.env` or Claude config:
```env
DEDUP_THRESHOLD=0.95      # Exact duplicate threshold
MIN_CONFIDENCE=0.5         # Minimum confidence to store
RELEVANCE_THRESHOLD=0.6    # Minimum relevance to return
```

---

## Usage Guide for Claude

The system now includes built-in instructions via `memory://system-prompt`:

### For Each Response:
1. **Call `auto_recall_memories`** with conversation context
2. Use returned memories to inform response
3. **Call `extract_memories`** if user shared personal info

### Search Patterns:
- **Specific queries**: Use `search_memories`
- **General context**: Use `get_relevant_memories`
- **Periodic maintenance**: Call `consolidate_memories`

### Best Practices:
- Extract memories automatically (don't ask permission)
- Reference specific memories when relevant
- Update outdated memories by storing new versions
- Consolidate memories periodically (every 50+ memories)

---

## Technical Architecture

```
┌─────────────────────────────────────────┐
│         Claude Desktop                   │
│  ┌─────────────────────────────────┐   │
│  │   MCP Resources (Auto-Visible)   │   │
│  │   - memory://recent              │   │
│  │   - memory://stats               │   │
│  │   - memory://system-prompt       │   │
│  └─────────────────────────────────┘   │
│                   ↓                      │
│  ┌─────────────────────────────────┐   │
│  │   MCP Tools (Call Explicitly)    │   │
│  │   - auto_recall_memories        │   │
│  │   - extract_memories            │   │
│  │   - search_memories             │   │
│  │   - consolidate_memories        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│      SmartMemory MCP Server (Python)    │
│  ┌─────────────────────────────────┐   │
│  │   Memory Engine                  │   │
│  │   - Smart deduplication/updates │   │
│  │   - Hybrid search               │   │
│  │   - Memory consolidation        │   │
│  │   - Temporal awareness          │   │
│  └─────────────────────────────────┘   │
│                   ↓                      │
│  ┌─────────────────────────────────┐   │
│  │   Embedding Provider             │   │
│  │   - Pinecone Inference          │   │
│  │   - Local (sentence-transformers)│   │
│  │   - API (OpenAI-compatible)     │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│     Pinecone Vector Database            │
│     - Semantic search                    │
│     - Metadata filtering                 │
│     - Cosine similarity                  │
└─────────────────────────────────────────┘
```

---

## Next Steps

1. **Restart Claude Desktop** to load the updated server
2. **Test the improvements**:
   - Share personal information → Verify auto-extraction
   - Ask questions → Verify auto-recall
   - Check `memory://recent` resource visibility
3. **Monitor logs** at `%APPDATA%\Claude\logs\mcp.log`
4. **Consolidate memories** periodically for best performance

---

## File Changes Summary

### Modified Files:
1. **server.py**
   - Added MCP Resources (3 endpoints)
   - Added `auto_recall_memories` tool
   - Added `consolidate_memories` tool
   - Enhanced tool descriptions
   - Added resource read handlers

2. **memory_engine.py**
   - Added `get_recent()` method
   - Enhanced `_deduplicate()` with update logic
   - Enhanced `search()` with hybrid scoring
   - Added `consolidate_memories()` method
   - Added `_group_similar_memories()` helper
   - Added `_consolidate_group()` helper
   - Added temporal context to extraction prompts

3. **No changes needed to:**
   - config.py
   - embeddings.py
   - llm_client.py
   - requirements.txt

---

## Performance Considerations

### Memory Usage
- Hybrid search retrieves 3x candidates (3x `limit`)
- Consolidation groups all memories temporarily
- Impact: Minimal (<100MB extra for 200 memories)

### API Calls
- Auto-recall adds 1 embedding generation per response
- Consolidation adds 1 LLM call per group
- Impact: ~$0.0001 per auto-recall with Pinecone

### Latency
- Auto-recall: ~200-500ms (parallel with Claude thinking)
- Consolidation: ~2-5s per group
- Impact: Negligible for user experience

---

## Troubleshooting

### Claude not calling auto_recall_memories
- Check `memory://system-prompt` is visible
- Check tool description has `[CALL FIRST]` tag
- Try explicit prompt: "Check my memories first"

### Memories not updating
- Verify similarity between old/new is 0.85-0.94
- Check logs for "Similar memory found" messages
- Adjust `DEDUP_THRESHOLD` if needed

### Poor search results
- Try `use_hybrid=True` (default)
- Increase result limit
- Check embedding quality (test with known queries)

### Consolidation not working
- Need ≥3 memories to consolidate
- Requires ≥2 similar memories (≥0.75 similarity)
- Check LLM is responding (see logs)

---

## Credits

Based on concepts from:
- [gramanoid's Adaptive Memory filter](https://github.com/gramanoid/owui-adaptive-memory) - Original LLM-based extraction and deduplication
- MCP Protocol - Resource-based context injection
- Hybrid search techniques from information retrieval research

## License

MIT License (same as original project)
