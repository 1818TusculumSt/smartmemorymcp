# SmartMemory MCP Server - Enhancement Summary

## Overview
This document outlines all the improvements made to transform SmartMemory from a reactive to a proactive memory system that automatically recalls and injects relevant context.

## Key Problem Solved
**Original Issue**: Claude only accessed memories when explicitly asked via tool calls. Memories existed but weren't automatically surfaced as context.

**Solution**: Multiple complementary improvements to make memories visible and automatically recalled.

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
