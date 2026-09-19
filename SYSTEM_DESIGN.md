# LearnForge AI Support - System Design

## Executive Summary

LearnForge AI Support is a RAG-based chatbot that reduces hallucination by grounding responses in a knowledge base, maintains conversation context, and escalates uncertain answers to humans.

**Key Design Principles:**
- Ground truth from knowledge base (no hallucination)
- Confidence-based escalation (no guessing)
- Multi-turn memory (context-aware)
- Human-in-the-loop (safety net)

---

## System Architecture

### High-Level Flow

```
User Query
    │
    ├─ [Chat History] ──┐
    │                   │
    ▼                   │
┌─────────────────────────────────┐
│  RETRIEVAL (Vector DB)          │
│  ─────────────────────────────  │
│  1. Embed query                 │
│  2. Search Chroma               │
│  3. Top-5 similar chunks        │
└─────────────┬───────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Context Building   │
    │  ─────────────────  │
    │  Query              │
    │  + Retrieved Docs   │
    │  + Chat History     │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────────────────────┐
    │  GENERATION (Claude API)            │
    │  ─────────────────────────────────  │
    │  Model: Claude Haiku                │
    │  Prompt: System + Context + Query   │
    │  Output: Answer + <confidence>X</> │
    └──────────┬──────────────────────────┘
               │
    ┌──────────┴────────────┐
    │                       │
    ▼                       ▼
Confidence         Confidence
≥ 0.7              < 0.7
│                  │
▼                  ▼
RESPOND        ESCALATE
Return         Log to
Answer         JSON
             (Human Review)
```

### Component Details

#### 1. Retrieval Pipeline

**Input:** User query (string)

**Process:**
```python
query = "How do I access a course?"

# Step 1: Embed query using Chroma's default embeddings
embedding = embed(query)  # 384-dim vector

# Step 2: Search vector DB with cosine similarity
results = chroma.query(
    query_embeddings=[embedding],
    n_results=5
)

# Step 3: Return top-5 chunks with metadata
{
  "id": ["chunk_0", "chunk_1", ...],
  "documents": ["FAQ-01: ...", "POLICY-02: ...", ...],
  "distances": [0.15, 0.22, ...],  # Lower = more similar
  "metadatas": [{"doc_type": "FAQ"}, ...]
}
```

**Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions)
- General-purpose, fast
- Good for support domain
- Already cached in Chroma

**Search Metric:** Cosine Similarity
- Range: [0, 2] (0 = identical, 2 = opposite)
- Typical: 0.1-0.5 (we use top-5 regardless of distance)

**Output:** List of relevant chunks with metadata

---

#### 2. Context Building

**Input:** Query + Retrieved chunks + Chat history

**Process:**

```
System Prompt (fixed):
  "You are LearnForge AI Support. 
   Use ONLY the provided context. 
   Output confidence <confidence>X</confidence>"

Current Message:
  "RELEVANT DOCS:
   [Source 1: FAQ] FAQ-01 content...
   [Source 2: POLICY] POLICY-03 content...
   ...
   
   User Question: How do I access a course?"

Chat History (last 10 messages):
  [{"role": "user", "content": "..."}, 
   {"role": "assistant", "content": "..."},
   ...]
```

**Message Structure for LLM:**
```
[
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": "First question"},
  {"role": "assistant", "content": "First answer"},
  {"role": "user", "content": "Follow-up question"},
  {"role": "assistant", "content": "Follow-up answer"},
  {"role": "user", "content": "[CURRENT] context + query"}
]
```

**Output:** Formatted prompt ready for LLM

---

#### 3. Generation (LLM)

**Model:** Claude Haiku
- **Speed:** ~1-2 sec per request
- **Quality:** Good for support (8/10)
- **Cost:** $0.80 per 1M input tokens
- **Rate Limit:** No strict rate limit
- **Max Tokens:** 200K context window

**Prompt Structure:**
```
System:
  "You are LearnForge AI Support. Your job is to help users 
   with course access, refunds, policies using ONLY the 
   provided context.
   
   IMPORTANT: If context doesn't answer, respond with:
   'I'm not certain. Let me connect you with a human agent.'
   
   ALWAYS end with: <confidence>0.85</confidence>"

User Input:
  "[DOCS]
   [FAQ-01] How do I access course...
   ...
   User: How do I access a course?"
```

**Output:**
```
"After purchasing, your course appears in 'My Learning' 
within a few minutes. Make sure you're logged in with the 
correct email.

If it doesn't appear:
1. Sign out and sign back in
2. Refresh the page
3. Check mobile app sync

If still missing after 30 min, contact support with your 
receipt or transaction ID.

<confidence>0.92</confidence>"
```

**Confidence Extraction:**
```python
match = re.search(r'<confidence>([\d.]+)</confidence>', response)
if match:
    confidence = float(match.group(1))  # 0.92
    return max(0.0, min(1.0, confidence))  # Clamp to [0,1]
```

---

#### 4. Confidence & Escalation

**Threshold:** 0.7 (configurable in Streamlit)

**Logic:**
```
if confidence < 0.7:
  1. Display yellow warning to user
  2. Log to escalations.json
  3. Suggest human agent
else:
  1. Display green success badge
  2. Save to chat history
  3. Continue conversation
```

**Escalation Log Entry:**
```json
{
  "timestamp": "2026-09-18T14:30:00.123456",
  "user_query": "How do I get a refund?",
  "assistant_response": "I'm not certain about this...",
  "confidence": 0.45,
  "retrieved_sources": [
    {"doc_type": "POLICY", "source_file": "policies.md"},
    {"doc_type": "TICKET", "source_file": "tickets.md"}
  ]
}
```

---

#### 5. Multi-Turn Conversation

**Storage:** In-memory list in Streamlit session state

```python
st.session_state.chat_history = [
  {"role": "user", "content": "Q1"},
  {"role": "assistant", "content": "A1", "confidence": 0.85},
  {"role": "user", "content": "Q2"},
  {"role": "assistant", "content": "A2", "confidence": 0.91},
]
```

**Usage:**
- Last 10 messages passed to LLM for context
- Each new query benefits from prior questions
- History cleared on user request or app restart

**Example:**
```
User: "How do I access my course?"
Assistant: "..."

User: "What if I still can't see it after 30 minutes?"  
(LLM sees previous Q&A, understands context)
Assistant: "Based on your earlier question, contact support..."
```

---

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                      STREAMLIT FRONTEND                      │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ User Chat Interface                                      │ │
│ │ - Input box for questions                               │ │
│ │ - Display answers with confidence badge                 │ │
│ │ - Show retrieved sources (expandable)                    │ │
│ │ - Settings: confidence threshold, top_k                 │ │
│ │ - Escalation log download                               │ │
│ └──────────────────────────────────────────────────────────┘ │
└────┬─────────────────────────────────────────────────────────┘
     │
     │ user_query
     ▼
┌─────────────────────────────────────────┐
│    DATA LOADER & VECTOR DB              │
│ ┌─────────────────────────────────────┐ │
│ │ Load KB files (if first run)        │ │
│ │ faqs.md → 15 chunks                 │ │
│ │ policies.md → 10 chunks             │ │
│ │ tickets.md → 15 chunks              │ │
│ │ Total: ~40 chunks                   │ │
│ └─────────────────────────────────────┘ │
│              │                          │
│              ▼                          │
│ ┌─────────────────────────────────────┐ │
│ │ Chroma Vector DB (Local)            │ │
│ │ - embeddings: 384-dim vectors       │ │
│ │ - search: cosine similarity         │ │
│ │ - collection: "learnforge_kb"       │ │
│ │ - storage: ./chroma_db/             │ │
│ └─────────────────────────────────────┘ │
└────┬──────────────────────────────────────┘
     │
     │ query embedding
     ▼
┌─────────────────────────────────────┐
│   RETRIEVAL                         │
│ Query → Top-5 chunks               │
│ Distance: [0.15, 0.22, 0.31, ...] │
└────┬──────────────────────────────┘
     │
     │ retrieved_chunks + history
     ▼
┌──────────────────────────────────┐
│   GENERATION (Groq API)          │
│ - Model: Mixtral-8x7b            │
│ - Temp: 0.7 (default)            │
│ - Max tokens: 500                │
│ - Timeout: 30s                   │
└────┬───────────────────────────┘
     │
     │ answer + confidence
     ▼
┌──────────────────────────────────┐
│   PROCESSING                     │
│ - Extract confidence score       │
│ - Clean response text            │
│ - Check escalation threshold     │
└────┬───────────────────────────┘
     │
  ┌──┴──────────────────┐
  │                     │
  ▼                     ▼
CONFIDENT          ESCALATE
0.7+               <0.7
│                  │
├─ Return to UI   ├─ Log entry
├─ Save history   ├─ Warn user
└─ Continue       └─ Suggest human
```

---

## Database Schema

### Chroma Collection: "learnforge_kb"

**Documents Table:**
```
id          | document_text           | source_file  | doc_type | char_count
------------|-------------------------|--------------|----------|----------
chunk_0    | "FAQ-01: How do I..."   | faqs.md      | FAQ      | 1250
chunk_1    | "POLICY-02: Refund..." | policies.md  | POLICY   | 890
chunk_2    | "TICKET-03: User..." | tickets.md   | TICKET   | 1100
```

**Embeddings:**
```
id          | embedding (384-dim)
------------|--------------------
chunk_0    | [0.23, -0.45, 0.12, ..., 0.04]
chunk_1    | [0.18, -0.38, 0.09, ..., 0.02]
```

**Metadata:**
```json
{
  "chunk_0": {
    "source_file": "faqs.md",
    "doc_type": "FAQ",
    "char_count": 1250
  }
}
```

### Escalations Log (escalations.json)

```json
[
  {
    "timestamp": "2026-09-18T14:30:00.123",
    "user_query": "How do I return a course?",
    "assistant_response": "I'm not certain about returns...",
    "confidence": 0.45,
    "retrieved_sources": [
      {
        "doc_type": "FAQ",
        "source_file": "faqs.md"
      }
    ]
  }
]
```

### Chat History (in-memory)

```python
[
  {"role": "user", "content": "...", "timestamp": "..."},
  {"role": "assistant", "content": "...", "confidence": 0.85, "timestamp": "..."}
]
```

---

## Error Handling

### Scenario 1: LLM Error
```
try:
  response = groq_client.messages.create(...)
except Exception as e:
  return {
    "answer": "I encountered an error. Let me connect you...",
    "confidence": 0.0
  }
  # → Escalates due to confidence = 0
```

### Scenario 2: No Retrieved Documents
```
retrieved = db.search(query, top_k=5)
# Returns empty list if no good matches

context_text = "No relevant docs found"

# LLM sees no context, should say "I'm not sure"
# → Escalates if LLM still outputs confidence < 0.7
```

### Scenario 3: Malformed Confidence Score
```
response = "Here's the answer... <confidence>abc</confidence>"

match = re.search(r'<confidence>([\d.]+)</confidence>', response)
if not match:
  confidence = 0.5  # Default to neutral
```

### Scenario 4: Rate Limit Hit (Claude API)
```
try:
  response = anthropic_client.messages.create(...)
except RateLimitError:
  # User sees "System busy, please try again"
  # For demo: rate limit unlikely with normal usage
```

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Embed query | 50-100ms | TF-IDF vectorization |
| Vector search (top-5) | 10-50ms | SQLite cosine similarity |
| LLM inference | 1-2 sec | Claude Haiku |
| **Total latency** | **1.5-2.5 sec** | User sees delay, acceptable |

| Metric | Value |
|--------|-------|
| KB size | ~40 chunks (9 KB text) |
| Avg chunk size | 250 words |
| Embedding dim | 384 |
| Storage (Chroma) | ~2 MB |
| Storage (conversation) | ~1 KB per exchange |

---

## Scalability Limitations

### Current (SQLite + TF-IDF)
- ✅ 0-10K documents
- ✅ 1 concurrent user
- ✅ Hundreds of requests per day

### At 100K documents
- ❌ SQLite search still OK (~100ms)
- ❌ SQLite storage grows to ~500 MB
- ❌ No distributed search
- **Solution:** Migrate to Pinecone/Weaviate Cloud

### At 1M documents
- ❌ Cannot index locally
- ❌ Need sharding/partitioning
- **Solution:** Multi-shard Weaviate or Elasticsearch

### At 10M+ requests/day
- ❌ Claude API rate limits
- ❌ Single Streamlit instance
- **Solution:** FastAPI backend + queue + horizontal scaling

---

## Trade-offs Made

| Decision | Why This | Downside | When to Change |
|----------|----------|----------|-----------------|
| SQLite + TF-IDF | Fast setup | No cloud scale | 10K+ docs |
| Claude Haiku | Cheap & fast | Lower accuracy than Opus | Budget available |
| LLM self-scores confidence | Simple | Overconfident | With validation data |
| Top-5 retrieval, no re-rank | Speed | May miss best doc | More budget for inference |
| JSON escalation log | Simplicity | Manual review | Need real-time alerts |
| Streamlit | Quick UI | Single-user | Production deployment |
| No auth | Demo focus | Security risk | Production deployment |
| 4-6 hour scope | Time constraint | Limited features | Post-launch iterations |

---

## Future Improvements (Post-Launch)

### High Priority
1. **User feedback loop**: Track "Helpful?" feedback on answers
2. **Continuous learning**: Use escalations + feedback to fine-tune system
3. **Better confidence**: Train separate confidence model on real data
4. **Re-ranking**: Add LLM re-ranker for top-5 chunks

### Medium Priority
1. **Production deployment**: FastAPI + Redis + horizontal scaling
2. **Slack integration**: Real-time escalation alerts to support team
3. **Analytics**: Dashboard of common escalation reasons
4. **KB updates**: Auto-sync updated FAQ/policy docs

### Low Priority
1. **Multi-language**: Support non-English queries
2. **Voice input**: Phone/voice support
3. **Custom embeddings**: Fine-tune embedding model on support domain

---

## Conclusion

This system prioritizes **safety over performance**:
- Grounds responses in knowledge base (no hallucination)
- Escalates when uncertain (human safety net)
- Maintains context (helpful multi-turn)
- Logs everything (auditability)

Trade-offs favor **simplicity for 4-6 hour sprint**, with clear paths to scaling.

---

**Document Date:** 2026-09-18  
**Version:** 1.0
