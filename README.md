# LearnForge AI Support Assistant

An AI-powered customer support assistant that answers user questions using retrieval-augmented generation (RAG) with a vector database, multi-turn conversations, confidence scoring, and human escalation.

## 🎯 Features

- **Retrieval-Augmented Generation (RAG)**: Retrieves relevant documents from knowledge base before generating responses
- **Vector Database**: Uses Chroma for fast semantic search over knowledge base
- **Multi-turn Conversations**: Maintains chat history for context-aware responses
- **Confidence Scoring**: LLM assigns confidence scores; low-confidence answers are escalated
- **Human Escalation**: Automatically logs low-confidence queries for human review
- **Streamlit UI**: Interactive web interface for testing and demonstration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query (Streamlit UI)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────┐
    │  Vector DB (SQLite + SentenceTransformers)
    │  - Embed query with all-MiniLM-L6-v2  │
    │  - Search top-5 similar chunks        │
    │  - Cosine similarity ranking          │
    └────────────┬───────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────┐
    │  LLM Generation (Groq - Mixtral-8x7b)   │
    │  - Input: Query + Context + History     │
    │  - Output: Answer + Confidence Score    │
    └────────────┬─────────────────────────────┘
                 │
         ┌───────┴───────┐
         │               │
    Confidence        Confidence
    ≥ 0.7             < 0.7
    │                 │
    ▼                 ▼
Return Answer    Log Escalation
                 (escalations.json)
```

## 📊 Data Schema

### Knowledge Base Structure
Files in `knowledge-base/`:
- `faqs.md` - 15 Q&A entries (FAQ)
- `policies.md` - 10 policy excerpts (POLICY)
- `tickets.md` - 15 past support tickets (TICKET)

### Chunk Schema (stored in Chroma)
```json
{
  "id": "chunk_0",
  "content": "Full text of FAQ/policy/ticket",
  "metadata": {
    "source_file": "faqs.md",
    "doc_type": "FAQ",
    "char_count": 1234
  }
}
```

### Escalation Log (escalations.json)
```json
[
  {
    "timestamp": "2026-09-18T14:30:00",
    "user_query": "How do I...?",
    "assistant_response": "I'm not certain...",
    "confidence": 0.45,
    "retrieved_sources": [
      {
        "doc_type": "POLICY",
        "source_file": "policies.md"
      }
    ]
  }
]
```

## 🚀 Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create .env File
```bash
# Copy template
cp .env.example .env

# Edit .env and add your Anthropic API key
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
```


### 3. Run the App
```bash
streamlit run app.py
```

The app will:
- Load knowledge base files
- Initialize SQLite vector DB with TF-IDF (one-time setup)
- Start interactive chat interface at `http://localhost:8501`

## 🎮 Usage

1. **Ask Questions**: Type your question in the chat input
2. **View Context**: Click "View Retrieved Sources" to see which KB docs were used
3. **Monitor Confidence**: Green checkmark = confident, Yellow warning = escalated
4. **Review Escalations**: Check sidebar "Escalation Review" section for low-confidence queries

### Example Queries
- "How do I access my course?"
- "Can I get a refund?"
- "Why isn't my progress saving?"
- "What's your refund policy?"

## ⚠️ Failure Handling

### Low Confidence Answers (< 0.7)
- **Detection**: LLM outputs confidence score
- **Action**: Logged to `escalations.json` for human review
- **User Experience**: Shows warning and suggests human agent

### Missing or Stale Data
- **Detection**: Retrieved chunks don't match query intent
- **Fallback**: System prompts LLM to say "I'm not certain" rather than hallucinate
- **Result**: Escalates to human

### Bad Retrieval (Wrong Documents)
- **Symptom**: Semantic search returns irrelevant chunks
- **Mitigation**: Using Chroma's embedding quality + top-5 retrieval reduces false positives
- **Monitoring**: Review `retrieved_sources` in escalation logs to detect retrieval failures

## 📈 Evaluation Plan

### Metrics
1. **Accuracy**: Manually review 20 sample Q&A pairs - % with correct answers
2. **Hallucination Rate**: % of responses that fabricate info not in KB
3. **Retrieval Quality**: % of top-5 chunks that are relevant to query
4. **Escalation Precision**: % of escalations that genuinely needed human review
5. **Confidence Calibration**: Confidence score vs actual correctness

### Test Dataset
```
Query: "How do I access a course?"
Expected: Mentions login, email verification, mobile app sync
Actual: [See escalations.json for sample responses]

Query: "What's your refund policy?"
Expected: Within 14 days, hasn't consumed significantly
Actual: [See escalations.json for sample responses]
```

### Evaluation Process
1. Run 20 diverse queries through the system
2. Log results in `eval_results.json`
3. Calculate metrics above
4. Review escalations for false positives/negatives

## 🔄 Trade-offs & Design Decisions

### Why SQLite + TF-IDF (not Pinecone/Chroma)?
- **Trade-off**: Local storage vs cloud scalability
- **Decision**: SQLite wins for 4-6 hour sprint
- **Cost**: Free, no setup, no dependencies
- **Scale Limit**: Works for <10K docs; would need cloud DB at 100K+ docs
- **With More Time**: Migrate to Pinecone or Weaviate for infinite scale & neural embeddings

### Why Claude Haiku (Anthropic)?
- **Trade-off**: Speed vs quality
- **Decision**: Haiku is fast (1-2 sec) and cost-effective
- **Accuracy**: Good for support tasks (80% of Opus at 10% cost)
- **With More Time**: Use Claude Opus for higher accuracy (3-5 sec, more expensive)

### Why Confidence in LLM Output (not external scorer)?
- **Trade-off**: Simplicity vs accuracy
- **Decision**: LLM self-scores (simpler, faster)
- **Limitation**: LLM can overstate confidence
- **With More Time**: Add separate confidence model (trains on correctness data)

### Why Top-5 Retrieval (not re-ranking)?
- **Trade-off**: Speed vs precision
- **Decision**: Simple cosine similarity + top-5 (no LLM re-ranker)
- **Cost**: Groq rate limit is 10 req/min; re-ranking adds latency
- **With More Time**: Add LLM re-ranker to pick best 1-2 of top-5

### Why Manual Escalation Logging (not Slack/Email)?
- **Trade-off**: Integration vs simplicity
- **Decision**: JSON file (easier for demo)
- **With More Time**: Add Slack webhook to notify support team in real-time

### Why No User Authentication?
- **Trade-off**: Security vs rapid prototyping
- **Decision**: Single-user app for demo; no auth needed
- **With More Time**: Add OAuth + user session tracking for prod

## 🐛 Known Limitations

1. **Hallucination Risk**: LLM may invent details not in KB; mitigation is prompt + escalation threshold
2. **Knowledge Base Conflicts**: FAQs/policies may contradict; system flags these for human review
3. **Rate Limiting**: Claude API has rate limits; sufficient for demo (hundreds of queries)
4. **No Feedback Loop**: Escalations logged but not used to improve system automatically
5. **Static KB**: Knowledge base not auto-updated; requires manual re-indexing and restart

## 🔐 Security Notes

- Groq API key stored in `.env` (not in git)
- No user data persistence
- Escalation logs contain sanitized queries only
- No external APIs except Groq

## 📦 Dependencies

- **streamlit** - Web UI
- **anthropic** - Claude API client
- **scikit-learn** - TF-IDF vectorization
- **python-dotenv** - Environment variables
- **numpy** - Math utilities

## 📝 Files

```
edversity/
├── app.py                # Streamlit UI
├── data_loader.py        # Parse KB files
├── vector_db.py          # SQLite + TF-IDF search
├── llm_handler.py        # Claude API + confidence scoring
├── .env.example          # Template (copy to .env)
├── .env                  # Your API keys (add manually)
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── SYSTEM_DESIGN.md      # Architecture & diagrams
├── CODEBASE_GUIDE.md     # Detailed code explanation
├── QUICK_START.md        # Setup guide
├── escalations.json      # Logged escalations (auto-generated)
├── vector_db.sqlite      # Vector DB file (auto-generated)
└── knowledge-base/       # Given sample data
    ├── README.md
    ├── faqs.md
    ├── policies.md
    └── tickets.md
```

## 🚀 Next Steps

1. Set `ANTHROPIC_API_KEY` in `.env` (get from https://console.anthropic.com/account/keys)
2. Run `streamlit run app.py`
3. Ask questions and test the system
4. Review escalations in sidebar
5. Check `SYSTEM_DESIGN.md` for architecture details
6. See `CODEBASE_GUIDE.md` for detailed code explanation

---

**Built for LearnForge AI/LLM Engineer Take-Home Assignment**
