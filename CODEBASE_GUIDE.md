# LearnForge AI Support - Detailed Codebase Guide

## 📁 Project Structure

```
edversity/
├── app.py                    # Streamlit UI (main app - what users see)
├── data_loader.py            # Parse knowledge base files into chunks
├── vector_db.py              # Vector database (SQLite + TF-IDF search)
├── llm_handler.py            # Claude API integration + confidence scoring
├── requirements.txt          # Python dependencies
├── .env                      # API keys (user adds their own)
├── README.md                 # Full documentation
├── SYSTEM_DESIGN.md          # Architecture & design decisions
├── QUICK_START.md            # Setup instructions
├── CODEBASE_GUIDE.md         # This file
├── knowledge-base/           # Sample data (provided)
│   ├── faqs.md              # 15 FAQ entries
│   ├── policies.md          # 10 policy documents
│   └── tickets.md           # 15 support tickets
├── vector_db.sqlite          # Vector DB storage (auto-created)
├── escalations.json          # Escalation logs (auto-created)
└── chroma_db/               # (old, not used anymore)
```

---

## 🔄 How It All Works Together

```
User Types Question in Streamlit
        ↓
app.py receives input
        ↓
data_loader.py loads KB (first time only)
        ↓
vector_db.py searches for relevant docs
        ↓
llm_handler.py calls Claude API with context
        ↓
Claude generates answer + confidence score
        ↓
app.py displays answer or escalates
        ↓
escalations.json logs low-confidence queries
```

---

## 📄 File-by-File Detailed Explanation

### **1. app.py** (Streamlit UI - 150 lines)

**What it does:** This is the main app - the web interface users interact with.

**Key Components:**

```python
import streamlit as st  # Web UI framework
import json
import os
from datetime import datetime
from pathlib import Path
from vector_db import VectorDB  # Import search
from llm_handler import LLMHandler  # Import LLM
from dotenv import load_dotenv

# Load .env file (reads ANTHROPIC_API_KEY)
load_dotenv()

# Set up page title and layout
st.set_page_config(page_title="LearnForge AI Support", layout="wide")
```

**Session State Management:**

```python
# Session state = data that persists during user's browser session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores all messages

if "db" not in st.session_state:
    st.session_state.db = VectorDB()  # Create vector DB once
    st.session_state.db.initialize()  # Load KB

if "llm" not in st.session_state:
    st.session_state.llm = LLMHandler()  # Create LLM once
```

**Why?** Streamlit re-runs the entire script on every interaction. Session state preserves data between runs.

**Sidebar Settings:**

```python
with st.sidebar:
    # Slider to adjust confidence threshold (0-1)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
    
    # Slider to control how many KB chunks to retrieve (1-10)
    top_k = st.slider("Number of Retrieved Documents", 1, 10, 5)
    
    # Button to clear chat
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
    
    # Button to download escalation log as JSON
    if st.button("Download Escalation Log"):
        # Read escalations.json and let user download
```

**Display Chat History:**

```python
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):  # "user" or "assistant"
        st.write(message["content"])  # Display text
        
        # If assistant message, show confidence badge
        if message["role"] == "assistant" and "confidence" in message:
            if message["confidence"] < confidence_threshold:
                st.warning(f"Low Confidence: {message['confidence']:.2f}")
            else:
                st.success(f"Confidence: {message['confidence']:.2f}")
```

**Process User Input:**

```python
# Streamlit's built-in chat input widget
if user_input := st.chat_input("Ask me anything..."):
    
    # 1. Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # 2. Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # 3. Process in assistant message block
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            
            # RETRIEVE: Search for relevant KB chunks
            retrieved = st.session_state.db.search(user_input, top_k=top_k)
            
            # GENERATE: Call Claude with context
            result = st.session_state.llm.generate_answer(
                user_input,
                retrieved,
                st.session_state.chat_history[:-1]  # All prior messages
            )
            
            answer = result["answer"]
            confidence = result["confidence"]
            
            # Display answer
            st.write(answer)
            
            # 4. CHECK CONFIDENCE
            if confidence < confidence_threshold:
                # LOW CONFIDENCE = ESCALATE
                st.warning(f"Low Confidence: {confidence:.2f}")
                
                # Log to escalations.json
                escalation_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "user_query": user_input,
                    "assistant_response": answer,
                    "confidence": confidence,
                    "retrieved_sources": [...]
                }
                
                # Append to escalations.json file
                escalations = []
                if Path("escalations.json").exists():
                    with open("escalations.json") as f:
                        escalations = json.load(f)
                escalations.append(escalation_entry)
                with open("escalations.json", 'w') as f:
                    json.dump(escalations, f, indent=2)
            else:
                # HIGH CONFIDENCE = SHOW SUCCESS
                st.success(f"Confidence: {confidence:.2f}")
            
            # Show retrieved sources in expandable section
            with st.expander("View Retrieved Sources"):
                for i, chunk in enumerate(retrieved, 1):
                    st.markdown(f"**[{i}] {chunk['metadata']['doc_type']}**")
                    st.text(chunk['content'][:500] + "...")
            
            # 5. Save to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "confidence": confidence
            })
```

---

### **2. data_loader.py** (KB Parser - 50 lines)

**What it does:** Reads the knowledge base files (faqs.md, policies.md, tickets.md) and splits them into searchable chunks.

**Key Logic:**

```python
def load_knowledge_base():
    """Load and parse all knowledge base files into chunks."""
    
    kb_path = Path("knowledge-base")
    chunks = []
    chunk_id = 0
    
    # Define which files to read and their document types
    files = {
        "faqs.md": "FAQ",
        "policies.md": "POLICY",
        "tickets.md": "TICKET"
    }
    
    for filename, doc_type in files.items():
        filepath = kb_path / filename
        
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by "---" to get individual entries
        # (Each FAQ/policy/ticket is separated by ---)
        entries = content.split('\n---\n')
        
        for entry in entries:
            entry = entry.strip()
            
            # Skip very short entries (noise)
            if len(entry) < 50:
                continue
            
            # Create chunk with metadata
            chunks.append({
                "id": chunk_id,
                "content": entry,  # Full text
                "source": filename,  # Where it came from
                "doc_type": doc_type,  # FAQ/POLICY/TICKET
                "metadata": {
                    "source_file": filename,
                    "doc_type": doc_type,
                    "char_count": len(entry)
                }
            })
            chunk_id += 1
    
    return chunks  # Returns list of ~40 chunks
```

**Example Output:**
```
[
  {
    "id": 0,
    "content": "# FAQ-01 — How do I access a course...",
    "source": "faqs.md",
    "doc_type": "FAQ",
    "metadata": {"source_file": "faqs.md", "doc_type": "FAQ", ...}
  },
  {
    "id": 1,
    "content": "# FAQ-02 — Can I get a refund...",
    ...
  },
  ...
]
```

---

### **3. vector_db.py** (Search Engine - 130 lines)

**What it does:** Creates a searchable database of knowledge base chunks using TF-IDF embeddings. When a user asks a question, it finds the most relevant chunks.

**Key Concepts:**

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Converts text into numerical vectors
   - Gives high scores to words that are important in a doc but rare across all docs
   - Example: "refund" appears in many docs, but "voucher" appears in few → "voucher" gets higher weight

2. **Cosine Similarity**
   - Measures how similar two vectors are (0-1 scale)
   - Higher = more similar

**Detailed Code:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import numpy as np

class VectorDB:
    def __init__(self, db_path="vector_db.sqlite"):
        self.db_path = db_path
        
        # TfidfVectorizer converts text → numerical vectors
        # max_features=500: Use top 500 words (skip rare words)
        # stop_words='english': Ignore common words like "the", "is"
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        
        self.embeddings_matrix = None  # Will store all chunk embeddings
        self._init_db()  # Create SQLite table
    
    def _init_db(self):
        """Create SQLite database to store chunks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table with 4 columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,              # "chunk_0", "chunk_1", etc
                content TEXT NOT NULL,            # Full text
                source_file TEXT NOT NULL,        # "faqs.md", etc
                doc_type TEXT NOT NULL            # "FAQ", "POLICY", "TICKET"
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def initialize(self):
        """Load KB into vector DB (first time only)."""
        
        # Check if already initialized
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM chunks')
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            print(f"[OK] Vector DB already initialized with {count} documents")
            self._load_embeddings()  # Load embeddings from disk
            return
        
        # First time: load KB
        chunks = load_knowledge_base()  # Get ~40 chunks from files
        
        # Insert into SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for chunk in chunks:
            cursor.execute('''
                INSERT INTO chunks (id, content, source_file, doc_type)
                VALUES (?, ?, ?, ?)
            ''', (
                f"chunk_{chunk['id']}",
                chunk['content'],
                chunk['metadata']['source_file'],
                chunk['metadata']['doc_type']
            ))
        
        conn.commit()
        conn.close()
        
        # Create TF-IDF embeddings
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Convert all chunks to numerical vectors."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT content FROM chunks ORDER BY id')
        rows = cursor.fetchall()
        conn.close()
        
        # Extract just the text
        texts = [row[0] for row in rows]
        
        # TfidfVectorizer learns from all texts
        # Outputs: matrix of shape (num_docs, 500)
        # Each row = one document's TF-IDF vector
        self.embeddings_matrix = self.vectorizer.fit_transform(texts)
    
    def search(self, query, top_k=5):
        """Search for relevant chunks."""
        
        if not self.is_initialized:
            self.initialize()
        
        # 1. Convert query to TF-IDF vector
        # Shape: (1, 500) - same format as documents
        query_embedding = self.vectorizer.transform([query])
        
        # 2. Calculate cosine similarity between query and all documents
        # Result: array of similarities, one per document
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        
        # 3. Get indices of top-k most similar documents
        # argsort()[::-1] = sort descending, [:top_k] = take first k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 4. Fetch actual chunks from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, content, source_file, doc_type FROM chunks ORDER BY id')
        all_rows = cursor.fetchall()
        conn.close()
        
        # 5. Build results
        results = []
        for idx in top_indices:
            if idx < len(all_rows):
                _, content, source_file, doc_type = all_rows[idx]
                results.append({
                    "content": content,
                    "metadata": {
                        "source_file": source_file,
                        "doc_type": doc_type,
                        "char_count": len(content)
                    },
                    "similarity": float(similarities[idx])
                })
        
        return results
```

**Example Flow:**

```
User asks: "How do I access my course?"

1. Vectorizer converts to TF-IDF:
   [0.2, 0.0, 0.15, 0.1, 0.0, ..., 0.3]  (500 dimensions)

2. Compare with all 40 chunks:
   Chunk 0 (FAQ-01 about access): similarity = 0.85 ✓
   Chunk 1 (FAQ-02 about refunds): similarity = 0.12
   Chunk 2 (POLICY about payments): similarity = 0.08
   ...
   Chunk 39: similarity = 0.05

3. Return top-5:
   [
     {content: "FAQ-01...", similarity: 0.85},
     {content: "FAQ-10...", similarity: 0.32},
     {content: "POLICY-02...", similarity: 0.28},
     ...
   ]
```

---

### **4. llm_handler.py** (Claude AI Integration - 100 lines)

**What it does:** Calls Claude API to generate answers and extract confidence scores.

**Key Logic:**

```python
from anthropic import Anthropic
import re

class LLMHandler:
    def __init__(self, api_key=None):
        # Get API key from .env file
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        # Create Anthropic client
        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-haiku-4-5-20251001"  # Fast, cheap model
        
        # System prompt tells Claude how to behave
        self.system_prompt = """You are LearnForge AI Support Assistant...
        
IMPORTANT RULES:
1. ONLY use the provided context documents
2. If context doesn't answer: say "I'm not certain"
3. Include confidence score: <confidence>0.85</confidence>
4. Be concise and helpful
"""
    
    def generate_answer(self, user_query, retrieved_context, conversation_history):
        """Generate answer using Claude."""
        
        # 1. FORMAT CONTEXT
        context_text = "RELEVANT KNOWLEDGE BASE:\n"
        for i, chunk in enumerate(retrieved_context, 1):
            context_text += f"\n[Source {i}: {chunk['metadata']['doc_type']}]\n"
            context_text += chunk['content'] + "\n"
        
        # 2. BUILD MESSAGE HISTORY
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add prior conversation (for multi-turn context)
        for msg in conversation_history[-10:]:  # Keep last 10 messages
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current question with context
        messages.append({
            "role": "user",
            "content": f"{context_text}\n\nUser Question: {user_query}"
        })
        
        # 3. CALL CLAUDE API
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,  # Max response length
                messages=messages
            )
            
            # Extract text from response
            full_response = response.choices[0].message.content
            
            # 4. EXTRACT CONFIDENCE SCORE
            # Look for <confidence>0.XX</confidence> tag
            confidence = self._extract_confidence(full_response)
            
            # 5. CLEAN RESPONSE
            # Remove confidence tag for display
            clean_response = re.sub(
                r'<confidence>[\d.]+</confidence>', 
                '', 
                full_response
            ).strip()
            
            return {
                "answer": clean_response,
                "confidence": confidence,
                "raw_response": full_response
            }
        
        except Exception as e:
            # If error, return low confidence so it escalates
            return {
                "answer": "I encountered an error. Let me connect you...",
                "confidence": 0.0,
                "raw_response": str(e)
            }
    
    def _extract_confidence(self, response_text):
        """Extract confidence score from Claude's response."""
        
        # Look for <confidence>0.XX</confidence> pattern
        match = re.search(r'<confidence>([\d.]+)</confidence>', response_text)
        
        if match:
            try:
                score = float(match.group(1))
                # Clamp to [0.0, 1.0]
                return max(0.0, min(1.0, score))
            except:
                return 0.5  # Default if parsing fails
        
        return 0.5  # Default if no confidence tag found
```

**Example Interaction:**

```
User Query: "How do I access my course?"

Retrieved Context:
[Source 1: FAQ]
FAQ-01 — How do I access a course after purchasing it?
QUESTION: I purchased a LearnForge course, but I don't see it...
ANSWER: After completing a purchase, your course should...

[Source 2: FAQ]
FAQ-10 — Can I share my course account with someone else?
...

Messages sent to Claude:
[
  {"role": "system", "content": "You are LearnForge AI Support..."},
  {"role": "user", "content": "[CONTEXT]\n\nUser Question: How do I access my course?"}
]

Claude's Response:
"After purchasing a course, it should appear in your 'My Learning' 
dashboard within a few minutes. Make sure you're signed in with 
the same email you used for checkout...

<confidence>0.92</confidence>"

Extraction:
- answer: "After purchasing a course, it should appear..."
- confidence: 0.92
```

---

## 🔗 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ USER TYPES QUESTION IN STREAMLIT (app.py)              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
         ┌──────────────────┐
         │ First Time Only: │
         │ data_loader.py   │
         │ Reads KB files   │
         │ → 40 chunks      │
         └────────┬─────────┘
                  │ (inserts into vector_db.sqlite)
                  ▼
    ┌────────────────────────────┐
    │ vector_db.py               │
    │ - TfidfVectorizer          │
    │ - embeddings_matrix        │
    │ - SQLite chunks table      │
    └────┬───────────────────────┘
         │
         │ (User types question)
         │
         ▼
    ┌──────────────────────────┐
    │ vector_db.search()       │
    │ - Embed query            │
    │ - Cosine similarity      │
    │ - Top-5 chunks           │
    └────┬─────────────────────┘
         │ (Returns: [chunk1, chunk2, ...])
         │
         ▼
    ┌──────────────────────────┐
    │ llm_handler.py           │
    │ - Format context         │
    │ - Build messages         │
    │ - Call Claude API        │
    │ - Extract confidence     │
    └────┬─────────────────────┘
         │ (Returns: {answer, confidence})
         │
         ▼
    ┌────────────────────────────┐
    │ app.py                     │
    │ - Display answer           │
    │ - Check confidence < 0.7?  │
    │ - If YES: Log escalation   │
    │ - Save to chat_history     │
    └────────────────────────────┘
```

---

## 🎯 Key Algorithms Explained

### **1. TF-IDF Search**

**Problem:** How do we find relevant documents?

**Solution:** Convert documents and queries to numerical vectors, then measure similarity.

**Steps:**

1. **Tokenize:** Split text into words
   ```
   "How do I access my course?" 
   → ["how", "do", "i", "access", "my", "course"]
   ```

2. **Count term frequency (TF):** How often does each word appear?
   ```
   Document 1: "How do I access a course?"
   TF: {how: 1, do: 1, i: 1, access: 1, a: 1, course: 1}
   ```

3. **Calculate inverse document frequency (IDF):** How rare is each word across all documents?
   ```
   "access" appears in 3/40 docs → IDF = log(40/3) = high weight
   "the" appears in 35/40 docs → IDF = log(40/35) = low weight
   ```

4. **Combine TF × IDF:** Final weight for each word
   ```
   Word "access" in doc 1: TF(1) × IDF(high) = high score
   Word "the" in doc 1: TF(1) × IDF(low) = low score
   ```

5. **Create vector:** One number per important word
   ```
   Document vector = [0.5, 0.0, 0.3, 0.8, 0.0, ...] (500 dimensions)
   Query vector    = [0.4, 0.0, 0.2, 0.9, 0.0, ...]
   ```

6. **Calculate similarity:** Dot product of vectors
   ```
   similarity = 0.5×0.4 + 0.0×0.0 + 0.3×0.2 + 0.8×0.9 + ...
              = 0.85  (high = similar!)
   ```

### **2. Confidence Scoring**

**Problem:** How do we know if Claude's answer is trustworthy?

**Solution:** Claude outputs a confidence score, which we extract and validate.

**Flow:**

```
Claude generates:
"After purchasing, the course appears in My Learning within minutes...
<confidence>0.92</confidence>"

Extraction:
1. Regex search for <confidence>[\d.]+</confidence>
2. Extract number: "0.92"
3. Convert to float: 0.92
4. Validate: 0 ≤ 0.92 ≤ 1.0 ✓
5. Return: 0.92

Decision:
- If 0.92 >= 0.7: Show green ✓ badge, save answer
- If 0.92 < 0.7:  Show yellow ⚠️ warning, log escalation
```

### **3. Multi-turn Conversation**

**Problem:** How do we remember prior messages?

**Solution:** Keep conversation history and pass it to Claude each time.

**Example:**

```
Turn 1:
User: "How do I access my course?"
Claude: "After purchasing, it appears in My Learning..." (0.92)
Save to history.

Turn 2:
User: "What if it still doesn't appear?"
Messages sent to Claude:
  [
    {"role": "system", "content": "You are LearnForge..."},
    {"role": "user", "content": "How do I access my course?"},
    {"role": "assistant", "content": "After purchasing..."},
    {"role": "user", "content": "What if it still doesn't appear?"}
  ]

Claude sees prior context, gives follow-up answer.
```

---

## 🔐 Security & Error Handling

### **API Key Management:**
```python
# .env file (NOT in git)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx

# Python code
api_key = os.getenv("ANTHROPIC_API_KEY")  # Reads from .env
client = Anthropic(api_key=api_key)
```

### **Error Handling:**
```python
try:
    response = self.client.messages.create(...)
except Exception as e:
    # Return error message with 0 confidence → escalates
    return {"answer": "Error occurred...", "confidence": 0.0}
```

### **Escalation Logic:**
```python
if confidence < 0.7:  # Threshold from sidebar
    # Log to escalations.json
    escalation = {
        "timestamp": "2026-09-19T10:30:00",
        "user_query": "How do I...",
        "confidence": 0.45,
        "retrieved_sources": [...]
    }
    # Append to file
```

---

## 📊 Performance Characteristics

| Component | Time | Notes |
|-----------|------|-------|
| Data load (first run) | 2-3 sec | ~40 chunks, no cache |
| Vectorization (first run) | 1 sec | TF-IDF fitting |
| Search | 50-100ms | Cosine similarity on 40 vecs |
| LLM inference | 1-2 sec | Claude Haiku is fast |
| **Total per query** | **1.5-2.5 sec** | Cached after first run |

---

## 🚀 How to Extend

### **Add New KB Document:**
1. Add text to `knowledge-base/faqs.md` (or other file)
2. Separate with `---`
3. Delete `vector_db.sqlite` to force re-indexing
4. Restart app → will re-read KB

### **Change Confidence Threshold:**
```python
# In app.py sidebar:
threshold = st.slider("Threshold", 0.0, 1.0, 0.7)
# Users can adjust on the fly!
```

### **Change LLM Model:**
```python
# In llm_handler.py:
self.model = "claude-opus-4-1-20250805"  # Stronger model
```

### **Add Slack Integration:**
```python
# Log escalations to Slack instead of JSON:
import requests

def log_escalation_to_slack(message):
    requests.post(SLACK_WEBHOOK_URL, json={"text": message})
```

---

## 📝 Summary

**The system is a 4-stage pipeline:**

1. **Data Loader** - Parse KB files into chunks
2. **Vector DB** - Search for relevant chunks using TF-IDF
3. **LLM Handler** - Call Claude with context + extract confidence
4. **Streamlit App** - Display answers, log escalations, maintain chat history

All components work together to create a reliable, human-in-the-loop support system.

