# Quick Start Guide

## 🚀 Get Running in 3 Steps

### Step 1: Add Your Groq API Key

Edit `.env` file and paste your Groq API key:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

**Don't have a key?** Get one free at: https://console.groq.com/keys

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `streamlit` — web UI
- `chromadb` — vector database
- `groq` — LLM API
- `python-dotenv` — environment variables
- `numpy` — math utilities

**Expected time:** 1-2 minutes

### Step 3: Run the App

```bash
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Streamlit will automatically open your browser. If not, go to: **http://localhost:8501**

---

## 🎮 Test It

### First Run (Initialization)
When you first run the app:
1. It loads knowledge base files (`faqs.md`, `policies.md`, `tickets.md`)
2. Creates embeddings (~40 chunks)
3. Initializes Chroma vector DB
4. **Then** you can start chatting

**Expected time:** 5-10 seconds

### Try These Questions

✅ "How do I access my course?"  
✅ "What's your refund policy?"  
✅ "Why isn't my progress saving?"  
✅ "Can I get a refund after 20 days?"

### Watch For

- 🟢 **Green checkmark** = Confident answer (≥0.7)
- 🟡 **Yellow warning** = Low confidence (<0.7) → Escalates
- 📚 **View Retrieved Sources** = Click to see which KB docs were used

---

## 🔍 Verify Setup

### Check 1: Dependencies
```bash
python -c "import streamlit; import chromadb; import groq; print('✓ All imports OK')"
```

### Check 2: Groq API Key
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); key = os.getenv('GROQ_API_KEY'); print('✓ GROQ_API_KEY set' if key else '❌ Missing GROQ_API_KEY')"
```

### Check 3: Knowledge Base Files
```bash
ls -la knowledge-base/
# Should show: faqs.md, policies.md, tickets.md, README.md
```

---

## ⚠️ Troubleshooting

### Error: "GROQ_API_KEY not found"
**Fix:** Add your key to `.env` file and restart the app

### Error: "No module named 'streamlit'"
**Fix:** Run `pip install -r requirements.txt`

### Error: "Connection refused" on first run
**Reason:** Chroma DB initializing (takes 5-10 sec)  
**Fix:** Wait 10 seconds and refresh the page

### App runs but no responses
**Reason:** Groq API key invalid or rate limit hit  
**Fix:** Check API key, or wait 1 minute for rate limit reset (10 req/min)

### Vector DB persists between runs?
**Yes!** The `chroma_db/` folder stores embeddings. Delete it if you want a fresh start:
```bash
rm -rf chroma_db/
```

---

## 📊 What's Happening Behind the Scenes

```
You type question
        ↓
App embeds your query
        ↓
Chroma searches for similar docs (top-5)
        ↓
Groq LLM generates answer using docs + chat history
        ↓
Extract confidence score
        ↓
If confidence < 0.7 → Log escalation
If confidence ≥ 0.7 → Show answer
```

**Total time:** 1.5-2.5 seconds per response

---

## 📈 Next: Production Ready?

After testing, read:
- `README.md` — Full documentation
- `SYSTEM_DESIGN.md` — Architecture & trade-offs
- Check `escalations.json` — Review what got escalated

---

## 🆘 Need Help?

1. **Check README.md** — Full docs and architecture
2. **Review SYSTEM_DESIGN.md** — How it works
3. **Check code comments** — Each file has docstrings
4. **Test individual components:**
   ```bash
   python data_loader.py      # Test KB loading
   python vector_db.py        # Test Chroma search
   python llm_handler.py      # Test LLM + confidence
   ```

---

**You're ready! Run `streamlit run app.py` and start testing.** 🚀
