import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
from vector_db import VectorDB
from llm_handler import LLMHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CONSTANTS (not user-configurable)
CONFIDENCE_THRESHOLD = 0.7
TOP_K = 5

# Page config
st.set_page_config(page_title="LearnForge AI Support", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db" not in st.session_state:
    st.session_state.db = VectorDB()
    st.session_state.db.initialize()

if "llm" not in st.session_state:
    try:
        st.session_state.llm = LLMHandler()
    except ValueError as e:
        st.error(f"Error: {e}\n\nPlease set ANTHROPIC_API_KEY in your .env file")
        st.stop()

# UI
st.title("🎓 LearnForge AI Support Assistant")
st.markdown("Ask questions about courses, refunds, policies, and more.")

# Sidebar with escalation review
with st.sidebar:
    st.header("⚙️ System Info")
    st.info(f"""
    **Confidence Threshold:** {CONFIDENCE_THRESHOLD}
    **Retrieved Documents:** {TOP_K}

    Queries with confidence < {CONFIDENCE_THRESHOLD} are escalated.
    """)

    st.divider()

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat cleared!")

    st.divider()

    # Escalation Review Section
    st.header("🚨 Escalation Review")
    escalation_file = Path("escalations.json")

    if escalation_file.exists():
        with open(escalation_file) as f:
            escalations = json.load(f)

        if escalations:
            st.info(f"**Total Escalations:** {len(escalations)}")

            # Display each escalation
            for i, esc in enumerate(escalations, 1):
                with st.expander(f"#{i} - Confidence: {esc['confidence']:.2f}"):
                    # Timestamp
                    st.markdown(f"**Time:** {esc['timestamp']}")

                    # User query
                    st.markdown("**User Query:**")
                    st.text(esc['user_query'])

                    # AI response
                    st.markdown("**Assistant Response:**")
                    st.text(esc['assistant_response'])

                    # Confidence
                    st.metric("Confidence Score", f"{esc['confidence']:.2f} / 1.0")

                    # Retrieved sources
                    st.markdown("**Retrieved Sources:**")
                    for src in esc['retrieved_sources']:
                        st.text(f"- {src['doc_type']} from {src['source_file']}")
        else:
            st.success("No escalations yet! ✓")
    else:
        st.info("No escalations yet")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "confidence" in message:
            col1, col2 = st.columns([3, 1])
            with col2:
                if message["confidence"] < CONFIDENCE_THRESHOLD:
                    st.warning(f"⚠️ {message['confidence']:.2f}")
                else:
                    st.success(f"✓ {message['confidence']:.2f}")

# User input
if user_input := st.chat_input("Ask me anything about LearnForge..."):

    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Process with AI
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            # Retrieve relevant chunks
            retrieved = st.session_state.db.search(user_input, top_k=TOP_K)

            # Generate answer
            result = st.session_state.llm.generate_answer(
                user_input,
                retrieved,
                st.session_state.chat_history[:-1]
            )

            answer = result["answer"]
            confidence = result["confidence"]

            # Display answer
            st.write(answer)

            # Show confidence
            col1, col2 = st.columns([3, 1])
            with col2:
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(f"⚠️ Low Confidence: {confidence:.2f}")
                else:
                    st.success(f"✓ Confidence: {confidence:.2f}")

            # Show retrieved sources (expander)
            with st.expander("📚 View Retrieved Sources"):
                for i, chunk in enumerate(retrieved, 1):
                    st.markdown(f"**[{i}] {chunk['metadata']['doc_type']} from {chunk['metadata']['source_file']}**")
                    st.text(chunk['content'][:400] + "...")

            # Escalation logging
            if confidence < CONFIDENCE_THRESHOLD:
                escalation_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "user_query": user_input,
                    "assistant_response": answer,
                    "confidence": confidence,
                    "retrieved_sources": [
                        {
                            "doc_type": chunk["metadata"]["doc_type"],
                            "source_file": chunk["metadata"]["source_file"]
                        }
                        for chunk in retrieved
                    ]
                }

                # Load existing escalations
                escalations = []
                escalation_file = Path("escalations.json")
                if escalation_file.exists():
                    with open(escalation_file) as f:
                        escalations = json.load(f)

                # Append new escalation
                escalations.append(escalation_entry)
                with open(escalation_file, 'w') as f:
                    json.dump(escalations, f, indent=2)

            # Add to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "confidence": confidence
            })
