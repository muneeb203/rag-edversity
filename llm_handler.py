import os
import re
from anthropic import Anthropic

class LLMHandler:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or arguments")

        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-haiku-4-5-20251001"

        self.system_prompt = """You are LearnForge AI Support Assistant. Your job is to help users with questions about courses, refunds, account access, and policies.

IMPORTANT RULES:
1. ONLY use the provided context documents to answer questions
2. If the context does not contain information to answer the question, respond with: "I'm not certain about this. Let me connect you with a human agent who can help."
3. After your answer, always include a confidence score on a scale of 0-1 (0 = not confident, 1 = very confident)
4. Format: End your response with exactly: <confidence>0.XX</confidence>
5. Be concise and helpful. If multiple sources conflict, mention that and recommend human review.
"""

    def generate_answer(self, user_query, retrieved_context, conversation_history):
        """Generate answer from LLM with confidence score."""

        # Format context
        context_text = ""
        if retrieved_context:
            context_text = "RELEVANT KNOWLEDGE BASE:\n"
            for i, chunk in enumerate(retrieved_context, 1):
                context_text += f"\n[Source {i}: {chunk['metadata']['doc_type']}]\n{chunk['content']}\n"
        else:
            context_text = "\n[No relevant documents found in knowledge base]"

        # Build messages with history
        messages = []

        # Add conversation history (last 10 messages to avoid token limit)
        for msg in conversation_history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Add current query with context
        messages.append({
            "role": "user",
            "content": f"{context_text}\n\nUser Question: {user_query}"
        })

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=self.system_prompt,
                messages=messages
            )

            full_response = response.content[0].text

            # Extract confidence score
            confidence = self._extract_confidence(full_response)

            # Clean response (remove confidence tag for display)
            clean_response = re.sub(r'<confidence>[\d.]+</confidence>', '', full_response).strip()

            return {
                "answer": clean_response,
                "confidence": confidence,
                "raw_response": full_response
            }

        except Exception as e:
            print(f"LLM Error: {e}")
            return {
                "answer": "I encountered an error. Let me connect you with a human agent.",
                "confidence": 0.0,
                "raw_response": str(e)
            }

    def _extract_confidence(self, response_text):
        """Extract confidence score from response."""
        match = re.search(r'<confidence>([\d.]+)</confidence>', response_text)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except:
                return 0.5
        return 0.5  # Default if not found


if __name__ == "__main__":
    handler = LLMHandler()

    # Test
    test_context = [{
        "content": "Refunds are allowed within 14 days of purchase.",
        "metadata": {"doc_type": "POLICY"}
    }]

    test_history = []

    result = handler.generate_answer(
        "Can I get a refund?",
        test_context,
        test_history
    )

    print("Answer:", result['answer'])
    print(f"Confidence: {result['confidence']}")
