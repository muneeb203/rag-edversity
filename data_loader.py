import os
from pathlib import Path

def load_knowledge_base():
    """Load and parse all knowledge base files into chunks."""
    kb_path = Path("knowledge-base")
    chunks = []
    chunk_id = 0

    files = {
        "faqs.md": "FAQ",
        "policies.md": "POLICY",
        "tickets.md": "TICKET"
    }

    for filename, doc_type in files.items():
        filepath = kb_path / filename
        if not filepath.exists():
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by "---" or "# " to get individual entries
        entries = content.split('\n---\n')

        for entry in entries:
            entry = entry.strip()
            if len(entry) < 50:  # Skip very short entries
                continue

            chunks.append({
                "id": chunk_id,
                "content": entry,
                "source": filename,
                "doc_type": doc_type,
                "metadata": {
                    "source_file": filename,
                    "doc_type": doc_type,
                    "char_count": len(entry)
                }
            })
            chunk_id += 1

    print(f"[OK] Loaded {len(chunks)} chunks from knowledge base")
    return chunks


if __name__ == "__main__":
    chunks = load_knowledge_base()
    for chunk in chunks[:2]:
        print(f"\nChunk {chunk['id']} ({chunk['doc_type']}):")
        print(chunk['content'][:200] + "...")
