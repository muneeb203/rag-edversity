import sqlite3
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_knowledge_base

class VectorDB:
    def __init__(self, db_path="vector_db.sqlite"):
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.is_initialized = False
        self.embeddings_matrix = None
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_file TEXT NOT NULL,
                doc_type TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def initialize(self):
        """Load KB into vector DB (only once)."""
        if self.is_initialized:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM chunks')
        count = cursor.fetchone()[0]
        conn.close()

        if count > 0:
            print(f"[OK] Vector DB already initialized with {count} documents")
            self._load_embeddings()
            self.is_initialized = True
            return

        # Load and add chunks
        chunks = load_knowledge_base()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        chunk_texts = []
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
            chunk_texts.append(chunk['content'])

        conn.commit()
        conn.close()

        # Create TF-IDF embeddings
        print(f"[OK] Added {len(chunks)} chunks to vector DB")
        self._load_embeddings()
        self.is_initialized = True

    def _load_embeddings(self):
        """Load all chunks and create embeddings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT content FROM chunks ORDER BY id')
        rows = cursor.fetchall()
        conn.close()

        if rows:
            texts = [row[0] for row in rows]
            self.embeddings_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=5):
        """Search for relevant chunks."""
        if not self.is_initialized:
            self.initialize()

        if self.embeddings_matrix is None or self.embeddings_matrix.shape[0] == 0:
            return []

        # Embed query
        query_embedding = self.vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Fetch chunks from DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, content, source_file, doc_type FROM chunks ORDER BY id')
        all_rows = cursor.fetchall()
        conn.close()

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


if __name__ == "__main__":
    db = VectorDB()
    db.initialize()

    # Test search
    results = db.search("How do I access a course?", top_k=3)
    print("\n--- Test Search Results ---")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['metadata']['doc_type']} (similarity: {result['similarity']:.3f})")
        print(result['content'][:300] + "...")
