import os
import faiss
import joblib
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self):
        self.index = None
        self.metadata = None
        self.embedder = None
        self.is_loaded = False

    def load_db(self):
        faiss_path = 'models/saved/reviews.faiss'
        meta_path = 'models/saved/metadata.pkl'
        
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(faiss_path)
            self.metadata = joblib.load(meta_path)
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.is_loaded = True
            return True
        return False

    def retrieve_context(self, query, top_k=3):
        if not self.is_loaded:
            return "Knowledge base not loaded."
            
        query_vector = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append(f"[Source: {meta['source']} | Sentiment: {meta['label']}] {meta['text']}")
                
        return "\n".join(results)