import os
import faiss
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from models.data_loader import DataLoader

def build_vector_db():
    print("Loading datasets...")
    df = DataLoader.get_unified_dataset()
    
    # Use a lightweight, highly effective embedding model
    print("Loading SentenceTransformer model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = df['Text'].tolist()
    labels = df['Label'].tolist()
    sources = df['Source'].tolist()
    
    print(f"Encoding {len(texts)} records (This may take a moment)...")
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save Metadata for RAG context
    metadata = [{'text': t, 'label': l, 'source': s} for t, l, s in zip(texts, labels, sources)]
    
    os.makedirs('models/saved', exist_ok=True)
    faiss.write_index(index, 'models/saved/reviews.faiss')
    joblib.dump(metadata, 'models/saved/metadata.pkl')
    
    print("Vector Database successfully built and saved to 'models/saved/'.")

if __name__ == "__main__":
    build_vector_db()

    