from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from typing import Dict, List

class NAICSVectorIndex:
    def __init__(self, naics_df: pd.DataFrame):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store text data for hybrid search
        self.descriptions = naics_df['2022 NAICS Title'].fillna('').tolist()
        self.naics_data = naics_df[['2022 NAICS Code', '2022 NAICS Title']].copy()
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.descriptions)
        
        # Generate embeddings
        self.embeddings = self.model.encode(self.descriptions)
        self.index.add(self.embeddings)

    def hybrid_search(self, query: str, top_k: int = 10, vector_weight: float = 0.7) -> List[Dict]:
        """Improved hybrid search with proper normalization"""
        # Vector search
        query_embedding = self.model.encode(query)
        vector_scores_raw, indices = self.index.search(query_embedding.reshape(1, -1), top_k*2)
        
        # TF-IDF search
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores_raw = np.squeeze(query_tfidf.dot(self.tfidf_matrix.T).toarray())
        
        # Get valid indices
        combined_indices = np.unique(np.concatenate([indices[0], np.argsort(tfidf_scores_raw)[-top_k*2:]]))
        valid_indices = [idx for idx in combined_indices if 0 <= idx < len(self.descriptions)]
        
        # Score normalization
        vector_scores = 1 / (1 + vector_scores_raw[0])  # Convert distance to similarity
        tfidf_scores = tfidf_scores_raw / (np.linalg.norm(tfidf_scores_raw) or 1)
        
        # Combine scores
        matches = []
        for idx in valid_indices:
            vector_score = vector_scores[list(indices[0]).index(idx)] if idx in indices[0] else 0
            tfidf_score = tfidf_scores[idx]
            
            combined_score = (vector_weight * vector_score + 
                            (1 - vector_weight) * tfidf_score)
            
            match = self.naics_data.iloc[idx].to_dict()
            match.update({
                'similarity_score': float(combined_score),
                'vector_score': float(vector_score),
                'keyword_score': float(tfidf_score)
            })
            matches.append(match)
        
        # Sort by combined score
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        print(matches)
        return matches[:top_k]

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Pure keyword search fallback"""
        query_tfidf = self.tfidf_vectorizer.transform([query])
        scores = np.squeeze(query_tfidf.dot(self.tfidf_matrix.T).toarray())
        indices = np.argsort(scores)[-top_k:][::-1]
        
        return [{
            **self.naics_data.iloc[idx].to_dict(),
            'keyword_score': scores[idx]
        } for idx in indices if scores[idx] > 0]

def create_naics_search_tool(vector_index: NAICSVectorIndex):
    """Create hybrid search tool for pydantic_ai"""
    
    async def hybrid_naics_search(query: str) -> Dict[str, List[Dict]]:
        """Hybrid search combining vector and keyword matching"""
        try:
            return {
                "matches": vector_index.hybrid_search(query),
                "query": query,
                "search_type": "hybrid"
            }
        except Exception as e:
            raise RuntimeError(f"Hybrid search failed: {str(e)}")
    
    return hybrid_naics_search
