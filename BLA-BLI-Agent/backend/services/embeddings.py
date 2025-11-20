from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        print(f"[*] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"[+] Embedding model loaded")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to encode
            
        Returns:
            numpy array embedding
        """
        return self.model.encode([text], show_progress_bar=False)[0]


# Global instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
