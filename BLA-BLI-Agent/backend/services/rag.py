import faiss
import numpy as np
from typing import List, Dict, Tuple
from services.embeddings import get_embedding_service
from utils.loader import load_products


class RAGService:
    """RAG (Retrieval Augmented Generation) service for product search"""
    
    def __init__(self):
        self.products: List[Dict] = []
        self.index = None
        self.embedding_service = get_embedding_service()
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
    def initialize(self, products_path: str = "data/sample_products.json"):
        """
        Initialize the RAG service with products
        
        Args:
            products_path: Path to products JSON file
        """
        print("[*] Initializing RAG service...")
        
        # Load products
        self.products = load_products(products_path)
        
        if not self.products:
            print("[!] No products loaded. RAG service will return empty results.")
            return
        
        # Create embeddings for all products
        print("[*] Generating embeddings for products...")
        product_texts = self._create_product_texts()
        embeddings = self.embedding_service.encode(product_texts)
        
        # Create FAISS index
        print("[*] Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"[+] RAG service initialized with {len(self.products)} products")
    
    def _create_product_texts(self) -> List[str]:
        """
        Create searchable text representations of products
        
        Returns:
            List of text strings for each product
        """
        texts = []
        for product in self.products:
            # Combine title, description, tags, and category
            parts = [
                product.get("title", ""),
                product.get("description", ""),
                " ".join(product.get("tags", [])),
                product.get("category", "")
            ]
            text = " ".join(filter(None, parts))
            texts.append(text)
        return texts
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for products using vector similarity
        
        Args:
            query: User search query
            top_k: Number of top results to return
            
        Returns:
            List of top matching products
        """
        if not self.products or self.index is None:
            print("[!] RAG service not initialized")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_service.encode_single(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.products)))
        
        # Get matching products
        results = []
        for idx in indices[0]:
            if idx < len(self.products):
                results.append(self.products[idx])
        
        return results
    
    def get_product_by_id(self, product_id: int) -> Dict:
        """Get a specific product by index"""
        if 0 <= product_id < len(self.products):
            return self.products[product_id]
        return {}


# Global instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get or create the global RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
