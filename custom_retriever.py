from langchain.schema import BaseRetriever
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import numpy as np


class CustomRetriever(BaseRetriever):
    """
    Custom retriever that combines multiple retrieval strategies
    """
    
    def __init__(self, vectorstore, k=4, search_type="similarity", 
                 similarity_threshold=0.7, use_hybrid_search=False):
        """
        Initialize the custom retriever
        
        Args:
            vectorstore: The vector store to search in
            k: Number of documents to retrieve
            search_type: Type of search ("similarity", "mmr", "hybrid")
            similarity_threshold: Minimum similarity score threshold
            use_hybrid_search: Whether to use hybrid search combining multiple strategies
        """
        self.vectorstore = vectorstore
        self.k = k
        self.search_type = search_type
        self.similarity_threshold = similarity_threshold
        self.use_hybrid_search = use_hybrid_search
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents based on the query
        
        Args:
            query: The search query
            
        Returns:
            List of relevant documents
        """
        if self.use_hybrid_search:
            return self._hybrid_search(query)
        
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, k=self.k)
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(query, k=self.k)
        else:
            docs = self.vectorstore.similarity_search(query, k=self.k)
            
        return self._filter_by_threshold(docs, query)
    
    def _hybrid_search(self, query: str) -> List[Document]:
        """
        Perform hybrid search combining multiple strategies
        """
        # Get documents from different search methods
        similarity_docs = self.vectorstore.similarity_search(query, k=self.k//2)
        mmr_docs = self.vectorstore.max_marginal_relevance_search(query, k=self.k//2)
        
        # Combine and deduplicate
        all_docs = docs = similarity_docs + mmr_docs
        unique_docs = self._deduplicate_documents(all_docs)
        
        # Sort by relevance and return top k
        return unique_docs[:self.k]
    
    def _filter_by_threshold(self, docs: List[Document], query: str) -> List[Document]:
        """
        Filter documents by similarity threshold
        """
        if not docs:
            return []
            
        # Get documents with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k*2)
        
        # Filter by threshold
        filtered_docs = [
            doc for doc, score in docs_with_scores 
            if score >= self.similarity_threshold
        ]
        
        return filtered_docs[:self.k] if filtered_docs else docs
    
    def _deduplicate_documents(self, docs: List[Document]) -> List[Document]:
        """
        Remove duplicate documents based on content
        """
        seen = set()
        unique_docs = []
        
        for doc in docs:
            # Create a hash of the document content
            content_hash = hash(doc.page_content[:100])  # First 100 chars
            
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
                
        return unique_docs
    
    def get_relevant_documents_with_score(self, query: str) -> List[tuple]:
        """
        Retrieve documents with similarity scores
        
        Args:
            query: The search query
            
        Returns:
            List of tuples (document, score)
        """
        return self.vectorstore.similarity_search_with_score(query, k=self.k)
    
    def invoke(self, query: str) -> List[Document]:
        """
        Main method to invoke the retriever
        
        Args:
            query: The search query
            
        Returns:
            List of relevant documents
        """
        return self.get_relevant_documents(query)
    
    def set_search_parameters(self, k: Optional[int] = None, 
                            search_type: Optional[str] = None,
                            similarity_threshold: Optional[float] = None):
        """
        Update search parameters dynamically
        
        Args:
            k: Number of documents to retrieve
            search_type: Type of search
            similarity_threshold: Similarity threshold
        """
        if k is not None:
            self.k = k
        if search_type is not None:
            self.search_type = search_type
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get current search configuration
        
        Returns:
            Dictionary with current search parameters
        """
        return {
            "k": self.k,
            "search_type": self.search_type,
            "similarity_threshold": self.similarity_threshold,
            "use_hybrid_search": self.use_hybrid_search
        }