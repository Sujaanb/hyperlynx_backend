"""
RAG Service Module for Hyperlynx

This module provides RAG (Retrieval-Augmented Generation) capabilities for the /generate endpoint.
It handles:
1. Non-persistent (in-memory) vectorization of uploaded documents
2. Retrieval of relevant documents from Chroma Cloud
3. Combining context for comprehensive LLM responses
"""

import os
import logging
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG Service for compliance document analysis.
    
    This service connects to Chroma Cloud to retrieve relevant compliance documents
    and uses non-persistent (in-memory) vectorization for uploaded user documents.
    """
    
    # Collection name matching what's used in upload_documents.py
    COLLECTION_NAME = "compliance_collection"
    
    def __init__(self):
        """
        Initialize the RAG service with embeddings and Chroma client.
        """
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
        self.chroma_api_key = os.getenv("CHROMA_API_KEY")
        self.chroma_tenant = os.getenv("CHROMA_TENANT")
        self.chroma_database = os.getenv("CHROMA_DATABASE")
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Initialize Chroma client for cloud access
        self._init_chroma_client()
        
        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        logger.info("RAG Service initialized successfully")
    
    def _init_embeddings(self):
        """
        Initialize embeddings based on the selected provider.
        """
        if self.embedding_provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            )
            logger.info(f"Using OpenAI embeddings: {os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')}")
            
        elif self.embedding_provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            
            model_name = os.getenv(
                "HUGGINGFACE_EMBEDDING_MODEL", 
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            device = os.getenv("EMBEDDING_DEVICE", "cpu")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Using HuggingFace embeddings: {model_name} on {device}")
        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")
    
    def _init_chroma_client(self):
        """
        Initialize Chroma client for cloud access.
        """
        if self.chroma_api_key:
            logger.info("Connecting to Chroma Cloud...")
            self.client = chromadb.CloudClient(
                api_key=self.chroma_api_key,
                tenant=self.chroma_tenant,
                database=self.chroma_database,
            )
            logger.info(f"Connected to Chroma Cloud (tenant: {self.chroma_tenant or 'auto'})")
        else:
            # Fallback to local persistent Chroma
            logger.info("Using local persistent Chroma storage...")
            self.client = chromadb.PersistentClient(
                path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
            )
    
    def get_vectorstore(self) -> Chroma:
        """
        Get the Chroma vectorstore for the compliance collection.
        """
        return Chroma(
            client=self.client,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings
        )
    
    def retrieve_relevant_documents(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Document]:
        """
        Retrieve relevant documents from Chroma Cloud based on the query.
        
        Args:
            query: The search query or question
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        try:
            vectorstore = self.get_vectorstore()
            docs = vectorstore.similarity_search(query, k=top_k)
            logger.info(f"Retrieved {len(docs)} relevant documents from Chroma Cloud")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_documents_by_similarity_to_content(
        self, 
        document_content: str, 
        top_k: int = 5
    ) -> List[Document]:
        """
        Retrieve documents from Chroma Cloud that are similar to the provided content.
        This is used to find compliance documents relevant to an uploaded document.
        
        Args:
            document_content: The content of the uploaded document
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        try:
            # Chunk the document and use the first few chunks for similarity search
            chunks = self.text_splitter.split_text(document_content)
            
            # Use the first chunk (or first few) to find similar documents
            search_text = " ".join(chunks[:3]) if len(chunks) > 3 else document_content
            
            vectorstore = self.get_vectorstore()
            docs = vectorstore.similarity_search(search_text[:2000], k=top_k)  # Limit search text
            logger.info(f"Retrieved {len(docs)} similar documents from Chroma Cloud")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving similar documents: {str(e)}")
            return []
    
    def build_rag_context(
        self, 
        question: str, 
        uploaded_document_content: Optional[str] = None,
        top_k: int = 5
    ) -> Tuple[str, List[str]]:
        """
        Build the context for RAG-based generation.
        
        Args:
            question: The user's question
            uploaded_document_content: Optional content from an uploaded document
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (context_string, list_of_source_names)
        """
        retrieved_docs = []
        sources = []
        
        # If document content is provided, find similar compliance documents
        if uploaded_document_content:
            # Get documents similar to the uploaded content
            similar_docs = self.retrieve_documents_by_similarity_to_content(
                uploaded_document_content, 
                top_k=top_k
            )
            retrieved_docs.extend(similar_docs)
            
            # Also get documents relevant to the question
            question_docs = self.retrieve_relevant_documents(question, top_k=top_k//2)
            retrieved_docs.extend(question_docs)
        else:
            # No uploaded document, just query based on the question
            retrieved_docs = self.retrieve_relevant_documents(question, top_k=top_k)
        
        # Remove duplicates based on content
        seen_content = set()
        unique_docs = []
        for doc in retrieved_docs:
            content_hash = hash(doc.page_content[:200])  # Use first 200 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Build context string
        context_parts = []
        
        if uploaded_document_content:
            context_parts.append("=== USER UPLOADED DOCUMENT ===")
            # Truncate if too long
            truncated_content = uploaded_document_content[:5000]
            if len(uploaded_document_content) > 5000:
                truncated_content += "\n... [Content truncated for length]"
            context_parts.append(truncated_content)
            context_parts.append("")
        
        if unique_docs:
            context_parts.append("=== RELEVANT COMPLIANCE DOCUMENTS FROM DATABASE ===")
            for i, doc in enumerate(unique_docs, 1):
                source_name = doc.metadata.get("file_name", doc.metadata.get("source", f"Document {i}"))
                sources.append(source_name)
                context_parts.append(f"\n--- Reference {i}: {source_name} ---")
                context_parts.append(doc.page_content)
        
        context = "\n".join(context_parts)
        return context, sources


# Singleton instance for reuse
_rag_service_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Get or create the RAG service singleton instance.
    """
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
