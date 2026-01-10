import os
import json
from typing import List
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from chroma_files_mapping import get_chroma_files_mapping, JSON_FILE_PATH
from util.utility import Utility

# Load environment variables
load_dotenv()


class ChromaUploader:
    def __init__(self):
        """
        Initialize the Chroma uploader with OpenAI embeddings and Chroma client.
        """
        # Get configuration from environment
        self.chroma_api_key = os.getenv("CHROMA_API_KEY")
        self.chroma_tenant = os.getenv("CHROMA_TENANT")
        self.chroma_database = os.getenv("CHROMA_DATABASE")
        self.chroma_server_host = os.getenv("CHROMA_SERVER_HOST")
        self.chroma_server_http_port = os.getenv("CHROMA_SERVER_HTTP_PORT")
        
        # Initialize embeddings based on provider
        self._init_embeddings()
        
        # Initialize Chroma client
        self._init_chroma_client()
    
    def _init_embeddings(self):
        """
        Initialize OpenAI embeddings.
        """
        print("Using OpenAI embeddings...")
        from langchain_openai import OpenAIEmbeddings
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
        print(f"✓ Loaded OpenAI embeddings: {os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')}")
    
    def _init_chroma_client(self):
        """
        Initialize Chroma client based on available configuration.
        Supports Chroma Cloud, self-hosted Chroma server, or local persistent storage.
        """
        if self.chroma_api_key:
            # Using Chroma Cloud with CloudClient
            print("Connecting to Chroma Cloud...")
            self.client = chromadb.CloudClient(
                api_key=self.chroma_api_key,
                tenant=self.chroma_tenant,  # Optional: inferred from API key if not provided
                database=self.chroma_database,  # Optional: inferred from API key if not provided
            )
            print(f"Connected to Chroma Cloud (tenant: {self.chroma_tenant or 'auto'}, database: {self.chroma_database or 'auto'})")
        elif self.chroma_server_host:
            # Using self-hosted Chroma server
            print(f"Connecting to Chroma server at {self.chroma_server_host}...")
            self.client = chromadb.HttpClient(
                host=self.chroma_server_host,
                port=int(self.chroma_server_http_port) if self.chroma_server_http_port else 8000,
                ssl=False,
                tenant=self.chroma_tenant or "default_tenant",
                database=self.chroma_database or "default_database"
            )
        else:
            # Using local persistent Chroma
            print("Using local persistent Chroma storage...")
            self.client = chromadb.PersistentClient(
                path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
            )
    
    def upload_document(self, file_mapping: dict) -> bool:
        """
        Uploads a single document to Chroma.
        
        Args:
            file_mapping: Dictionary containing file information
        
        Returns:
            bool: True if successful, False otherwise
        """
        file_path = file_mapping["file_path"]
        file_name = file_mapping["file_name"]
        collection_name = file_mapping["chroma_collection_name"]
        doc_id = file_mapping["doc_id"]
        
        try:
            print(f"\nProcessing: {file_name} ({doc_id})")
            
            # Read the document using the Utility class
            print(f"  Reading document from: {file_path}")
            raw_documents = Utility.read_file_content(file_path)
            
            if not raw_documents:
                print(f"  Warning: No content extracted from {file_name}")
                return False
            
            # Split documents into smaller chunks to avoid request size limits (16KB quota)
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            documents = text_splitter.split_documents(raw_documents)
            
            print(f"  Extracted {len(raw_documents)} raw page(s), split into {len(documents)} chunk(s)")
            
            # Add metadata to each document
            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["file_name"] = file_name
                doc.metadata["doc_id"] = doc_id
                doc.metadata["source_file"] = file_path
            
            # Create or get the collection
            print(f"  Uploading to collection: {collection_name}")
            
            # Use LangChain's Chroma wrapper for easier integration
            vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            # Generate unique IDs for each chunk
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(documents))]
            
            # Add documents to the collection
            # Process in batches to be safe
            batch_size = 10
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                print(f"    Uploaded batch {i//batch_size + 1}/{total_batches}", end='\r')
            
            print(f"\n  ✓ Successfully uploaded {file_name} to {collection_name}")
            return True
            
        except Exception as e:
            print(f"  ✗ Error uploading {file_name}: {str(e)}")
            return False
    
    def update_json_status(self, doc_id: str, uploaded_status: int):
        """
        Updates the uploaded status for a specific document in the JSON file.
        
        Args:
            doc_id: The document ID to update
            uploaded_status: The new status (0 or 1)
        """
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        for mapping in mappings:
            if mapping["doc_id"] == doc_id:
                mapping["uploaded"] = uploaded_status
                break
        
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=4, ensure_ascii=False)
    
    def process_all_documents(self):
        """
        Main method to process all documents that haven't been uploaded yet.
        """
        print("=" * 60)
        print("Starting document upload process...")
        print("=" * 60)
        
        # Get the current mapping
        mappings = get_chroma_files_mapping()
        
        # Filter documents that need to be uploaded
        pending_uploads = [m for m in mappings if m["uploaded"] == 0]
        
        if not pending_uploads:
            print("\n✓ All documents are already uploaded!")
            return
        
        print(f"\nFound {len(pending_uploads)} document(s) pending upload.")
        
        success_count = 0
        failure_count = 0
        
        for mapping in pending_uploads:
            success = self.upload_document(mapping)
            
            if success:
                # Update the JSON to mark as uploaded
                self.update_json_status(mapping["doc_id"], 1)
                success_count += 1
            else:
                failure_count += 1
        
        print("\n" + "=" * 60)
        print("Upload Summary:")
        print(f"  ✓ Successful: {success_count}")
        if failure_count > 0:
            print(f"  ✗ Failed: {failure_count}")
        print("=" * 60)


def main():
    """
    Main function to run the upload process.
    """
    try:
        uploader = ChromaUploader()
        uploader.process_all_documents()
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
