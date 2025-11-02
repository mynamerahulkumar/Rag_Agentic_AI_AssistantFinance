from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Optional
import os
import tempfile
import hashlib
import json
from pathlib import Path

class RAGModule:
    """
    RAG Module for document processing, embedding generation, and vector store management.
    """
    
    def __init__(self, openai_api_key: str, persist_directory: str = "./vectorstore_db"):
        """
        Initialize RAG Module with OpenAI API key.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            persist_directory: Directory to persist FAISS vectorstore (default: ./vectorstore_db)
        """
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = None
        self.persist_directory = persist_directory
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def load_documents(self, uploaded_files: List) -> List:
        """
        Load documents from uploaded files.
        
        Args:
            uploaded_files: List of uploaded file objects from Streamlit
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        print(f"📚 Loading {len(uploaded_files)} file(s)...")
        
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            # Determine file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            print(f"  📄 File {idx}/{len(uploaded_files)}: {uploaded_file.name} ({file_extension.upper()})")
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                print(f"    💾 Saved to temp file: {tmp_path}")
            
            try:
                # Load document based on file type
                if file_extension == 'pdf':
                    print(f"    📖 Loading PDF...")
                    loader = PyPDFLoader(tmp_path)
                elif file_extension == 'txt':
                    print(f"    📖 Loading text file...")
                    loader = TextLoader(tmp_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")
                
                # Load and add documents
                print(f"    🔄 Calling loader.load()...")
                loaded_docs = loader.load()
                print(f"    ✅ Loaded {len(loaded_docs)} page(s)/section(s) from {uploaded_file.name}")
                
                # Check if documents have content
                if len(loaded_docs) > 0:
                    print(f"    📝 Checking document content...")
                    total_chars = sum(len(doc.page_content) if doc.page_content else 0 for doc in loaded_docs)
                    print(f"    📊 Total characters extracted: {total_chars}")
                    
                    # Show preview of first document
                    if loaded_docs[0].page_content:
                        preview = loaded_docs[0].page_content[:200]
                        print(f"    📄 First page preview: {preview}...")
                    else:
                        print(f"    ⚠️ WARNING: First document has no content!")
                    
                    # Check if any documents are empty
                    empty_docs = sum(1 for doc in loaded_docs if not doc.page_content or len(doc.page_content.strip()) == 0)
                    if empty_docs > 0:
                        print(f"    ⚠️ WARNING: {empty_docs} out of {len(loaded_docs)} documents are empty!")
                    
                    if total_chars == 0:
                        print(f"    ❌ ERROR: No text extracted from PDF! PDF might be:")
                        print(f"       - Image-based (scanned PDF)")
                        print(f"       - Protected/encrypted")
                        print(f"       - Corrupted")
                        print(f"    💡 Try using OCR or extracting text differently")
                else:
                    print(f"    ❌ ERROR: No documents loaded from {uploaded_file.name}!")
                    print(f"    PDF might be empty or unreadable")
                
                documents.extend(loaded_docs)
                
            except Exception as e:
                print(f"    ❌ Error loading {uploaded_file.name}: {str(e)}")
                raise
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    print(f"    🗑️ Cleaned up temp file")
        
        print(f"✅ Total documents loaded: {len(documents)}")
        
        # Final check: ensure we have documents with content
        if len(documents) == 0:
            print("❌ CRITICAL: No documents loaded from any files!")
            return documents
        
        total_content = sum(len(doc.page_content) if doc.page_content else 0 for doc in documents)
        print(f"📊 Total characters across all documents: {total_content}")
        
        if total_content == 0:
            print("❌ CRITICAL: All documents are empty!")
            print("   This means the PDF files are:")
            print("   - Image-based (scanned PDFs need OCR)")
            print("   - Protected or encrypted")
            print("   - Corrupted or unreadable")
            print("   - Empty files")
        else:
            print(f"✅ Documents contain {total_content} characters of text")
            # Show sample from a document that has content
            for doc in documents:
                if doc.page_content and len(doc.page_content.strip()) > 0:
                    sample = doc.page_content[:300]
                    print(f"📄 Sample content: {sample}...")
                    break
        
        return documents
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        print(f"✂️ Splitting {len(documents)} documents into chunks...")
        print(f"   Chunk size: {self.text_splitter._chunk_size}, Overlap: {self.text_splitter._chunk_overlap}")
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        if len(chunks) > 0:
            # Check chunk content
            total_chunk_chars = sum(len(chunk.page_content) if chunk.page_content else 0 for chunk in chunks)
            empty_chunks = sum(1 for chunk in chunks if not chunk.page_content or len(chunk.page_content.strip()) == 0)
            
            print(f"📊 Total characters in chunks: {total_chunk_chars}")
            if empty_chunks > 0:
                print(f"⚠️ WARNING: {empty_chunks} chunks are empty!")
            
            # Show sample chunks
            non_empty_chunks = [chunk for chunk in chunks if chunk.page_content and len(chunk.page_content.strip()) > 0]
            if len(non_empty_chunks) > 0:
                print(f"✅ {len(non_empty_chunks)} chunks have content")
                sample = non_empty_chunks[0].page_content[:200]
                print(f"📄 Sample chunk: {sample}...")
            else:
                print("❌ ERROR: All chunks are empty!")
        
        return chunks
    
    def create_embeddings(self):
        """
        Initialize embeddings model.
        Note: Embeddings are created when creating vector store.
        """
        return self.embeddings
    
    def get_vectorstore_path(self, file_names: List[str]) -> str:
        """
        Generate a path for storing vectorstore based on file names.
        
        Args:
            file_names: List of uploaded file names
            
        Returns:
            Path to vectorstore directory
        """
        # Create a hash from file names to create unique directory
        file_names_str = ",".join(sorted(file_names))
        file_hash = hashlib.md5(file_names_str.encode()).hexdigest()[:8]
        vectorstore_path = os.path.join(self.persist_directory, f"vectorstore_{file_hash}")
        return vectorstore_path
    
    def create_vectorstore(self, chunks: List, file_names: Optional[List[str]] = None, save_to_disk: bool = True):
        """
        Create FAISS vector store from document chunks and optionally save to disk.
        
        Args:
            chunks: List of document chunks
            file_names: List of file names (optional, for persistence path)
            save_to_disk: Whether to save vectorstore to disk (default: True)
            
        Returns:
            FAISS vector store
        """
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")
        
        print(f"🔧 Creating FAISS vector store with {len(chunks)} chunks...")
        print("   Generating embeddings (this may take a moment)...")
        print("   This involves calling OpenAI API for each chunk...")
        
        import time
        start_time = time.time()
        
        try:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            elapsed_time = time.time() - start_time
            print(f"✅ Vector store created successfully!")
            print(f"⏱️ Embedding generation took {elapsed_time:.2f} seconds")
            print(f"   Processed {len(chunks)} chunks")
            
            # Verify the vectorstore
            try:
                test_count = len(self.vectorstore.index_to_docstore_id)
                print(f"✅ Verified: Vectorstore contains {test_count} embeddings")
            except Exception as ve:
                print(f"⚠️ Could not verify vectorstore size: {str(ve)}")
            
            # Save to disk if requested
            if save_to_disk and file_names:
                vectorstore_path = self.get_vectorstore_path(file_names)
                print(f"💾 Saving vectorstore to disk at: {vectorstore_path}")
                try:
                    self.vectorstore.save_local(vectorstore_path)
                    print(f"✅ Vectorstore saved successfully!")
                    
                    # Save metadata about files
                    metadata_file = os.path.join(vectorstore_path, "metadata.json")
                    metadata = {
                        "file_names": file_names,
                        "num_chunks": len(chunks),
                        "created_at": str(time.time())
                    }
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    print(f"✅ Metadata saved to {metadata_file}")
                except Exception as save_error:
                    print(f"⚠️ Warning: Could not save vectorstore to disk: {str(save_error)}")
            
            return self.vectorstore
        except Exception as e:
            print(f"❌ ERROR creating vector store: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_vectorstore(self, file_names: Optional[List[str]] = None, persist_directory: Optional[str] = None):
        """
        Load existing vector store from disk.
        
        Args:
            file_names: List of file names (optional, to find matching vectorstore)
            persist_directory: Directory where vector store is persisted (optional, if not using file_names)
            
        Returns:
            FAISS vector store if found, None otherwise
        """
        if persist_directory:
            vectorstore_path = persist_directory
        elif file_names:
            vectorstore_path = self.get_vectorstore_path(file_names)
        else:
            print("⚠️ Warning: No file names or persist_directory provided")
            return None
        
        if os.path.exists(vectorstore_path):
            print(f"📂 Loading vectorstore from: {vectorstore_path}")
            try:
                # Check if required files exist
                index_file = os.path.join(vectorstore_path, "index.faiss")
                pkl_file = os.path.join(vectorstore_path, "index.pkl")
                
                if not os.path.exists(index_file) or not os.path.exists(pkl_file):
                    print(f"⚠️ Warning: Vectorstore files not found at {vectorstore_path}")
                    return None
                
                self.vectorstore = FAISS.load_local(
                    vectorstore_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Vectorstore loaded successfully from disk!")
                
                # Load and display metadata
                metadata_file = os.path.join(vectorstore_path, "metadata.json")
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        print(f"📄 Metadata: {metadata.get('num_chunks', 'N/A')} chunks from {len(metadata.get('file_names', []))} file(s)")
                    except Exception as e:
                        print(f"⚠️ Could not load metadata: {str(e)}")
                
                return self.vectorstore
            except Exception as e:
                print(f"❌ Error loading vectorstore: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print(f"ℹ️  No existing vectorstore found at: {vectorstore_path}")
            return None
    
    def find_or_create_vectorstore(self, file_names: List[str], chunks: List):
        """
        Find existing vectorstore for files or create new one.
        
        Args:
            file_names: List of file names
            chunks: List of document chunks (if creating new vectorstore)
            
        Returns:
            FAISS vector store
        """
        # Try to load existing vectorstore
        existing_vectorstore = self.load_vectorstore(file_names=file_names)
        
        if existing_vectorstore:
            print("✅ Using existing vectorstore from disk")
            self.vectorstore = existing_vectorstore
            return self.vectorstore
        else:
            print("📝 Creating new vectorstore (not found on disk)")
            return self.create_vectorstore(chunks, file_names=file_names, save_to_disk=True)
    
    def retrieve_documents(self, query: str, k: int = 3) -> List:
        """
        Retrieve relevant documents based on query.
        
        Args:
            query: User query string
            k: Number of documents to retrieve (default: 3)
            
        Returns:
            List of relevant document chunks
        """
        print(f"🔎 Starting retrieval with query: '{query}'")
        print(f"   Retrieving top {k} documents")
        
        if self.vectorstore is None:
            print("❌ ERROR: Vector store is None!")
            raise ValueError("Vector store not initialized. Please create or load vector store first.")
        
        print("✅ Vector store exists, checking accessibility...")
        try:
            # Try to get document count if possible
            # Note: FAISS doesn't expose a direct count, but we can check with a test search
            # Use a generic query instead of empty string
            test_search = self.vectorstore.similarity_search("test", k=1)
            print(f"✅ Vector store is accessible (test search returned {len(test_search)} doc)")
            if len(test_search) == 0:
                print("⚠️ WARNING: Vector store is empty - no documents found!")
        except Exception as e:
            print(f"⚠️ Warning: Could not test vector store: {str(e)}")
        
        print("🔄 Performing similarity search with scores...")
        print(f"   This may take a moment if generating embeddings...")
        
        try:
            # First try with similarity_search_with_score to see actual scores
            # Use a larger k to ensure we get results
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=max(k, 5))
            print(f"✅ Similarity search with scores completed!")
            print(f"   Found {len(docs_with_scores)} document(s) with scores")
            
            if len(docs_with_scores) > 0:
                print("📊 Similarity Scores (lower = more similar):")
                for idx, (doc, score) in enumerate(docs_with_scores[:5], 1):
                    content_preview = doc.page_content[:100] if doc.page_content else "[EMPTY]"
                    print(f"   {idx}. Score: {score:.4f}, Length: {len(doc.page_content)} chars")
                    print(f"      Preview: {content_preview}...")
                
                # FAISS uses L2 distance, so lower score = more similar
                # For OpenAI embeddings, typical good matches are < 1.0, acceptable < 1.5
                # But we'll be lenient and accept most results unless score is extremely high
                threshold = 2.0  # Lenient threshold to ensure we get results
                filtered_docs = [(doc, score) for doc, score in docs_with_scores if score <= threshold]
                
                # If we have any docs with reasonable scores, use them
                # Otherwise, use all docs regardless of score (something is better than nothing)
                if len(filtered_docs) >= k:
                    docs = [doc for doc, score in filtered_docs[:k]]
                    print(f"✅ Using {len(docs)} docs with scores <= {threshold}")
                elif len(filtered_docs) > 0:
                    # Use filtered docs even if less than k
                    docs = [doc for doc, score in filtered_docs]
                    print(f"✅ Using {len(docs)} docs with scores <= {threshold} (less than requested {k})")
                else:
                    # If all scores are too high, still use the top k (maybe query is just not well matched)
                    docs = [doc for doc, score in docs_with_scores[:k]]
                    print(f"⚠️ All scores > {threshold}, using top {len(docs)} docs anyway")
                    print(f"   Best score: {docs_with_scores[0][1]:.4f}")
                
                # Ensure we have non-empty documents
                docs = [doc for doc in docs if doc.page_content and len(doc.page_content.strip()) > 0]
                if len(docs) == 0 and len(docs_with_scores) > 0:
                    print("⚠️ WARNING: All retrieved docs have empty content!")
                    # Try to get docs that might have content
                    docs = [doc for doc, score in docs_with_scores if doc.page_content and len(doc.page_content.strip()) > 0][:k]
                
                print(f"✅ Returning {len(docs)} document(s)")
            else:
                print("⚠️ WARNING: No documents retrieved from similarity search!")
                docs = []
            
            # Fallback: If we get no results, try MMR search
            if len(docs) == 0:
                print("🔄 Trying MMR (Maximum Marginal Relevance) search as fallback...")
                try:
                    docs = self.vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=min(k*3, 20))
                    print(f"✅ MMR search returned {len(docs)} document(s)")
                except Exception as mmr_error:
                    print(f"⚠️ MMR search also failed: {str(mmr_error)}")
            
            # Final fallback: If still no results, try retrieving ALL documents (last resort)
            if len(docs) == 0:
                print("🔄 Last resort: Trying to retrieve any documents from vectorstore...")
                try:
                    # Try multiple generic queries
                    fallback_queries = ["the", "a", "and", "or", query]  # Include original query too
                    for fallback_query in fallback_queries:
                        try:
                            all_docs = self.vectorstore.similarity_search(fallback_query, k=max(5, k*2))
                            print(f"   Trying '{fallback_query}': returned {len(all_docs)} docs")
                            if len(all_docs) > 0:
                                docs = all_docs[:k]
                                print(f"✅ Retrieved {len(docs)} document(s) with fallback query: '{fallback_query}'")
                                print("⚠️ WARNING: Using fallback retrieval - query matching may not work properly")
                                break
                        except Exception as fb_error:
                            print(f"   Fallback query '{fallback_query}' failed: {str(fb_error)}")
                            continue
                    
                    # If still nothing, try with ANY query string that should exist
                    if len(docs) == 0:
                        print("   Trying with k=20 to get any results...")
                        any_docs = self.vectorstore.similarity_search(query, k=20)
                        if len(any_docs) > 0:
                            docs = any_docs[:k]
                            print(f"✅ Retrieved {len(docs)} docs with k=20")
                        else:
                            print("❌ Even with k=20, no documents retrieved!")
                except Exception as generic_error:
                    print(f"❌ Even generic retrieval failed: {str(generic_error)}")
            
            # FINAL CHECK: If we still have no docs but vectorstore exists, something is very wrong
            if len(docs) == 0:
                print("🚨 CRITICAL: No documents retrieved despite having vectorstore!")
                print("   This indicates a serious issue with the vectorstore or embeddings")
                print("   Vectorstore might be empty or corrupted")
            else:
                print(f"✅ Final result: Returning {len(docs)} document(s)")
            
            return docs
        except Exception as e:
            print(f"❌ ERROR during similarity search: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Last resort: try simple similarity_search without scores
            print("🔄 Attempting fallback: Simple similarity_search...")
            try:
                docs = self.vectorstore.similarity_search(query, k=k)
                print(f"✅ Fallback search returned {len(docs)} document(s)")
                return docs
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {str(fallback_error)}")
                raise

