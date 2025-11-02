from src.langgraphagenticai.RAG.rag_module import RAGModule
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import os

class RAGNode:
    """
    RAG Node for document processing and question answering.
    """
    
    def __init__(self, llm, openai_api_key: str):
        """
        Initialize the RAGNode.
        
        Args:
            llm: Language model for generating responses
            openai_api_key: OpenAI API key for embeddings
        """
        self.llm = llm
        self.rag_module = RAGModule(openai_api_key)
        self.state = {}
        self.vectorstore_created = False
    
    def process_documents(self, state: dict) -> dict:
        """
        Process uploaded documents and create vector store.
        
        Args:
            state: State dictionary containing 'uploaded_files' and 'messages'
            
        Returns:
            Updated state with vector store information
        """
        print("=" * 50)
        print("STEP 1: Processing Documents")
        print("=" * 50)
        
        # Debug: Print state keys and uploaded files info
        print(f"🔍 State keys: {list(state.keys())}")
        print(f"🔍 Vectorstore already exists: {self.vectorstore_created}")
        print(f"🔍 Vectorstore object exists: {self.rag_module.vectorstore is not None}")
        
        uploaded_files = state.get('uploaded_files', [])
        print(f"📄 Number of uploaded files: {len(uploaded_files)}")
        
        # If vectorstore already exists in memory, skip reprocessing
        if self.vectorstore_created and self.rag_module.vectorstore is not None:
            print("♻️ Vectorstore already exists in memory - skipping document processing")
            print("   Reusing existing vectorstore for retrieval")
            state['documents_processed'] = True
            state['skip_processing'] = True
            return state
        
        # If no files uploaded, try to load vectorstore from disk (for subsequent queries)
        if not uploaded_files or len(uploaded_files) == 0:
            print("ℹ️  No documents uploaded in this request")
            print("   Checking for existing vectorstore on disk...")
            
            # Try to find any existing vectorstore in the persist directory
            persist_dir = self.rag_module.persist_directory
            if os.path.exists(persist_dir):
                vectorstore_dirs = [d for d in os.listdir(persist_dir) if os.path.isdir(os.path.join(persist_dir, d)) and d.startswith("vectorstore_")]
                if vectorstore_dirs:
                    # Use the most recent vectorstore
                    latest_dir = max(vectorstore_dirs, key=lambda d: os.path.getmtime(os.path.join(persist_dir, d)))
                    vectorstore_path = os.path.join(persist_dir, latest_dir)
                    print(f"   📂 Found existing vectorstore: {latest_dir}")
                    existing_vectorstore = self.rag_module.load_vectorstore(persist_directory=vectorstore_path)
                    if existing_vectorstore:
                        print("   ✅ Loaded existing vectorstore from disk")
                        self.rag_module.vectorstore = existing_vectorstore
                        self.vectorstore_created = True
                        state['documents_processed'] = True
                        state['vectorstore_source'] = "loaded_from_disk"
                        return state
            
            print("   ⚠️  No existing vectorstore found - user needs to upload documents first")
            state['error'] = "No documents uploaded and no existing vectorstore found. Please upload a document first."
            return state
        
        try:
            # Get file names for vectorstore persistence
            file_names = [f.name for f in uploaded_files]
            print(f"📋 Files to process: {', '.join(file_names)}")
            
            # Try to load existing vectorstore first
            print("🔍 Checking for existing vectorstore on disk...")
            existing_vectorstore = self.rag_module.load_vectorstore(file_names=file_names)
            
            if existing_vectorstore:
                print("✅ Found existing vectorstore - using it!")
                self.rag_module.vectorstore = existing_vectorstore
                self.vectorstore_created = True
                state['documents_processed'] = True
                state['vectorstore_source'] = "loaded_from_disk"
                state['num_chunks'] = len(existing_vectorstore.index_to_docstore_id) if hasattr(existing_vectorstore, 'index_to_docstore_id') else 0
                print(f"✅ Using existing vectorstore with {state['num_chunks']} embeddings")
                return state
            
            print("📝 No existing vectorstore found - processing documents...")
            
            print("📖 Loading documents...")
            # Load documents
            documents = self.rag_module.load_documents(uploaded_files)
            print(f"✅ Loaded {len(documents)} document(s)")
            
            if len(documents) == 0:
                print("❌ CRITICAL ERROR: No documents were loaded from files!")
                print("   This could mean:")
                print("   - PDF is image-based (scanned) - needs OCR")
                print("   - PDF is protected/encrypted")
                print("   - PDF is corrupted")
                print("   - File is empty")
                state['error'] = "No content extracted from uploaded files. PDF might be image-based (scanned) and need OCR, or it might be protected/encrypted."
                return state
            
            # Check if documents have actual content
            total_chars = sum(len(doc.page_content) if doc.page_content else 0 for doc in documents)
            if total_chars == 0:
                print("❌ CRITICAL ERROR: Documents loaded but contain no text!")
                print("   PDF appears to be image-based or unreadable")
                state['error'] = "PDF loaded but contains no extractable text. This might be a scanned PDF that requires OCR."
                return state
            
            print(f"✅ Documents contain {total_chars} characters of text")
            
            print("✂️ Splitting documents into chunks...")
            # Split documents into chunks
            chunks = self.rag_module.split_documents(documents)
            print(f"✅ Created {len(chunks)} chunk(s)")
            
            if len(chunks) == 0:
                print("⚠️ WARNING: No chunks created from documents!")
                state['error'] = "No chunks created from documents"
                return state
            
            print("🔧 Creating and saving vector store to disk...")
            print(f"   Creating embeddings for {len(chunks)} chunks...")
            print(f"   This will be saved to disk for future use")
            # Create vector store and save to disk
            self.rag_module.find_or_create_vectorstore(file_names=file_names, chunks=chunks)
            self.vectorstore_created = True
            print("✅ Vector store created and saved to disk successfully")
            
            # Verify vectorstore has documents with multiple test queries
            print("🔍 Verifying vectorstore contains documents...")
            verification_queries = ["test", "the", "a", chunks[0].page_content[:50] if chunks else "test"]
            verified = False
            for vq in verification_queries:
                try:
                    test_docs = self.rag_module.vectorstore.similarity_search(vq, k=min(5, len(chunks)))
                    if len(test_docs) > 0:
                        print(f"✅ Verified: Vectorstore contains documents (query '{vq[:30]}...' returned {len(test_docs)} docs)")
                        verified = True
                        # Show a preview of what's in the vectorstore
                        if test_docs[0].page_content:
                            preview = test_docs[0].page_content[:150]
                            print(f"   Sample content: {preview}...")
                        break
                except Exception as ve:
                    print(f"   Verification with '{vq[:30]}...' failed: {str(ve)}")
                    continue
            
            if not verified:
                print("❌ CRITICAL: Could not verify vectorstore has any documents!")
                print("   Vectorstore might be empty or corrupted")
                state['error'] = "Vectorstore created but appears to be empty"
                return state
            
            state['documents_processed'] = True
            state['num_chunks'] = len(chunks)
            print(f"✅ Documents processed: {len(chunks)} chunks ready")
            print("=" * 50)
            return state
            
        except Exception as e:
            print(f"❌ Error processing documents: {str(e)}")
            import traceback
            traceback.print_exc()
            state['error'] = f"Error processing documents: {str(e)}"
            state['error_traceback'] = traceback.format_exc()
            print(f"🚨 ERROR DETAILS:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"   Full traceback:\n{traceback.format_exc()}")
            return state
    
    def retrieve_context(self, state: dict) -> dict:
        """
        Retrieve relevant context from vector store based on user query.
        
        Args:
            state: State dictionary containing 'messages' with user query
            
        Returns:
            Updated state with retrieved context
        """
        print("=" * 50)
        print("STEP 2: Retrieving Context")
        print("=" * 50)
        
        # First check if there was an error in previous step
        if 'error' in state:
            print(f"❌ Error from previous step: {state['error']}")
            print("   Cannot retrieve context because document processing failed")
            return state
        
        # Debug: Check vectorstore status
        print(f"🔍 Vectorstore created flag: {self.vectorstore_created}")
        print(f"🔍 Vectorstore object exists: {self.rag_module.vectorstore is not None}")
        
        # If vectorstore doesn't exist but documents were processed, something went wrong
        # But we should still try to retrieve if vectorstore exists
        if self.rag_module.vectorstore is None:
            print("❌ Error: Vector store not initialized")
            print("   This might mean documents weren't processed in previous step")
            print("   Checking if documents were processed...")
            
            # Check if documents_processed flag is in state
            if state.get('documents_processed'):
                print("   ⚠️ Documents were marked as processed but vectorstore doesn't exist!")
                print("   This indicates a bug - vectorstore should have been created")
            else:
                print("   ❌ Documents were NOT processed successfully in Step 1")
                print("   Check Step 1 logs for the actual error")
            
            state['error'] = "Vector store not initialized - document processing likely failed"
            return state
        
        # Get user query from messages
        user_query = state['messages'][0].content if state.get('messages') else ""
        print(f"🔍 User query: {user_query}")
        
        if not user_query:
            print("❌ Error: No query provided")
            state['error'] = "No query provided"
            return state
        
        try:
            print("🔎 Starting retrieval process...")
            print(f"   User query: '{user_query}'")
            print("   Calling retrieve_documents...")
            
            # Retrieve relevant documents with timeout handling
            import time
            start_time = time.time()
            
            try:
                # Increase k to 5 to ensure we get more results
                retrieved_docs = self.rag_module.retrieve_documents(user_query, k=5)
                elapsed_time = time.time() - start_time
                print(f"⏱️ Retrieval took {elapsed_time:.2f} seconds")
            except Exception as retrieval_error:
                print(f"❌ ERROR in retrieve_documents: {str(retrieval_error)}")
                import traceback
                traceback.print_exc()
                state['error'] = f"Error during retrieval: {str(retrieval_error)}"
                return state
            
            print(f"✅ Retrieved {len(retrieved_docs)} relevant chunk(s)")
            
            if len(retrieved_docs) == 0:
                print("⚠️ WARNING: No documents retrieved from vector store!")
                print("   This might indicate:")
                print("   - Vector store is empty")
                print("   - Embedding mismatch")
                print("   - Query doesn't match any content")
                print("   - Embedding generation issue")
                
                # Try to diagnose and force retrieval
                print("   Attempting diagnostic test and forced retrieval...")
                try:
                    # Try with the actual query again but with more lenient settings
                    test_docs = self.rag_module.vectorstore.similarity_search(user_query, k=10)
                    print(f"   Diagnostic test with k=10 returned {len(test_docs)} docs")
                    
                    if len(test_docs) > 0:
                        print("   ✅ Vector store has documents - using all retrieved docs!")
                        retrieved_docs = test_docs[:5]  # Use top 5
                        print(f"   ✅ Forced retrieval returned {len(retrieved_docs)} docs")
                    else:
                        # Try with a generic query
                        print("   Trying generic query...")
                        generic_docs = self.rag_module.vectorstore.similarity_search("the", k=5)
                        if len(generic_docs) > 0:
                            print(f"   ✅ Generic query returned {len(generic_docs)} docs - using these!")
                            retrieved_docs = generic_docs
                        else:
                            # Last resort: try to get ANY documents
                            print("   Last resort: trying empty query...")
                            any_docs = self.rag_module.vectorstore.similarity_search("", k=5)
                            if len(any_docs) > 0:
                                print(f"   ✅ Empty query returned {len(any_docs)} docs!")
                                retrieved_docs = any_docs
                            else:
                                print("   ❌ Vector store appears to be completely empty!")
                                state['retrieved_context'] = ""
                                state['query'] = user_query
                                state['retrieved_docs'] = []
                                return state
                except Exception as diag_error:
                    print(f"   ❌ Diagnostic test failed: {str(diag_error)}")
                    import traceback
                    traceback.print_exc()
                    # Still return empty to avoid crashing
                    state['retrieved_context'] = ""
                    state['query'] = user_query
                    state['retrieved_docs'] = []
                    return state
            
            print("📝 Formatting retrieved context...")
            # Format retrieved context - ensure we have valid content
            context_parts = []
            for doc in retrieved_docs:
                if doc.page_content and len(doc.page_content.strip()) > 0:
                    context_parts.append(doc.page_content.strip())
            
            context = "\n\n".join(context_parts)
            print(f"📝 Context length: {len(context)} characters")
            print(f"📄 First 200 chars of context: {context[:200]}...")
            
            if len(context.strip()) == 0:
                print("⚠️ WARNING: Context is empty after formatting!")
                print("   Document page_content might be empty")
                state['error'] = "Retrieved documents but context is empty"
                return state
            
            # Store in state - make sure we're modifying the state dict
            state['query'] = user_query
            state['retrieved_context'] = context
            # Store retrieved_docs as list of page_content strings for serialization
            state['retrieved_docs_content'] = [doc.page_content for doc in retrieved_docs]
            
            print("✅ Context retrieved and formatted successfully")
            print(f"🔍 DEBUG: State keys before return: {list(state.keys())}")
            print(f"🔍 DEBUG: Retrieved context length: {len(state.get('retrieved_context', ''))}")
            print(f"🔍 DEBUG: Query value: '{state.get('query', '')}'")
            print(f"🔍 DEBUG: State 'query' key exists: {'query' in state}")
            print(f"🔍 DEBUG: State 'retrieved_context' key exists: {'retrieved_context' in state}")
            
            # Return state with all fields explicitly
            result_state = dict(state)  # Make a copy to ensure all fields are included
            result_state['query'] = user_query
            result_state['retrieved_context'] = context
            
            print(f"🔍 DEBUG: Result state keys: {list(result_state.keys())}")
            print(f"🔍 DEBUG: Result state query: '{result_state.get('query', 'MISSING')}'")
            print(f"🔍 DEBUG: Result state context length: {len(result_state.get('retrieved_context', ''))}")
            
            return result_state
            
        except Exception as e:
            print(f"❌ Error retrieving context: {str(e)}")
            import traceback
            traceback.print_exc()
            state['error'] = f"Error retrieving context: {str(e)}"
            return state
    
    def generate_response(self, state: dict) -> dict:
        """
        Generate LLM response with retrieved context.
        
        Args:
            state: State dictionary containing 'retrieved_context' and 'query'
            
        Returns:
            Updated state with LLM response
        """
        print("=" * 50)
        print("STEP 3: Generating Response")
        print("=" * 50)
        
        # Debug: Print entire state to see what we have
        print(f"🔍 State keys: {list(state.keys())}")
        print(f"🔍 State values preview:")
        for key in state.keys():
            if key == 'messages':
                print(f"   {key}: {len(state[key])} messages")
            elif key == 'uploaded_files':
                print(f"   {key}: {len(state[key])} files")
            else:
                value = state.get(key, '')
                if isinstance(value, str):
                    print(f"   {key}: '{value[:100] if len(str(value)) > 100 else value}' (length: {len(value)})")
                else:
                    print(f"   {key}: {value}")
        
        if 'error' in state:
            print(f"❌ Error found in state: {state['error']}")
            state['messages'] = [HumanMessage(content=f"Error: {state['error']}")]
            return state
        
        # Try multiple ways to get query and context
        query = state.get('query', '')
        context = state.get('retrieved_context', '')
        
        # If query is empty, try to get it from messages
        if not query and state.get('messages'):
            try:
                query = state['messages'][0].content
                print(f"🔍 Got query from messages: '{query}'")
            except:
                pass
        
        print(f"❓ Query from state: '{query}'")
        print(f"📄 Context available: {len(context) > 0}")
        print(f"📄 Context length: {len(context)} characters")
        
        # Debug: Check if retrieved_context exists but is empty
        if 'retrieved_context' not in state:
            print("⚠️ WARNING: 'retrieved_context' key not in state!")
            print(f"   Available keys: {list(state.keys())}")
        elif state.get('retrieved_context') == '':
            print("⚠️ WARNING: 'retrieved_context' exists but is empty string!")
        else:
            print(f"✅ 'retrieved_context' found in state with {len(context)} characters")
        
        # If context is missing but retrieved_docs_content exists, recreate it
        if not context and state.get('retrieved_docs_content'):
            print("🔄 Recreating context from retrieved_docs_content...")
            context = "\n\n".join(state['retrieved_docs_content'])
            state['retrieved_context'] = context
            print(f"✅ Recreated context: {len(context)} characters")
        
        if not context or len(context.strip()) == 0:
            print("❌ No context in state!")
            print("   Checking if retrieved_docs exists...")
            retrieved_docs = state.get('retrieved_docs', [])
            print(f"   Retrieved docs count: {len(retrieved_docs)}")
            
            # If we have retrieved_docs but no context, recreate context
            if len(retrieved_docs) > 0:
                print("   🔄 Recreating context from retrieved_docs...")
                context = "\n\n".join([doc.page_content for doc in retrieved_docs if doc.page_content])
                state['retrieved_context'] = context
                print(f"   ✅ Recreated context: {len(context)} characters")
            else:
                print("   ⚠️ No documents were retrieved in previous step!")
                state['messages'] = [HumanMessage(content="No relevant documents found in the uploaded files. Please try a different question or check if the documents contain relevant information.")]
                return state
        
        if context:
            print(f"📄 Context found: {len(context)} characters")
            print(f"📄 First 300 chars: {context[:300]}...")
        else:
            print("❌ No relevant context found")
            state['messages'] = [HumanMessage(content="No relevant context found")]
            return state
        
        try:
            print("🤖 Generating LLM response...")
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions based on the provided context. 
                Use only the information from the context to answer the question. 
                If the context doesn't contain enough information, say so."""),
                ("user", "Context:\n{context}\n\nQuestion: {query}")
            ])
            
            # Generate response
            formatted_prompt = prompt_template.format(context=context, query=query)
            response = self.llm.invoke(formatted_prompt)
            
            print("✅ Response generated successfully")
            print(f"📝 Response length: {len(response.content) if hasattr(response, 'content') else 'N/A'} characters")
            
            # Update state with response
            state['messages'] = [response]
            
            print("=" * 50)
            print("✅ RAG Pipeline Complete")
            print("=" * 50)
            
            return state
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            import traceback
            traceback.print_exc()
            state['error'] = f"Error generating response: {str(e)}"
            state['messages'] = [HumanMessage(content=f"Error: {str(e)}")]
            return state

