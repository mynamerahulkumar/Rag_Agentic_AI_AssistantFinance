# 📋 Interview Guide: RAG Chatbot - Part 1: Document Processing

> **Comprehensive interview questions and answers about Document Processing in RAG (LangGraph)**

---

## 📑 Table of Contents

1. [Architecture Overview](#-architecture-overview)
2. [Document Loading Questions](#-document-loading-questions)
3. [Text Splitting Questions](#-text-splitting-questions)
4. [Document Processing Questions](#-document-processing-questions)
5. [Implementation Questions](#-implementation-questions)
6. [Code Walkthrough](#-code-walkthrough)
7. [Advanced Questions](#-advanced-questions)
8. [Troubleshooting Questions](#-troubleshooting-questions)
9. [Best Practices](#-best-practices)

---

## 🏗️ Architecture Overview

### System Architecture Diagram - Part 1: Document Processing

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI Layer                       │
│  • File Upload Widget (PDF, TXT)                             │
│  • OpenAI API Key Input                                       │
│  • Document Display                                           │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                         │
│  • LoadStreamlitUI.load_streamlit_ui()                       │
│  • Gets uploaded files + OpenAI API key                      │
│  • Configures RAG Module                                      │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Graph Builder Layer                        │
│  GraphBuilder.setup_graph("RAG Chatbot")                      │
│  └──> rag_build_graph(openai_api_key)                       │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Node 1: Process Documents                        │
│                                                               │
│   START ──────────► [process_documents] ──► [retrieve_context]│
│                      │                                        │
│                      │ State['uploaded_files']               │
│                      │                                        │
│                      ▼                                        │
│              RAGNode.process_documents()                       │
│                      │                                        │
│                      ├──► RAGModule.load_documents()         │
│                      │    └──> PyPDFLoader / TextLoader       │
│                      │                                        │
│                      ├──► RAGModule.split_documents()         │
│                      │    └──> RecursiveCharacterTextSplitter │
│                      │                                        │
│                      ├──► RAGModule.create_vectorstore()     │
│                      │    └──> FAISS.from_documents()        │
│                      │                                        │
│                      ▼                                        │
│              Updated State['documents_processed']            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Document Processing Flow

```
┌──────────────────────────────────────────────────────────┐
│              Document Processing Pipeline                  │
│                                                            │
│  1. File Upload (Streamlit)                               │
│     │                                                      │
│     ▼                                                      │
│  2. Save to Temp File                                      │
│     │                                                      │
│     ▼                                                      │
│  3. Detect File Type                                      │
│     ├── PDF → PyPDFLoader                                 │
│     └── TXT → TextLoader                                  │
│     │                                                      │
│     ▼                                                      │
│  4. Load Documents                                         │
│     ├── Extract text from files                           │
│     ├── Create Document objects                           │
│     └── Validate content                                   │
│     │                                                      │
│     ▼                                                      │
│  5. Split Documents into Chunks                           │
│     ├── chunk_size: 1000 characters                       │
│     ├── chunk_overlap: 200 characters                     │
│     └── Create multiple Document chunks                   │
│     │                                                      │
│     ▼                                                      │
│  6. Generate Embeddings (Part 2)                          │
│     │                                                      │
│     ▼                                                      │
│  7. Create Vector Store (Part 2)                          │
│     │                                                      │
│     ▼                                                      │
│  8. Save to Disk (Part 2)                                │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

### State Flow Diagram - Document Processing

```
┌──────────────────────────────────────────────────────────┐
│                    Initial State                           │
│  {                                                          │
│    "uploaded_files": [File1, File2],                      │
│    "messages": [HumanMessage("Process these documents")]  │
│  }                                                          │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│          RAGNode.process_documents()                       │
│                                                            │
│  Step 1: Check for existing vectorstore                   │
│    ├── Memory check (vectorstore_created)                  │
│    ├── Disk check (load_vectorstore)                       │
│    └── If exists: Skip processing                          │
│                                                            │
│  Step 2: Validate uploaded files                           │
│    ├── Check if files exist                               │
│    ├── Check file types (PDF/TXT)                          │
│    └── Raise error if invalid                              │
│                                                            │
│  Step 3: Load Documents                                    │
│    ├── For each file:                                      │
│    │   ├── Save to temp file                               │
│    │   ├── Detect file type                                │
│    │   ├── Load with PyPDFLoader/TextLoader                │
│    │   └── Validate content                                │
│    └── Return list of Document objects                     │
│                                                            │
│  Step 4: Split Documents                                   │
│    ├── Use RecursiveCharacterTextSplitter                  │
│    ├── chunk_size: 1000                                    │
│    ├── chunk_overlap: 200                                  │
│    └── Return list of Document chunks                      │
│                                                            │
│  Step 5: Create Vector Store (calls Part 2)               │
│    ├── Generate embeddings for chunks                      │
│    ├── Create FAISS vectorstore                            │
│    └── Save to disk                                        │
│                                                            │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│                  Updated State                              │
│  {                                                          │
│    "uploaded_files": [File1, File2],                       │
│    "documents_processed": True,                            │
│    "num_chunks": 150,                                      │
|    "vectorstore_source": "created_new"                     │
│  }                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## ❓ Document Loading Questions

### Q1: What is RAG and why do we need document processing?

**Answer:**

**RAG (Retrieval-Augmented Generation)** is a technique that enhances LLM responses by retrieving relevant information from a knowledge base (documents).

**Why Document Processing is Needed:**

1. **LLM Limitations**: LLMs have limited context window and training data cutoff
2. **External Knowledge**: Documents contain information not in LLM training data
3. **Domain-Specific Information**: PDFs/TXT files contain specialized knowledge
4. **Real-Time Updates**: Documents can be updated without retraining LLM

**Document Processing Pipeline:**
```
Documents → Load → Split → Embed → Store → Retrieve → Generate Response
```

**Benefits:**
- **Accuracy**: LLM uses actual document content, not memorized facts
- **Up-to-date**: Can answer questions about recent documents
- **Specificity**: Can answer questions about specific documents (PDFs, reports, etc.)

---

### Q2: How do you load PDF documents?

**Answer:**

**PDF Loading Process:**

1. **Save Uploaded File to Temp Location**
```python
import tempfile
import os

# Save uploaded file to temporary location
with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    tmp_path = tmp_file.name
```

2. **Use PyPDFLoader**
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(tmp_path)
documents = loader.load()
```

3. **Validate Content**
```python
if len(documents) == 0:
    raise ValueError("No documents loaded from PDF")

total_chars = sum(len(doc.page_content) for doc in documents)
if total_chars == 0:
    raise ValueError("PDF contains no extractable text")
```

**Complete Implementation:**
```python
def load_documents(self, uploaded_files: List) -> List:
    documents = []
    
    for uploaded_file in uploaded_files:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Load based on file type
            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_path)
            elif file_extension == 'txt':
                loader = TextLoader(tmp_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Load documents
            loaded_docs = loader.load()
            
            # Validate
            if len(loaded_docs) == 0:
                raise ValueError(f"No content extracted from {uploaded_file.name}")
            
            documents.extend(loaded_docs)
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return documents
```

---

### Q3: What happens if a PDF is image-based (scanned)?

**Answer:**

**Image-Based PDF Problem:**

Scanned PDFs are images, not text, so `PyPDFLoader` cannot extract text.

**Detection:**
```python
loaded_docs = loader.load()
total_chars = sum(len(doc.page_content) for doc in loaded_docs)

if total_chars == 0:
    print("❌ PDF is image-based (scanned)")
    print("   Solution: Use OCR to extract text")
```

**Solutions:**

1. **OCR (Optical Character Recognition)**
```python
# Option 1: Use pytesseract
from pdf2image import convert_from_path
import pytesseract

def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

# Option 2: Use UnstructuredLoader with OCR
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader(
    pdf_path,
    mode="elements",
    strategy="ocr_only"
)
documents = loader.load()
```

2. **Alternative Loaders**
```python
# Use UnstructuredLoader
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader(pdf_path, mode="single")
documents = loader.load()
```

**Error Handling:**
```python
def load_documents_with_ocr_fallback(self, uploaded_files: List) -> List:
    documents = []
    
    for uploaded_file in uploaded_files:
        try:
            # Try normal loading first
            loader = PyPDFLoader(tmp_path)
            loaded_docs = loader.load()
            
            # Check if text was extracted
            if sum(len(doc.page_content) for doc in loaded_docs) == 0:
                # Fallback to OCR
                print("⚠️ No text extracted, trying OCR...")
                loader = UnstructuredPDFLoader(tmp_path, mode="single")
                loaded_docs = loader.load()
            
            documents.extend(loaded_docs)
            
        except Exception as e:
            print(f"❌ Error loading {uploaded_file.name}: {e}")
            raise
    
    return documents
```

---

### Q4: How do you handle multiple file types (PDF, TXT)?

**Answer:**

**Multi-Format Support:**

```python
def load_documents(self, uploaded_files: List) -> List:
    documents = []
    
    for uploaded_file in uploaded_files:
        # Determine file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Route to appropriate loader
            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_path)
            elif file_extension == 'txt':
                loader = TextLoader(tmp_path)
            elif file_extension == 'docx':
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(tmp_path)
            elif file_extension == 'csv':
                from langchain_community.document_loaders import CSVLoader
                loader = CSVLoader(tmp_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Load documents
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            
        except Exception as e:
            print(f"❌ Error loading {uploaded_file.name}: {e}")
            raise
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return documents
```

**Supported Formats:**
- **PDF**: `PyPDFLoader`
- **TXT**: `TextLoader`
- **DOCX**: `Docx2txtLoader`
- **CSV**: `CSVLoader`
- **HTML**: `UnstructuredHTMLLoader`
- **Markdown**: `UnstructuredMarkdownLoader`

---

### Q5: Why do you save files to temporary locations?

**Answer:**

**Reasons for Temp Files:**

1. **File Object Format**: Streamlit uploads are in-memory, not file paths
2. **Loader Requirements**: LangChain loaders expect file paths, not file objects
3. **Temporary Processing**: Files are processed and then discarded
4. **Isolation**: Each upload gets its own temp file, avoiding conflicts

**Process:**
```python
# Streamlit upload
uploaded_file = st.file_uploader(...)  # In-memory file object

# Save to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
    tmp_file.write(uploaded_file.getvalue())  # Write to disk
    tmp_path = tmp_file.name  # Get path

# Loader uses path
loader = PyPDFLoader(tmp_path)  # Loader needs file path

# Clean up
os.unlink(tmp_path)  # Delete temp file
```

**Cleanup:**
```python
try:
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
finally:
    # Always clean up temp file
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
```

---

## ✂️ Text Splitting Questions

### Q6: Why do we need to split documents into chunks?

**Answer:**

**Reasons for Splitting:**

1. **Context Window Limits**: LLMs have token limits (e.g., 4096 tokens)
2. **Embedding Limitations**: Embeddings models have input size limits
3. **Semantic Granularity**: Smaller chunks provide more focused retrieval
4. **Efficient Search**: Smaller chunks = more precise retrieval

**Problem Without Splitting:**
```
Document: 50,000 characters
├── Too large for embedding (max ~8000 characters)
├── Too large for LLM context
└── Retrieval returns entire document (not specific)

Solution: Split into chunks
Document → 150 chunks × 1000 characters each
├── Each chunk can be embedded
├── Each chunk fits in context
└── Retrieval returns specific relevant chunks
```

---

### Q7: What is RecursiveCharacterTextSplitter and how does it work?

**Answer:**

**RecursiveCharacterTextSplitter** splits text by recursively trying different separators.

**How It Works:**

1. **Separator Priority**: Tries separators in order
2. **Recursive Splitting**: If chunk too large, recursively splits
3. **Overlap**: Maintains context between chunks

**Separator Priority:**
```python
separators = [
    "\n\n",      # Paragraphs (highest priority)
    "\n",        # Lines
    " ",         # Words
    ""           # Characters (lowest priority)
]
```

**Process:**
```
1. Try "\n\n" (paragraphs)
   ├── If chunks < max_size → Done
   └── If chunks > max_size → Continue

2. Try "\n" (lines)
   ├── If chunks < max_size → Done
   └── If chunks > max_size → Continue

3. Try " " (words)
   ├── If chunks < max_size → Done
   └── If chunks > max_size → Continue

4. Try "" (characters)
   └── Force split at max_size
```

**Example:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200     # Overlap between chunks
)

# Text
text = "Paragraph 1...\n\nParagraph 2...\n\nParagraph 3..."

# Split
chunks = splitter.split_text(text)

# Result:
# Chunk 1: Paragraph 1 (ends at "\n\n")
# Chunk 2: Paragraph 2 (starts with 200 chars from Chunk 1)
# Chunk 3: Paragraph 3 (starts with 200 chars from Chunk 2)
```

---

### Q8: What are chunk_size and chunk_overlap?

**Answer:**

**chunk_size**: Maximum characters per chunk
- **Default**: 1000 characters
- **Purpose**: Ensures chunks fit within embedding/context limits
- **Trade-off**: Larger = more context, smaller = more granular

**chunk_overlap**: Characters that overlap between adjacent chunks
- **Default**: 200 characters
- **Purpose**: Maintains context continuity between chunks
- **Benefit**: Prevents information loss at chunk boundaries

**Visual Example:**
```
Document: "This is a long document that needs to be split..."
        |------------------------------------------| (1000 chars)
                                        |------------------------| (1000 chars)
                                        ↑
                                   200 chars overlap
```

**Code:**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Each chunk max 1000 characters
    chunk_overlap=200     # 200 characters overlap between chunks
)

chunks = splitter.split_documents(documents)
# Chunk 1: chars 0-1000
# Chunk 2: chars 800-1800  (200 overlap with Chunk 1)
# Chunk 3: chars 1600-2600 (200 overlap with Chunk 2)
```

**Why Overlap Matters:**
```
Without Overlap:
  Chunk 1: "...the revenue was $100M and the profit was..."
  Chunk 2: "...$50M. The company expanded to..."

Problem: "profit was $50M" is split across chunks
         Retrieval might miss the connection

With Overlap:
  Chunk 1: "...the revenue was $100M and the profit was..."
  Chunk 2: "...profit was $50M. The company expanded to..."

Benefit: Both chunks contain "profit was $50M"
         Retrieval finds complete information
```

---

### Q9: How do you choose optimal chunk_size and chunk_overlap?

**Answer:**

**Factors to Consider:**

1. **Embedding Model Limits**
   - OpenAI embeddings: ~8000 tokens (~6000 chars)
   - Use smaller chunks (1000-2000 chars) for safety

2. **LLM Context Window**
   - GPT-4: 8192 tokens
   - Multiple chunks + query + response = need space
   - Typical: 3-5 chunks × 1000 chars = safe

3. **Document Type**
   - **Long paragraphs** (research papers): 2000 chars
   - **Short paragraphs** (articles): 1000 chars
   - **Code**: Smaller chunks (500 chars)

4. **Query Type**
   - **Specific questions**: Smaller chunks (more precise)
   - **General summaries**: Larger chunks (more context)

**Recommended Values:**
```python
# Small documents, precise queries
chunk_size=500, chunk_overlap=100

# General use (default)
chunk_size=1000, chunk_overlap=200

# Large documents, summaries
chunk_size=2000, chunk_overlap=400

# Research papers, reports
chunk_size=1500, chunk_overlap=300
```

**Testing:**
```python
def find_optimal_chunk_size(documents, embedding_model):
    """Test different chunk sizes"""
    sizes = [500, 1000, 1500, 2000]
    results = []
    
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=size // 5
        )
        chunks = splitter.split_documents(documents)
        
        # Measure retrieval quality
        quality = test_retrieval_quality(chunks)
        results.append((size, quality))
    
    # Return best size
    return max(results, key=lambda x: x[1])[0]
```

---

## 💻 Implementation Questions

### Q10: Walk me through the process_documents node implementation.

**Answer:**

**Complete Implementation Flow:**

**1. Check for Existing Vectorstore**
```python
def process_documents(self, state: dict) -> dict:
    # Check if vectorstore already exists in memory
    if self.vectorstore_created and self.rag_module.vectorstore is not None:
        state['documents_processed'] = True
        return state  # Skip processing
    
    # Check if vectorstore exists on disk
    uploaded_files = state.get('uploaded_files', [])
    if not uploaded_files:
        existing_vectorstore = self.rag_module.load_vectorstore()
        if existing_vectorstore:
            self.rag_module.vectorstore = existing_vectorstore
            state['documents_processed'] = True
            return state
```

**2. Load Documents**
```python
    file_names = [f.name for f in uploaded_files]
    
    # Load documents
    documents = self.rag_module.load_documents(uploaded_files)
    
    # Validate
    if len(documents) == 0:
        state['error'] = "No documents loaded"
        return state
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    if total_chars == 0:
        state['error'] = "Documents contain no text"
        return state
```

**3. Split Documents**
```python
    # Split into chunks
    chunks = self.rag_module.split_documents(documents)
    
    if len(chunks) == 0:
        state['error'] = "No chunks created"
        return state
```

**4. Create Vector Store**
```python
    # Create vectorstore (includes embeddings + FAISS)
    self.rag_module.find_or_create_vectorstore(
        file_names=file_names,
        chunks=chunks
    )
    
    self.vectorstore_created = True
    state['documents_processed'] = True
    state['num_chunks'] = len(chunks)
    return state
```

---

### Q11: How do you handle errors in document processing?

**Answer:**

**Error Handling Strategy:**

**1. File Loading Errors**
```python
try:
    loader = PyPDFLoader(tmp_path)
    loaded_docs = loader.load()
except Exception as e:
    print(f"❌ Error loading {uploaded_file.name}: {e}")
    state['error'] = f"Failed to load {uploaded_file.name}: {str(e)}"
    return state
```

**2. Content Validation**
```python
if len(loaded_docs) == 0:
    state['error'] = "No content extracted from PDF. PDF might be image-based (scanned)."
    return state

total_chars = sum(len(doc.page_content) for doc in loaded_docs)
if total_chars == 0:
    state['error'] = "PDF loaded but contains no text. This might be a scanned PDF."
    return state
```

**3. Splitting Errors**
```python
try:
    chunks = self.rag_module.split_documents(documents)
    if len(chunks) == 0:
        state['error'] = "Failed to split documents into chunks"
        return state
except Exception as e:
    print(f"❌ Error splitting documents: {e}")
    state['error'] = f"Error splitting documents: {str(e)}"
    return state
```

**4. Vectorstore Creation Errors**
```python
try:
    self.rag_module.find_or_create_vectorstore(file_names, chunks)
except Exception as e:
    print(f"❌ Error creating vectorstore: {e}")
    state['error'] = f"Error creating vectorstore: {str(e)}"
    import traceback
    state['error_traceback'] = traceback.format_exc()
    return state
```

**Complete Error Handling:**
```python
def process_documents(self, state: dict) -> dict:
    try:
        # ... processing steps ...
        return state
    except Exception as e:
        print(f"❌ Error processing documents: {str(e)}")
        import traceback
        traceback.print_exc()
        state['error'] = f"Error processing documents: {str(e)}"
        state['error_traceback'] = traceback.format_exc()
        return state
```

---

### Q12: How do you validate that documents were loaded correctly?

**Answer:**

**Validation Checks:**

**1. Document Count**
```python
if len(documents) == 0:
    raise ValueError("No documents loaded from files")
```

**2. Content Extraction**
```python
total_chars = sum(len(doc.page_content) for doc in documents)
if total_chars == 0:
    raise ValueError("Documents loaded but contain no text")
```

**3. Empty Documents Check**
```python
empty_docs = sum(1 for doc in documents if not doc.page_content or len(doc.page_content.strip()) == 0)
if empty_docs > 0:
    print(f"⚠️ WARNING: {empty_docs} documents are empty")
```

**4. Content Preview**
```python
if documents[0].page_content:
    preview = documents[0].page_content[:200]
    print(f"📄 First document preview: {preview}...")
else:
    print("⚠️ WARNING: First document has no content")
```

**5. Chunk Validation**
```python
chunks = self.rag_module.split_documents(documents)

if len(chunks) == 0:
    raise ValueError("No chunks created from documents")

empty_chunks = sum(1 for chunk in chunks if not chunk.page_content or len(chunk.page_content.strip()) == 0)
if empty_chunks > 0:
    print(f"⚠️ WARNING: {empty_chunks} chunks are empty")
```

**Complete Validation Function:**
```python
def validate_documents(self, documents: List) -> bool:
    """Validate documents are loaded correctly"""
    if len(documents) == 0:
        return False
    
    # Check content
    total_chars = sum(len(doc.page_content) if doc.page_content else 0 for doc in documents)
    if total_chars == 0:
        return False
    
    # Check for empty documents
    empty_docs = sum(1 for doc in documents if not doc.page_content or len(doc.page_content.strip()) == 0)
    if empty_docs == len(documents):
        return False
    
    # Check for reasonable content
    avg_chars = total_chars / len(documents)
    if avg_chars < 10:  # Too short
        return False
    
    return True
```

---

## 🚀 Advanced Questions

### Q13: How would you optimize document loading for large files?

**Answer:**

**Optimization Strategies:**

**1. Streaming for Large Files**
```python
from langchain_community.document_loaders import PyPDFLoader

def load_large_pdf_streaming(pdf_path, chunk_size=1000):
    """Load PDF in chunks to avoid memory issues"""
    loader = PyPDFLoader(pdf_path)
    
    documents = []
    for page in loader.lazy_load():  # Lazy loading
        if len(page.page_content) > chunk_size:
            # Split large pages
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
            chunks = splitter.split_documents([page])
            documents.extend(chunks)
        else:
            documents.append(page)
    
    return documents
```

**2. Parallel Processing**
```python
from concurrent.futures import ThreadPoolExecutor

def load_multiple_files_parallel(uploaded_files):
    """Load multiple files in parallel"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in uploaded_files:
            future = executor.submit(load_single_file, file)
            futures.append(future)
        
        documents = []
        for future in futures:
            docs = future.result()
            documents.extend(docs)
    
    return documents
```

**3. Progress Tracking**
```python
def load_documents_with_progress(uploaded_files, progress_callback):
    """Load documents with progress updates"""
    documents = []
    total_files = len(uploaded_files)
    
    for idx, file in enumerate(uploaded_files, 1):
        progress_callback(idx, total_files, f"Loading {file.name}...")
        docs = load_single_file(file)
        documents.extend(docs)
    
    return documents
```

**4. Caching**
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10)
def load_cached_document(file_hash, file_path):
    """Cache loaded documents"""
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_documents_cached(uploaded_files):
    documents = []
    for file in uploaded_files:
        file_hash = hashlib.md5(file.getvalue()).hexdigest()
        docs = load_cached_document(file_hash, tmp_path)
        documents.extend(docs)
    return documents
```

---

### Q14: How would you handle documents with different languages or special formatting?

**Answer:**

**Multi-Language Support:**

**1. Language Detection**
```python
from langdetect import detect

def detect_language(text):
    """Detect document language"""
    return detect(text)

def load_multilingual_documents(uploaded_files):
    """Load documents with language detection"""
    documents = []
    
    for file in uploaded_files:
        docs = load_single_file(file)
        
        for doc in docs:
            # Detect language
            language = detect_language(doc.page_content)
            doc.metadata['language'] = language
            
            # Store language in metadata
            documents.append(doc)
    
    return documents
```

**2. Special Formatting Handling**
```python
def clean_document_content(text):
    """Clean special formatting"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix line breaks
    text = text.replace('\r\n', '\n')
    
    # Remove special characters (optional)
    # text = re.sub(r'[^\w\s]', '', text)
    
    return text

def load_documents_clean(uploaded_files):
    """Load and clean documents"""
    documents = []
    
    for file in uploaded_files:
        docs = load_single_file(file)
        
        for doc in docs:
            # Clean content
            doc.page_content = clean_document_content(doc.page_content)
            documents.append(doc)
    
    return documents
```

**3. Preserve Formatting in Metadata**
```python
def load_documents_preserve_formatting(uploaded_files):
    """Load documents while preserving formatting info"""
    documents = []
    
    for file in uploaded_files:
        docs = load_single_file(file)
        
        for doc in docs:
            # Store formatting info in metadata
            doc.metadata['original_length'] = len(doc.page_content)
            doc.metadata['has_tables'] = '|' in doc.page_content
            doc.metadata['has_code'] = '```' in doc.page_content
            
            documents.append(doc)
    
    return documents
```

---

## 🐛 Troubleshooting Questions

### Q15: What if documents fail to load?

**Answer:**

**Common Failures and Solutions:**

**1. PDF Loading Failure**
```python
# Check if PDF is corrupted
try:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
except Exception as e:
    if "PDF" in str(e) or "corrupted" in str(e).lower():
        # Try alternative loader
        from pdfplumber import PDF
        with PDF(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages])
    raise
```

**2. Memory Issues**
```python
# Process files one at a time
def load_large_files(uploaded_files):
    """Load large files one at a time to avoid memory issues"""
    all_documents = []
    
    for file in uploaded_files:
        try:
            docs = load_single_file(file)
            all_documents.extend(docs)
            
            # Clear memory after each file
            import gc
            gc.collect()
        except MemoryError:
            print(f"⚠️ File {file.name} is too large, skipping...")
            continue
    
    return all_documents
```

**3. File Type Detection Errors**
```python
def safe_load_document(uploaded_file):
    """Safely load document with fallbacks"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Try primary loader
    try:
        if file_extension == 'pdf':
            return PyPDFLoader(tmp_path).load()
    except:
        pass
    
    # Fallback to unstructured
    try:
        from langchain_community.document_loaders import UnstructuredFileLoader
        return UnstructuredFileLoader(tmp_path).load()
    except:
        raise ValueError(f"Cannot load {uploaded_file.name}")
```

---

### Q16: What if splitting creates too many or too few chunks?

**Answer:**

**Dynamic Chunk Size Adjustment:**

**1. Monitor Chunk Count**
```python
def split_documents_adaptive(documents, target_chunks=100):
    """Adaptively split documents to target chunk count"""
    # Estimate current chunk size
    total_chars = sum(len(doc.page_content) for doc in documents)
    estimated_chunk_size = total_chars / target_chunks
    
    # Adjust based on document type
    if estimated_chunk_size < 500:
        chunk_size = 500
    elif estimated_chunk_size > 2000:
        chunk_size = 2000
    else:
        chunk_size = int(estimated_chunk_size)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5
    )
    
    chunks = splitter.split_documents(documents)
    
    # If still off, adjust
    if len(chunks) > target_chunks * 1.5:
        # Too many chunks, increase size
        chunk_size = int(chunk_size * 1.5)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        chunks = splitter.split_documents(documents)
    
    return chunks
```

**2. Set Minimum/Maximum Chunks**
```python
def split_with_limits(documents, min_chunks=10, max_chunks=500):
    """Split documents within chunk limits"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = splitter.split_documents(documents)
    
    if len(chunks) < min_chunks:
        # Too few chunks, decrease size
        splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        chunks = splitter.split_documents(documents)
    
    if len(chunks) > max_chunks:
        # Too many chunks, increase size
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        chunks = splitter.split_documents(documents)
    
    return chunks
```

---

## ✅ Best Practices

### Q17: What are best practices for document processing?

**Answer:**

**1. Error Handling**
```python
# Always wrap loading in try-except
try:
    documents = load_documents(uploaded_files)
except Exception as e:
    logger.error(f"Error loading documents: {e}")
    return error_state
```

**2. Validation**
```python
# Validate after each step
documents = load_documents(uploaded_files)
if not validate_documents(documents):
    return error_state

chunks = split_documents(documents)
if len(chunks) == 0:
    return error_state
```

**3. Logging**
```python
# Log progress at each step
logger.info(f"Loading {len(uploaded_files)} files...")
documents = load_documents(uploaded_files)
logger.info(f"Loaded {len(documents)} documents")

logger.info(f"Splitting into chunks...")
chunks = split_documents(documents)
logger.info(f"Created {len(chunks)} chunks")
```

**4. Memory Management**
```python
# Clean up temp files
try:
    documents = load_documents(uploaded_files)
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
```

**5. Progress Tracking**
```python
# Show progress to user
with st.spinner(f"Loading {file.name}..."):
    documents = load_documents([file])

st.success(f"Loaded {len(documents)} documents")
```

**6. Configurable Parameters**
```python
class RAGModule:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
```

---

## 📝 Summary

### Key Takeaways

1. **Document Loading**: Use appropriate loaders (PyPDFLoader, TextLoader) based on file type
2. **Error Handling**: Always validate documents and handle errors gracefully
3. **Text Splitting**: Use RecursiveCharacterTextSplitter with appropriate chunk_size and overlap
4. **Validation**: Check document count, content, and chunk quality
5. **Temp Files**: Save uploaded files to temp location for loaders

### Document Processing Flow
```
Upload → Save Temp → Load → Validate → Split → Validate → Ready for Embedding
```

### Interview Tips

- **Explain Loading Process**: How PyPDFLoader works, temp file usage
- **Text Splitting**: Why chunks are needed, how RecursiveCharacterTextSplitter works
- **Error Handling**: How to handle scanned PDFs, corrupted files
- **Validation**: Why validation is important at each step
- **Optimization**: How to handle large files, parallel processing

---

**End of Interview Guide: RAG Part 1 - Document Processing**

Continue to:
- `README_INTERVIEW_RAG_PART2_EMBEDDINGS_VECTORSTORE.md`
- `README_INTERVIEW_RAG_PART3_RETRIEVAL_RESPONSE.md`

