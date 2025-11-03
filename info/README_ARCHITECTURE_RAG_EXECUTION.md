# 🏗️ Architecture & Execution Flow: LangGraph RAG Agent

> **Comprehensive documentation of the architecture and execution flow of the RAG (Retrieval-Augmented Generation) Chatbot built with LangGraph**

---

## 📑 Table of Contents

1. [System Architecture Overview](#-system-architecture-overview)
2. [Component Architecture](#-component-architecture)
3. [Execution Flow Diagram](#-execution-flow-diagram)
4. [State Management](#-state-management)
5. [Data Flow Architecture](#-data-flow-architecture)
6. [Step-by-Step Execution](#-step-by-step-execution)
7. [Persistence & Caching](#-persistence--caching)
8. [Error Handling Flow](#-error-handling-flow)
9. [Component Interactions](#-component-interactions)
10. [Performance Considerations](#-performance-considerations)

---

## 🏗️ System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT UI LAYER                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │  User Interface  │  │  File Upload     │  │  API Key Input    │    │
│  │  • Chat Input    │  │  • PDF/TXT       │  │  • OpenAI Key     │    │
│  │  • Messages      │  │  • Multiple Files│  │  • Groq Key       │    │
│  │  • Status Updates│  │  • Validation    │  │  • Validation     │    │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘    │
└───────────┼──────────────────────┼──────────────────────┼──────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      APPLICATION ORCHESTRATION LAYER                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │ LoadStreamlitUI  │  │  GroqLLM         │  │ GraphBuilder     │    │
│  │ • UI Loading     │  │  • LLM Config    │  │ • Graph Setup    │    │
│  │ • Input Collect  │  │  • Model Init    │  │ • Node Addition  │    │
│  │ • Validation     │  │  • API Config    │  │ • Edge Creation  │    │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘    │
└───────────┼──────────────────────┼──────────────────────┼──────────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH EXECUTION LAYER                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                      RAG WORKFLOW GRAPH                              │ │
│  │                                                                       │ │
│  │   START                                                               │ │
│  │     │                                                                 │ │
│  │     ▼                                                                 │ │
│  │  ┌──────────────────────┐                                            │ │
│  │  │ process_documents    │                                            │ │
│  │  │ • Load Documents     │                                            │ │
│  │  │ • Split Chunks       │                                            │ │
│  │  │ • Create Embeddings  │                                            │ │
│  │  │ • Build Vectorstore  │                                            │ │
│  │  └──────────┬───────────┘                                            │ │
│  │             │                                                         │ │
│  │             ▼                                                         │ │
│  │  ┌──────────────────────┐                                            │ │
│  │  │ retrieve_context     │                                            │ │
│  │  │ • Extract Query      │                                            │ │
│  │  │ • Similarity Search   │                                            │ │
│  │  │ • Format Context      │                                            │ │
│  │  └──────────┬───────────┘                                            │ │
│  │             │                                                         │ │
│  │             ▼                                                         │ │
│  │  ┌──────────────────────┐                                            │ │
│  │  │ generate_response    │                                            │ │
│  │  │ • Create Prompt      │                                            │ │
│  │  │ • LLM Invocation     │                                            │ │
│  │  │ • Format Response    │                                            │ │
│  │  └──────────┬───────────┘                                            │ │
│  │             │                                                         │ │
│  │             ▼                                                         │ │
│  │            END                                                         │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CORE RAG COMPONENTS LAYER                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │   RAGModule      │  │   RAGNode        │  │   State          │    │
│  │  • Doc Loading   │  │  • Orchestration │  │  • TypedDict     │    │
│  │  • Text Splitting│  │  • Error Handle │  │  • State Mgmt    │    │
│  │  • Embeddings   │  │  • State Updates │  │  • Validation     │    │
│  │  • Vectorstore   │  │                  │  │                  │    │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘    │
└───────────┼──────────────────────┼──────────────────────┼──────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL SERVICES LAYER                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │  OpenAI API      │  │   Groq API       │  │   FAISS Library  │    │
│  │  • Embeddings    │  │  • LLM Models    │  │  • Vector Search │    │
│  │  • API Calls     │  │  • Text Gen      │  │  • Index Storage │    │
│  │  • Rate Limits   │  │  • Streaming     │  │  • Persistence   │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PERSISTENCE LAYER                                │
│  ┌──────────────────┐  ┌──────────────────┐                            │
│  │  Vectorstore DB  │  │  Metadata Store  │                            │
│  │  • FAISS Index   │  │  • File Info     │                            │
│  │  • Embeddings    │  │  • Chunk Count   │                            │
│  │  • Disk Storage  │  │  • Timestamps    │                            │
│  └──────────────────┘  └──────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Component Architecture

### 1. **LoadStreamlitUI** (`loadui.py`)
**Purpose**: Manages Streamlit UI initialization and user input collection

**Responsibilities**:
- Initialize Streamlit page configuration
- Render sidebar with LLM and use case selection
- Handle file uploads (PDF/TXT) for RAG
- Collect API keys (OpenAI, Groq)
- Validate user inputs

**Key Methods**:
- `load_streamlit_ui()`: Main method to load and configure UI
- `initialize_session()`: Initialize Streamlit session state

**Inputs**:
- User selections (LLM, model, use case)
- Uploaded files (for RAG)
- API keys

**Outputs**:
- `user_controls` dictionary with all user inputs

---

### 2. **GraphBuilder** (`graph_builder.py`)
**Purpose**: Builds and configures LangGraph workflows

**Responsibilities**:
- Initialize StateGraph with State TypedDict
- Add nodes to the graph
- Define edges between nodes
- Compile the graph for execution

**Key Methods**:
- `__init__(model)`: Initialize with LLM model
- `rag_build_graph(openai_api_key)`: Build RAG workflow graph
- `setup_graph(usecase, openai_api_key)`: Setup graph based on use case

**Graph Structure for RAG**:
```python
START → process_documents → retrieve_context → generate_response → END
```

**Node Addition**:
```python
graph_builder.add_node("process_documents", rag_node.process_documents)
graph_builder.add_node("retrieve_context", rag_node.retrieve_context)
graph_builder.add_node("generate_response", rag_node.generate_response)
```

**Edge Configuration**:
```python
graph_builder.set_entry_point("process_documents")
graph_builder.add_edge("process_documents", "retrieve_context")
graph_builder.add_edge("retrieve_context", "generate_response")
graph_builder.add_edge("generate_response", END)
```

---

### 3. **RAGNode** (`rag_node.py`)
**Purpose**: Orchestrates RAG workflow through three sequential nodes

**Responsibilities**:
- Coordinate document processing
- Manage context retrieval
- Generate LLM responses
- Handle state transitions
- Error handling and logging

**Node Methods**:

#### **Node 1: `process_documents(state)`**
- Loads documents from uploaded files
- Splits documents into chunks
- Creates/loads vectorstore
- Updates state with processing status

**Input State**:
```python
{
    "messages": [HumanMessage(content="user_query")],
    "uploaded_files": [FileUpload objects]
}
```

**Output State**:
```python
{
    "messages": [...],
    "uploaded_files": [...],
    "documents_processed": True,
    "num_chunks": 150,
    "vectorstore_source": "created_new" | "loaded_from_disk"
}
```

#### **Node 2: `retrieve_context(state)`**
- Extracts user query from messages
- Performs similarity search on vectorstore
- Formats retrieved context
- Updates state with context

**Input State**:
```python
{
    "messages": [HumanMessage(content="user_query")],
    "documents_processed": True,
    ...
}
```

**Output State**:
```python
{
    "messages": [...],
    "query": "user_query_string",
    "retrieved_context": "formatted_context_string",
    "retrieved_docs_content": ["doc1_content", "doc2_content", ...]
}
```

#### **Node 3: `generate_response(state)`**
- Creates prompt with context and query
- Invokes LLM to generate response
- Formats and returns response

**Input State**:
```python
{
    "query": "user_query_string",
    "retrieved_context": "formatted_context_string",
    "messages": [...]
}
```

**Output State**:
```python
{
    "messages": [AIMessage(content="generated_response")],
    "query": "...",
    "retrieved_context": "..."
}
```

---

### 4. **RAGModule** (`rag_module.py`)
**Purpose**: Core RAG functionality for document processing and retrieval

**Responsibilities**:
- Document loading (PDF/TXT)
- Text splitting into chunks
- Embedding generation (OpenAI)
- Vectorstore creation (FAISS)
- Document retrieval via similarity search
- Persistence management

**Key Methods**:

#### **`load_documents(uploaded_files)`**
- Accepts Streamlit file upload objects
- Saves files to temporary location
- Uses PyPDFLoader for PDFs, TextLoader for TXT
- Returns list of Document objects

**Process Flow**:
1. Iterate through uploaded files
2. Save each file to temporary location
3. Determine file type (PDF/TXT)
4. Instantiate appropriate loader
5. Call `loader.load()` to extract content
6. Validate extracted content
7. Clean up temporary files
8. Return all documents

#### **`split_documents(documents)`**
- Uses RecursiveCharacterTextSplitter
- Parameters: `chunk_size=1000`, `chunk_overlap=200`
- Splits documents while preserving context
- Returns list of Document chunks

#### **`create_vectorstore(chunks, file_names, save_to_disk)`**
- Generates embeddings for each chunk (OpenAI API calls)
- Creates FAISS vectorstore from chunks and embeddings
- Saves vectorstore to disk if requested
- Saves metadata (file names, chunk count, timestamp)

**Process Flow**:
1. Validate chunks are not empty
2. Call `FAISS.from_documents(chunks, embeddings)`
   - Internally calls OpenAI API for each chunk
   - Creates vector embeddings
   - Builds FAISS index
3. Save to disk at path: `vectorstore_db/vectorstore_{file_hash}/`
4. Save metadata JSON file
5. Return FAISS vectorstore object

#### **`load_vectorstore(file_names, persist_directory)`**
- Checks if vectorstore exists on disk
- Loads FAISS index files (`index.faiss`, `index.pkl`)
- Loads metadata JSON
- Returns vectorstore object or None

**Path Generation**:
```python
file_hash = md5(sorted(file_names).join(","))[:8]
vectorstore_path = f"vectorstore_db/vectorstore_{file_hash}"
```

#### **`find_or_create_vectorstore(file_names, chunks)`**
- First attempts to load existing vectorstore
- If not found, creates new one
- Returns vectorstore object

#### **`retrieve_documents(query, k=3)`**
- Performs similarity search on vectorstore
- Uses `similarity_search_with_score()` to get relevance scores
- Applies lenient threshold filtering
- Falls back to MMR search if no results
- Multiple fallback strategies for robust retrieval

**Retrieval Strategy**:
1. Primary: `similarity_search_with_score(query, k=max(k, 5))`
2. Score filtering: Accept scores ≤ 2.0 (lenient threshold)
3. Fallback 1: MMR search with `max_marginal_relevance_search()`
4. Fallback 2: Generic queries ("the", "a", "and")
5. Fallback 3: Increase k to 20
6. Final: Return empty list if all fail

---

### 5. **State** (`state.py`)
**Purpose**: Defines the shared state structure for LangGraph

**Type Definition**:
```python
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # Required
    uploaded_files: List                     # Optional
    query: str                               # Optional
    retrieved_context: str                   # Optional
    retrieved_docs: List                     # Optional
    retrieved_docs_content: List            # Optional
    documents_processed: bool                # Optional
    num_chunks: int                         # Optional
    vectorstore_source: str                 # Optional
    error: str                              # Optional
    error_traceback: str                    # Optional
```

**State Evolution Through Nodes**:

**Initial State** (from UI):
```python
{
    "messages": [HumanMessage(content="What is Amazon's revenue?")],
    "uploaded_files": [FileUpload("AMZN-Q3-2025.pdf")]
}
```

**After process_documents**:
```python
{
    "messages": [HumanMessage(...)],
    "uploaded_files": [...],
    "documents_processed": True,
    "num_chunks": 150,
    "vectorstore_source": "created_new"
}
```

**After retrieve_context**:
```python
{
    "messages": [HumanMessage(...)],
    "uploaded_files": [...],
    "documents_processed": True,
    "num_chunks": 150,
    "query": "What is Amazon's revenue?",
    "retrieved_context": "Amazon Q3 2025 Revenue: $143.1 billion...",
    "retrieved_docs_content": ["Amazon Q3 2025...", "Revenue breakdown..."]
}
```

**After generate_response** (Final):
```python
{
    "messages": [AIMessage(content="Based on the document, Amazon's Q3 2025 revenue was $143.1 billion...")],
    "uploaded_files": [...],
    "documents_processed": True,
    "num_chunks": 150,
    "query": "What is Amazon's revenue?",
    "retrieved_context": "..."
}
```

---

### 6. **DisplayResultStreamlit** (`display_result.py`)
**Purpose**: Renders graph execution results in Streamlit UI

**Responsibilities**:
- Display user messages
- Stream graph execution events
- Show progress for each step
- Display final responses
- Handle errors with user-friendly messages

**RAG Execution Display**:
```python
for event in graph.stream(initial_state):
    for node_name, node_output in event.items():
        if node_name == "process_documents":
            # Show processing status
        elif node_name == "retrieve_context":
            # Show retrieval status
        elif node_name == "generate_response":
            # Show final response
```

---

## 🔄 Execution Flow Diagram

### Complete RAG Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                              │
│  1. Select "RAG Chatbot" use case                                   │
│  2. Enter OpenAI API key                                            │
│  3. Upload PDF/TXT files                                            │
│  4. Enter query message                                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    main.py: load_langgraph_agenticai_app()          │
│                                                                      │
│  1. LoadStreamlitUI.load_streamlit_ui()                             │
│     → Collects: usecase, openai_api_key, uploaded_files,           │
│        user_message                                                  │
│                                                                      │
│  2. GroqLLM(user_input).get_llm_model()                             │
│     → Initializes Groq LLM model                                    │
│                                                                      │
│  3. GraphBuilder(model).setup_graph("RAG Chatbot", openai_api_key)  │
│     → Builds RAG graph with nodes and edges                        │
│                                                                      │
│  4. graph.compile()                                                 │
│     → Returns compiled LangGraph ready for execution                 │
│                                                                      │
│  5. DisplayResultStreamlit(...).display_result_on_ui()             │
│     → Executes graph and displays results                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              GraphBuilder.rag_build_graph(openai_api_key)           │
│                                                                      │
│  1. RAGNode(llm, openai_api_key)                                   │
│     → Initializes RAGNode with LLM and OpenAI API key             │
│     → Creates RAGModule instance internally                        │
│                                                                      │
│  2. Add Nodes:                                                       │
│     • process_documents                                              │
│     • retrieve_context                                               │
│     • generate_response                                              │
│                                                                      │
│  3. Add Edges:                                                       │
│     START → process_documents → retrieve_context →                  │
│     generate_response → END                                          │
│                                                                      │
│  4. Return compiled graph                                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│           DisplayResultStreamlit: graph.stream(initial_state)       │
│                                                                      │
│  initial_state = {                                                   │
│      "messages": [HumanMessage(content=user_message)],               │
│      "uploaded_files": [uploaded_files]                              │
│  }                                                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
        ┌───────────────────┐  ┌──────────────────┐
        │  NODE 1:         │  │  For each event  │
        │  process_documents│  │  from stream:    │
        └────────┬─────────┘  │  Update UI       │
                 │             │  Show progress    │
                 ▼             └──────────────────┘
        ┌───────────────────────────────┐
        │ RAGNode.process_documents()  │
        │                               │
        │ 1. Check if vectorstore       │
        │    exists in memory          │
        │    → Skip if exists           │
        │                               │
        │ 2. Check if no files uploaded│
        │    → Try load from disk       │
        │                               │
        │ 3. Load documents:            │
        │    RAGModule.load_documents()│
        │    • Save to temp file        │
        │    • PyPDFLoader/TextLoader   │
        │    • Extract content          │
        │                               │
        │ 4. Split documents:           │
        │    RAGModule.split_documents()│
        │    • RecursiveCharacterSplitter│
        │    • chunk_size=1000          │
        │    • chunk_overlap=200        │
        │                               │
        │ 5. Create/Find Vectorstore:  │
        │    RAGModule.find_or_create_ │
        │    vectorstore()              │
        │    • Check disk first         │
        │    • If not found:            │
        │      - Create embeddings      │
        │        (OpenAI API calls)     │
        │      - Build FAISS index      │
        │      - Save to disk           │
        │                               │
        │ 6. Update state:              │
        │    • documents_processed=True │
        │    • num_chunks=150           │
        │    • vectorstore_source=...   │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ State after Node 1:           │
        │ {                             │
        │   "messages": [...],          │
        │   "uploaded_files": [...],    │
        │   "documents_processed": True, │
        │   "num_chunks": 150,          │
        │   "vectorstore_source": "..." │
        │ }                             │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  NODE 2:                      │
        │  retrieve_context             │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ RAGNode.retrieve_context()    │
        │                               │
        │ 1. Check for errors from      │
        │    previous step              │
        │                               │
        │ 2. Validate vectorstore       │
        │    exists                     │
        │                               │
        │ 3. Extract query from         │
        │    state["messages"][0]       │
        │                               │
        │ 4. Retrieve documents:        │
        │    RAGModule.retrieve_         │
        │    documents(query, k=5)      │
        │    • similarity_search_        │
        │      with_score()             │
        │    • Score filtering          │
        │    • Fallback strategies      │
        │                               │
        │ 5. Format context:             │
        │    Join retrieved docs        │
        │    with "\n\n"                │
        │                               │
        │ 6. Update state:               │
        │    • query="user_query"       │
        │    • retrieved_context="..."  │
        │    • retrieved_docs_content=[...]│
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ State after Node 2:           │
        │ {                             │
        │   "messages": [...],          │
        │   "documents_processed": True,│
        │   "query": "What is Amazon's  │
        │             revenue?",        │
        │   "retrieved_context": "Amazon│
        │                Q3 2025 Revenue:│
        │                $143.1B...",   │
        │   "retrieved_docs_content": [...]│
        │ }                             │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  NODE 3:                      │
        │  generate_response            │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ RAGNode.generate_response()   │
        │                               │
        │ 1. Extract query and context  │
        │    from state                 │
        │                               │
        │ 2. Fallback if context empty: │
        │    Recreate from              │
        │    retrieved_docs_content     │
        │                               │
        │ 3. Create prompt template:    │
        │    ChatPromptTemplate         │
        │    • System message            │
        │    • User message with context │
        │                               │
        │ 4. Invoke LLM:                 │
        │    llm.invoke(formatted_prompt)│
        │    • Groq API call             │
        │    • Generate response         │
        │                               │
        │ 5. Update state:               │
        │    state["messages"] =        │
        │      [AIMessage(response)]     │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ Final State:                  │
        │ {                             │
        │   "messages": [AIMessage(     │
        │     "Based on the document,   │
        │      Amazon's Q3 2025         │
        │      revenue was $143.1B...")],│
        │   "query": "...",             │
        │   "retrieved_context": "...", │
        │   ...                          │
        │ }                             │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ DisplayResultStreamlit:        │
        │ Display final response         │
        │ in Streamlit chat UI           │
        └───────────────────────────────┘
```

---

## 📊 State Management

### State Initialization

**Source**: `main.py` → `DisplayResultStreamlit`

```python
initial_state = {
    "messages": [HumanMessage(content=user_message)],
    "uploaded_files": uploaded_files if uploaded_files else []
}
```

### State Propagation Through Nodes

LangGraph automatically propagates state between nodes. Each node:
1. Receives the complete state dictionary
2. Modifies/adds fields as needed
3. Returns the modified state
4. LangGraph merges returned state with existing state

**Key Mechanism**: `TypedDict` with `total=False` allows optional fields:
- Required: `messages` (with `add_messages` reducer)
- Optional: All other RAG-specific fields

**State Reducers**:
- `messages`: Uses `add_messages` reducer to append messages
- Other fields: Simple assignment/update

### State Updates in Each Node

**Node 1 - process_documents**:
```python
# Reads:
state.get('uploaded_files', [])

# Writes:
state['documents_processed'] = True
state['num_chunks'] = len(chunks)
state['vectorstore_source'] = "created_new" | "loaded_from_disk"
state['error'] = "..."  # if error occurs
```

**Node 2 - retrieve_context**:
```python
# Reads:
state['messages'][0].content  # user query
state.get('documents_processed')

# Writes:
state['query'] = user_query
state['retrieved_context'] = formatted_context
state['retrieved_docs_content'] = [doc1_content, doc2_content, ...]
state['error'] = "..."  # if error occurs
```

**Node 3 - generate_response**:
```python
# Reads:
state.get('query', '')
state.get('retrieved_context', '')
state.get('retrieved_docs_content', [])

# Writes:
state['messages'] = [AIMessage(content=generated_response)]
state['error'] = "..."  # if error occurs
```

---

## 🌊 Data Flow Architecture

### Document Processing Flow

```
PDF/TXT Files (Uploaded)
         │
         ▼
Streamlit FileUpload Objects
         │
         ▼
RAGNode.process_documents()
         │
         ▼
RAGModule.load_documents()
         │
         ├─► Save to temp file
         ├─► PyPDFLoader/TextLoader.load()
         └─► Extract text content
         │
         ▼
Document Objects (List)
         │
         ▼
RAGModule.split_documents()
         │
         ├─► RecursiveCharacterTextSplitter
         ├─► chunk_size=1000, overlap=200
         └─► Split into chunks
         │
         ▼
Document Chunks (List)
         │
         ▼
RAGModule.create_vectorstore()
         │
         ├─► For each chunk:
         │   └─► OpenAIEmbeddings.embed_query()
         │       └─► OpenAI API Call
         │           └─► Returns embedding vector (1536 dims)
         │
         ▼
FAISS.from_documents(chunks, embeddings)
         │
         ├─► Build FAISS index
         ├─► Store embeddings + metadata
         └─► Return FAISS vectorstore
         │
         ▼
vectorstore.save_local(path)
         │
         ├─► Save index.faiss
         ├─► Save index.pkl
         └─► Save metadata.json
```

### Query Processing Flow

```
User Query String
         │
         ▼
State["messages"][0].content
         │
         ▼
RAGNode.retrieve_context()
         │
         ▼
RAGModule.retrieve_documents(query, k=5)
         │
         ├─► OpenAIEmbeddings.embed_query(query)
         │   └─► OpenAI API Call
         │       └─► Query embedding vector (1536 dims)
         │
         ▼
vectorstore.similarity_search_with_score(query_embedding, k=5)
         │
         ├─► FAISS computes L2 distances
         ├─► Returns top k documents with scores
         └─► Filters by threshold (score <= 2.0)
         │
         ▼
Document Chunks (List) + Scores
         │
         ▼
Format Context
         │
         ├─► Join page_content with "\n\n"
         └─► Create context string
         │
         ▼
State["retrieved_context"]
```

### Response Generation Flow

```
State["query"] + State["retrieved_context"]
         │
         ▼
RAGNode.generate_response()
         │
         ▼
ChatPromptTemplate.format()
         │
         ├─► System: "You are a helpful assistant..."
         └─► User: "Context: {context}\n\nQuestion: {query}"
         │
         ▼
Formatted Prompt String
         │
         ▼
Groq LLM.invoke(formatted_prompt)
         │
         ├─► Groq API Call
         ├─► LLM processes prompt
         └─► Generates response
         │
         ▼
AIMessage(content=response)
         │
         ▼
State["messages"] = [AIMessage(...)]
         │
         ▼
DisplayResultStreamlit
         │
         └─► Display in Streamlit chat UI
```

---

## 🔍 Step-by-Step Execution

### Step 1: User Input Collection

**Location**: `main.py` → `load_langgraph_agenticai_app()`

1. **UI Loading**:
   ```python
   ui = LoadStreamlitUI()
   user_input = ui.load_streamlit_ui()
   ```
   - Renders Streamlit UI
   - Collects use case selection
   - Collects OpenAI API key (for RAG)
   - Handles file uploads

2. **User Message Input**:
   ```python
   user_message = st.chat_input("Enter your message:")
   ```

3. **LLM Initialization**:
   ```python
   obj_llm_config = GroqLLM(user_controls_input=user_input)
   model = obj_llm_config.get_llm_model()
   ```

---

### Step 2: Graph Building

**Location**: `graph_builder.py` → `setup_graph()`

1. **GraphBuilder Initialization**:
   ```python
   graph_builder = GraphBuilder(model)
   ```

2. **Graph Setup**:
   ```python
   graph = graph_builder.setup_graph("RAG Chatbot", openai_api_key=openai_api_key)
   ```
   
   **Internal Flow**:
   - Calls `rag_build_graph(openai_api_key)`
   - Creates `RAGNode(llm, openai_api_key)`
   - Adds three nodes to graph
   - Sets edges between nodes
   - Compiles graph

---

### Step 3: Graph Execution

**Location**: `display_result.py` → `display_result_on_ui()`

1. **Initial State Preparation**:
   ```python
   initial_state = {
       "messages": [HumanMessage(content=user_message)],
       "uploaded_files": uploaded_files if uploaded_files else []
   }
   ```

2. **Streaming Execution**:
   ```python
   for event in graph.stream(initial_state):
       for node_name, node_output in event.items():
           # Handle each node's output
   ```

   **Execution Order**:
   - `process_documents` node executes first
   - State is updated and passed to `retrieve_context`
   - `retrieve_context` executes
   - State is updated and passed to `generate_response`
   - `generate_response` executes
   - Final state is returned

---

### Step 4: Document Processing (Node 1)

**Location**: `rag_node.py` → `process_documents()`

**Detailed Flow**:

1. **Check Existing Vectorstore**:
   ```python
   if self.vectorstore_created and self.rag_module.vectorstore is not None:
       # Skip processing, reuse existing vectorstore
       return state
   ```

2. **Handle No Files Uploaded**:
   ```python
   if not uploaded_files:
       # Try to load vectorstore from disk
       existing_vectorstore = self.rag_module.load_vectorstore(...)
   ```

3. **Load Documents**:
   ```python
   documents = self.rag_module.load_documents(uploaded_files)
   ```
   - Saves files to temp locations
   - Uses PyPDFLoader/TextLoader
   - Extracts text content
   - Validates content extraction

4. **Split Documents**:
   ```python
   chunks = self.rag_module.split_documents(documents)
   ```
   - Uses RecursiveCharacterTextSplitter
   - Creates chunks with overlap

5. **Create/Load Vectorstore**:
   ```python
   self.rag_module.find_or_create_vectorstore(file_names, chunks)
   ```
   - Checks for existing vectorstore on disk
   - If not found:
     - Generates embeddings (OpenAI API calls)
     - Creates FAISS vectorstore
     - Saves to disk

6. **Update State**:
   ```python
   state['documents_processed'] = True
   state['num_chunks'] = len(chunks)
   state['vectorstore_source'] = "created_new" | "loaded_from_disk"
   ```

---

### Step 5: Context Retrieval (Node 2)

**Location**: `rag_node.py` → `retrieve_context()`

**Detailed Flow**:

1. **Error Check**:
   ```python
   if 'error' in state:
       return state  # Don't proceed if previous step failed
   ```

2. **Vectorstore Validation**:
   ```python
   if self.rag_module.vectorstore is None:
       state['error'] = "Vector store not initialized"
       return state
   ```

3. **Extract Query**:
   ```python
   user_query = state['messages'][0].content
   ```

4. **Retrieve Documents**:
   ```python
   retrieved_docs = self.rag_module.retrieve_documents(user_query, k=5)
   ```
   
   **Internal Retrieval Process**:
   - Embeds query using OpenAI API
   - Performs similarity search with scores
   - Filters by threshold
   - Falls back to MMR if needed
   - Multiple fallback strategies

5. **Format Context**:
   ```python
   context_parts = [doc.page_content for doc in retrieved_docs]
   context = "\n\n".join(context_parts)
   ```

6. **Update State**:
   ```python
   state['query'] = user_query
   state['retrieved_context'] = context
   state['retrieved_docs_content'] = [doc.page_content for doc in retrieved_docs]
   ```

---

### Step 6: Response Generation (Node 3)

**Location**: `rag_node.py` → `generate_response()`

**Detailed Flow**:

1. **Extract Query and Context**:
   ```python
   query = state.get('query', '')
   context = state.get('retrieved_context', '')
   ```

2. **Fallback for Missing Context**:
   ```python
   if not context and state.get('retrieved_docs_content'):
       context = "\n\n".join(state['retrieved_docs_content'])
   ```

3. **Create Prompt**:
   ```python
   prompt_template = ChatPromptTemplate.from_messages([
       ("system", "You are a helpful assistant..."),
       ("user", "Context:\n{context}\n\nQuestion: {query}")
   ])
   formatted_prompt = prompt_template.format(context=context, query=query)
   ```

4. **Invoke LLM**:
   ```python
   response = self.llm.invoke(formatted_prompt)
   ```
   - Makes Groq API call
   - LLM generates response based on context and query

5. **Update State**:
   ```python
   state['messages'] = [response]
   ```

6. **Return Final State**:
   - Contains AIMessage with generated response
   - All previous state fields preserved

---

### Step 7: Display Results

**Location**: `display_result.py` → `display_result_on_ui()`

1. **Extract Response**:
   ```python
   if final_result and final_result.get('messages'):
       response_message = final_result['messages'][0]
       # Display response_message.content
   ```

2. **Render in UI**:
   ```python
   with st.chat_message("assistant"):
       st.write(response_message.content)
   ```

---

## 💾 Persistence & Caching

### Vectorstore Persistence

**Storage Location**:
- Directory: `./vectorstore_db/`
- Path pattern: `vectorstore_db/vectorstore_{file_hash}/`
- File hash: MD5 hash of sorted file names (first 8 chars)

**Persisted Files**:
1. **`index.faiss`**: FAISS index file containing vector embeddings
2. **`index.pkl`**: Pickle file with document metadata and mappings
3. **`metadata.json`**: JSON file with:
   - `file_names`: List of original file names
   - `num_chunks`: Number of chunks in vectorstore
   - `created_at`: Timestamp of creation

**Loading Strategy**:
1. Check memory first (if vectorstore already loaded)
2. Check disk for existing vectorstore (by file names hash)
3. If not found, create new vectorstore and save to disk

**Benefits**:
- Avoid reprocessing same documents
- Faster subsequent queries
- Persistent across application restarts
- Reduces OpenAI API calls (embeddings only generated once)

### State Persistence

**In-Memory Only**:
- State is maintained during graph execution
- State flows through nodes sequentially
- State is not persisted to disk
- Each new query creates fresh initial state

**Session State** (Streamlit):
- Uploaded files: `st.session_state["uploaded_files"]`
- API keys: `st.session_state["OPENAI_API_KEY"]`, etc.
- Preserves user inputs across UI interactions

---

## ⚠️ Error Handling Flow

### Error Propagation

**Error Detection Points**:

1. **Document Loading Errors**:
   - Location: `RAGModule.load_documents()`
   - Errors: File reading failures, unsupported formats
   - Handling: Catches exceptions, logs error, returns empty list

2. **Vectorstore Creation Errors**:
   - Location: `RAGModule.create_vectorstore()`
   - Errors: OpenAI API failures, FAISS creation failures
   - Handling: Catches exceptions, logs traceback, raises error

3. **Retrieval Errors**:
   - Location: `RAGModule.retrieve_documents()`
   - Errors: Empty vectorstore, API failures
   - Handling: Multiple fallback strategies, returns empty list if all fail

4. **Response Generation Errors**:
   - Location: `RAGNode.generate_response()`
   - Errors: Missing context, LLM API failures
   - Handling: Catches exceptions, returns error message in state

### Error State Updates

**Each Node Updates State with Errors**:
```python
state['error'] = f"Error description: {str(e)}"
state['error_traceback'] = traceback.format_exc()
```

**Error Checking in Subsequent Nodes**:
```python
if 'error' in state:
    print(f"❌ Error from previous step: {state['error']}")
    return state  # Don't proceed
```

### UI Error Display

**Location**: `display_result.py`

```python
if node_output.get('error'):
    error_msg = node_output['error']
    status_placeholder.error(f"❌ Error: {error_msg}")
    st.error(f"**Error:** {error_msg}")
    
    # Show helpful suggestions
    if 'pypdf' in error_msg.lower():
        st.warning("💡 Install pypdf: `pip install pypdf`")
```

---

## 🔗 Component Interactions

### Interaction Diagram

```
┌─────────────────┐
│  LoadStreamlitUI │
│  • UI Setup      │
│  • Input Collect │
└────────┬─────────┘
         │
         ▼ user_input
┌─────────────────┐
│  main.py        │
│  • Orchestration│
└────────┬─────────┘
         │
         ├──────────────────────┐
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│  GroqLLM       │    │  GraphBuilder   │
│  • Model Init  │    │  • Graph Setup  │
└────────┬────────┘    └────────┬────────┘
         │                      │
         │ model                │ graph
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  DisplayResultStreamlit│
        │  • Execute Graph      │
        │  • Display Results    │
        └───────────┬───────────┘
                    │
                    ▼ graph.stream()
        ┌───────────────────────┐
        │  RAGNode              │
        │  • process_documents  │
        │  • retrieve_context   │
        │  • generate_response  │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  RAGModule            │
        │  • load_documents     │
        │  • split_documents    │
        │  • create_vectorstore │
        │  • retrieve_documents │
        └───────────┬───────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌───────────┐ ┌──────────┐ ┌──────────┐
│  OpenAI   │ │  Groq    │ │  FAISS   │
│  API      │ │  API     │ │  Library │
└───────────┘ └──────────┘ └──────────┘
```

### Method Call Chain

**Complete Execution Chain**:

```
main.load_langgraph_agenticai_app()
  └─► LoadStreamlitUI.load_streamlit_ui()
  └─► GroqLLM.get_llm_model()
  └─► GraphBuilder.setup_graph()
        └─► GraphBuilder.rag_build_graph()
              └─► RAGNode.__init__()
                    └─► RAGModule.__init__()
  └─► DisplayResultStreamlit.display_result_on_ui()
        └─► graph.stream(initial_state)
              ├─► RAGNode.process_documents()
              │     ├─► RAGModule.load_vectorstore()
              │     ├─► RAGModule.load_documents()
              │     ├─► RAGModule.split_documents()
              │     └─► RAGModule.find_or_create_vectorstore()
              │           ├─► RAGModule.create_vectorstore()
              │           │     └─► OpenAIEmbeddings.embed_documents()
              │           │           └─► OpenAI API Call (Multiple)
              │           └─► FAISS.from_documents()
              │
              ├─► RAGNode.retrieve_context()
              │     └─► RAGModule.retrieve_documents()
              │           ├─► OpenAIEmbeddings.embed_query()
              │           │     └─► OpenAI API Call
              │           └─► vectorstore.similarity_search_with_score()
              │
              └─► RAGNode.generate_response()
                    ├─► ChatPromptTemplate.format()
                    └─► llm.invoke()
                          └─► Groq API Call
```

---

## ⚡ Performance Considerations

### Optimization Strategies

1. **Vectorstore Caching**:
   - Saves embeddings to disk
   - Avoids reprocessing same documents
   - Reduces OpenAI API calls significantly

2. **Chunk Size Tuning**:
   - `chunk_size=1000`: Balance between context and granularity
   - `chunk_overlap=200`: Preserves context across boundaries

3. **Retrieval Optimization**:
   - `k=5`: Retrieves multiple relevant chunks
   - Score threshold: Filters low-relevance results
   - Multiple fallback strategies: Ensures results even with poor matches

4. **Parallel Processing**:
   - Embeddings generated in parallel by FAISS
   - OpenAI API handles batch processing internally

### Bottlenecks

1. **Embedding Generation**:
   - Each chunk requires OpenAI API call
   - Time: ~0.5-1 second per chunk
   - Solution: Persist vectorstore to avoid regeneration

2. **Document Loading**:
   - PDF parsing can be slow for large files
   - Solution: Async loading (future enhancement)

3. **Similarity Search**:
   - FAISS search is fast but scales with vectorstore size
   - Solution: Index optimization, approximate search

### Scalability

**Current Limitations**:
- Single-threaded execution
- In-memory vectorstore
- Sequential node execution

**Future Enhancements**:
- Async document processing
- Distributed vectorstores
- Parallel node execution
- Caching of query embeddings

---

## 📝 Summary

This architecture document provides a comprehensive overview of:

1. **System Architecture**: Multi-layered design from UI to persistence
2. **Component Details**: Each component's purpose and responsibilities
3. **Execution Flow**: Step-by-step execution through nodes
4. **State Management**: How state flows and evolves through nodes
5. **Data Flow**: Document and query processing pipelines
6. **Persistence**: Vectorstore and state caching strategies
7. **Error Handling**: Comprehensive error propagation and handling
8. **Component Interactions**: How components communicate
9. **Performance**: Optimization strategies and bottlenecks

This architecture enables:
- **Modularity**: Clear separation of concerns
- **Maintainability**: Well-defined interfaces
- **Scalability**: Foundation for future enhancements
- **Reliability**: Robust error handling and fallbacks
- **User Experience**: Real-time progress updates and error messages

---

**End of Architecture Documentation**

