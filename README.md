# 🤖 LangGraph Agentic AI Assistant - Finance

> **A comprehensive educational project demonstrating stateful Agentic AI applications using LangGraph, featuring multiple use cases including RAG (Retrieval-Augmented Generation) for document-based question answering.**

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Components & Structure](#-components--structure)
4. [Use Cases](#-use-cases)
5. [Technical Stack](#-technical-stack)
6. [Setup Instructions](#-setup-instructions)
7. [Usage Guide](#-usage-guide)
8. [Code Flow & Execution](#-code-flow--execution)
9. [Key Concepts for Students](#-key-concepts-for-students)

---

## 🎯 Project Overview

This project is a **stateful Agentic AI system** built using **LangGraph**, demonstrating how to create multi-step AI workflows with persistent state management. The system supports three distinct use cases:

1. **Basic Chatbot** - Simple conversational AI
2. **Chatbot with Tool** - AI agent with web search capabilities
3. **RAG Chatbot** - Document-based question answering with Retrieval-Augmented Generation

### Key Features

- ✅ **Stateful Workflows** - Maintains conversation context across interactions
- ✅ **Multi-Node Graphs** - Complex workflows with conditional routing
- ✅ **Tool Integration** - External API integration (Tavily Search)
- ✅ **RAG Implementation** - Document processing, embedding, and semantic search
- ✅ **Persistent Vectorstore** - FAISS-based document storage with disk persistence
- ✅ **Streamlit UI** - Interactive web interface for all use cases

---

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI Layer                        │
│  (LoadStreamlitUI, DisplayResultStreamlit)                      │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                           │
│  (main.py - load_langgraph_agenticai_app)                        │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Graph Builder Layer                        │
│  (GraphBuilder - setup_graph, build methods)                     │
└────────────────────────────┬────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
      │ Basic Chat  │ │ Chat + Tool │ │  RAG Chat   │
      │   Graph     │ │    Graph    │ │    Graph    │
      └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
             │               │               │
             ▼               ▼               ▼
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
      │   Nodes     │ │ Nodes+Tools  │ │ RAG Modules │
      └─────────────┘ └─────────────┘ └─────────────┘
```

### Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         LangGraph Core                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   State      │  │    Nodes     │  │    Edges     │       │
│  │  (TypedDict) │  │  (Functions) │  │ (Connections)│       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      State Management                          │
│  • Messages (conversation history)                           │
│  • Uploaded files (RAG)                                      │
│  • Retrieved context (RAG)                                   │
│  • Error states                                              │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    External Integrations                      │
│  • Groq LLM (language model)                                │
│  • OpenAI Embeddings (for RAG)                               │
│  • Tavily Search (web search tool)                           │
│  • FAISS (vector database)                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🧩 Components & Structure

### Project Directory Structure

```
Rag_Agentic_AI_AssistantFinance/
├── app.py                          # Entry point (Streamlit app)
├── main.py                         # (deprecated - use app.py)
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration
├── README.md                       # This file
│
├── src/
│   └── langgraphagenticai/
│       ├── __init__.py
│       ├── main.py                 # Main application logic
│       │
│       ├── graph/                   # Graph construction
│       │   ├── __init__.py
│       │   └── graph_builder.py    # GraphBuilder class
│       │
│       ├── state/                   # State management
│       │   ├── __init__.py
│       │   └── state.py            # State TypedDict definition
│       │
│       ├── nodes/                   # Graph node implementations
│       │   ├── __init__.py
│       │   ├── basic_chatbot_node.py
│       │   ├── chatbot_with_Tool_node.py
│       │   └── rag_node.py         # RAG pipeline nodes
│       │
│       ├── LLMS/                    # LLM configuration
│       │   ├── __init__.py
│       │   └── groqllm.py          # Groq LLM setup
│       │
│       ├── tools/                   # External tools
│       │   ├── __init__.py
│       │   └── serach_tool.py      # Tavily search tool
│       │
│       ├── RAG/                     # RAG implementation
│       │   ├── __init__.py
│       │   └── rag_module.py       # Document processing, embeddings, vectorstore
│       │
│       ├── ui/                      # Streamlit UI
│       │   ├── __init__.py
│       │   ├── uiconfigfile.py     # Configuration loader
│       │   ├── uiconfigfile.ini     # UI configuration
│       │   └── streamlitui/
│       │       ├── loadui.py       # UI input handling
│       │       └── display_result.py # Result display
│       │
│       └── inputrag/                # Sample documents
│           └── AMZN-Q3-2025-Earnings-Release.pdf
│
└── vectorstore_db/                  # Persistent FAISS vectorstore (generated)
    └── vectorstore_<hash>/
        ├── index.faiss
        ├── index.pkl
        └── metadata.json
```

### Core Components

#### 1. **State (`state/state.py`)**
- Defines the shared state structure using `TypedDict`
- Fields:
  - `messages`: Conversation history (LangChain messages)
  - `uploaded_files`: Document files (for RAG)
  - `query`: User query (for RAG)
  - `retrieved_context`: Retrieved document chunks (for RAG)
  - `documents_processed`: Processing status flag
  - `error`: Error messages

#### 2. **Graph Builder (`graph/graph_builder.py`)**
- **Purpose**: Constructs LangGraph graphs based on use case
- **Methods**:
  - `basic_chatbot_build_graph()`: Creates simple chatbot graph
  - `chatbot_with_tools_build_graph()`: Creates tool-enabled chatbot
  - `rag_build_graph()`: Creates RAG pipeline graph
  - `setup_graph()`: Main entry point - selects and builds graph

#### 3. **Nodes (`nodes/`)**
Each node is a function that processes state and returns updated state:

- **BasicChatbotNode**: Simple LLM response generation
- **ChatbotWithToolNode**: LLM with tool binding and conditional routing
- **RAGNode**: Three-node pipeline:
  - `process_documents`: Loads and processes PDFs/TXT files
  - `retrieve_context`: Semantic search in vectorstore
  - `generate_response`: LLM response with retrieved context

#### 4. **RAG Module (`RAG/rag_module.py`)**
- **Document Loading**: PDF and TXT file parsing
- **Text Splitting**: Chunking documents for embedding
- **Embedding Generation**: OpenAI embeddings for chunks
- **Vectorstore Management**: FAISS creation, persistence, and retrieval
- **Similarity Search**: Semantic search with multiple fallback strategies

#### 5. **UI Components (`ui/streamlitui/`)**
- **LoadStreamlitUI**: Handles user input (API keys, model selection, file uploads)
- **DisplayResultStreamlit**: Displays results for each use case

---

## 🎬 Use Cases

### Use Case 1: Basic Chatbot

**Description**: Simple conversational AI without external tools.

**Graph Structure**:
```
START → [chatbot] → END
```

**Flow**:
1. User sends message via Streamlit UI
2. Message added to state
3. `basic_chatbot_node.process()` invoked
4. LLM generates response
5. Response displayed in UI

**Code Flow**:
```python
State['messages'] → BasicChatbotNode → LLM.invoke() → Updated State['messages']
```

**Key Features**:
- Single node workflow
- Direct LLM interaction
- State preservation across conversations

---

### Use Case 2: Chatbot with Tool

**Description**: AI agent with web search capabilities using Tavily API.

**Graph Structure**:
```
START → [chatbot] → [conditional_edge] → [tools] → [chatbot] → END
                           ↓
                      END (if no tool call)
```

**Flow**:
1. User asks question requiring real-time information
2. LLM decides if tool use is needed
3. **Conditional Routing**: 
   - If tool needed → route to `tools` node
   - If no tool needed → route to END
4. Tool execution (Tavily search)
5. Tool results fed back to chatbot
6. LLM generates final response with tool results

**Code Flow**:
```python
State['messages'] → ChatbotNode (with tools) → 
    [Tool call detected?] → 
        YES → ToolNode → State['messages'] + ToolMessage → ChatbotNode
        NO → END
```

**Key Features**:
- **Tool Binding**: `llm.bind_tools(tools)` - LLM can call tools
- **Conditional Edges**: `tools_condition` - routes based on tool calls
- **Tool Execution**: `ToolNode` executes Tavily search
- **Multi-turn**: Tool results fed back for final response

---

### Use Case 3: RAG Chatbot

**Description**: Document-based question answering using Retrieval-Augmented Generation.

**Graph Structure**:
```
START → [process_documents] → [retrieve_context] → [generate_response] → END
```

#### Node 1: Process Documents

**Purpose**: Load, split, and embed documents into vectorstore.

**Process**:
1. Check if vectorstore exists (memory or disk)
2. If not exists:
   - Load documents (PDF/TXT)
   - Split into chunks (1000 chars, 200 overlap)
   - Generate embeddings (OpenAI)
   - Create FAISS vectorstore
   - Save to disk (`vectorstore_db/`)
3. Store metadata (file names, chunk count)

**Code Flow**:
```python
uploaded_files → RAGModule.load_documents() → 
    text_splitter.split_documents() → 
        embeddings.embed_documents() → 
            FAISS.from_documents() → 
                vectorstore.save_local()
```

#### Node 2: Retrieve Context

**Purpose**: Semantic search for relevant document chunks.

**Process**:
1. Extract user query from messages
2. Perform similarity search in vectorstore
3. Fallback strategies if no results:
   - MMR (Maximum Marginal Relevance)
   - Increased `k` (number of results)
   - Generic query variations
4. Format retrieved chunks as context

**Code Flow**:
```python
query → vectorstore.similarity_search_with_score(query, k=5) → 
    filter_by_score → format_context()
```

#### Node 3: Generate Response

**Purpose**: LLM generates answer using retrieved context.

**Process**:
1. Build prompt with:
   - Retrieved context (document chunks)
   - User query
   - Instructions
2. LLM generates response
3. Update state with response

**Code Flow**:
```python
context + query → PromptTemplate → LLM.invoke() → response
```

**Key Features**:
- **Document Persistence**: Vectorstore saved to disk
- **Smart Retrieval**: Multiple retrieval strategies
- **Context Injection**: Relevant chunks added to prompt
- **State Tracking**: Processing status at each step

---

## 🛠️ Technical Stack

### Core Frameworks

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Graph Engine** | LangGraph | Stateful workflow orchestration |
| **LLM Framework** | LangChain | LLM integration and message handling |
| **LLM Provider** | Groq | Fast inference language models |
| **UI Framework** | Streamlit | Interactive web interface |
| **Embeddings** | OpenAI | Text embeddings for RAG |
| **Vector Database** | FAISS | Efficient similarity search |
| **Search Tool** | Tavily | Real-time web search API |
| **Document Parsing** | PyPDF | PDF text extraction |

### Dependencies

```txt
langchain          # Core LangChain functionality
langgraph          # Graph-based workflows
langchain_community # Community integrations
langchain_core     # Core abstractions
langchain_groq     # Groq LLM integration
langchain_openai   # OpenAI embeddings
faiss-cpu          # Vector similarity search
streamlit          # Web UI framework
tavily-python      # Web search API
pypdf              # PDF document parsing
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10 or higher
- pip or uv (package manager)
- API Keys:
  - **Groq API Key** ([Get from Groq Console](https://console.groq.com/keys))
  - **Tavily API Key** ([Get from Tavily](https://app.tavily.com/home)) - for Chatbot with Tool
  - **OpenAI API Key** ([Get from OpenAI](https://platform.openai.com/api-keys)) - for RAG Chatbot

### Step 1: Clone or Navigate to Project

```bash
cd Rag_Agentic_AI_AssistantFinance
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Or using uv (faster)
uv venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "from src.langgraphagenticai.main import load_langgraph_agenticai_app; print('✅ Setup successful!')"
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Using Basic Chatbot

1. **Select Use Case**: Choose "Basic Chatbot" from sidebar
2. **Configure LLM**:
   - Select model (e.g., `llama3-70b-8192`)
   - Enter your Groq API Key
3. **Start Chatting**: Type your message in the chat input
4. **View Response**: AI response appears in the chat

**Example Conversation**:
```
User: "What is machine learning?"
AI: "Machine learning is a subset of artificial intelligence..."
```

---

### Using Chatbot with Tool

1. **Select Use Case**: Choose "Chatbot with Tool"
2. **Configure**:
   - Select Groq model
   - Enter Groq API Key
   - Enter Tavily API Key
3. **Ask Questions**: Ask about current events or real-time information

**Example**:
```
User: "What's the latest news about AI?"
AI: [Tool Call] → Searches web → "According to recent news..."
```

**What Happens**:
- LLM detects need for current information
- Automatically calls Tavily search tool
- Tool results integrated into response

---

### Using RAG Chatbot

1. **Select Use Case**: Choose "RAG Chatbot"
2. **Configure**:
   - Select Groq model
   - Enter Groq API Key
   - Enter OpenAI API Key (for embeddings)
3. **Upload Documents**:
   - Click "Upload PDF or TXT files"
   - Select one or more documents
   - Wait for processing (first time only)
4. **Ask Questions**: Ask questions about uploaded documents

**Example**:
```
User uploads: "AMZN-Q3-2025-Earnings-Release.pdf"
User: "What was Amazon's revenue in Q3 2025?"
AI: "Based on the earnings report, Amazon's revenue in Q3 2025 was..."
```

**Behind the Scenes**:
1. **Step 1**: Documents processed, split into chunks, embedded
2. **Step 2**: Query matched against document chunks
3. **Step 3**: Relevant chunks retrieved and added to context
4. **Step 4**: LLM generates answer using context

**Persistence**:
- Vectorstore saved to `vectorstore_db/`
- Subsequent queries use cached vectorstore (faster)
- No reprocessing needed

---

## 🔄 Code Flow & Execution

### Application Startup Flow

```
1. app.py
   └──> load_langgraph_agenticai_app()
       ├──> LoadStreamlitUI.load_streamlit_ui()
       │   └──> Reads config (uiconfigfile.ini)
       │   └──> Displays sidebar (API keys, model selection)
       ├──> User selects use case and enters API keys
       ├──> User sends message
       ├──> GroqLLM.get_llm_model()
       │   └──> Creates ChatGroq instance
       ├──> GraphBuilder.setup_graph(usecase)
       │   └──> Builds appropriate graph:
       │       ├──> basic_chatbot_build_graph()
       │       ├──> chatbot_with_tools_build_graph()
       │       └──> rag_build_graph()
       ├──> graph.compile() → CompiledGraph
       └──> DisplayResultStreamlit.display_result_on_ui()
           └──> graph.stream(initial_state)
               └──> Nodes execute in sequence
```

### State Flow in LangGraph

```python
# Initial State
state = {
    "messages": [HumanMessage("What is AI?")],
    "uploaded_files": [file1.pdf, file2.txt],  # RAG only
}

# After Node 1 (process_documents) - RAG only
state = {
    "messages": [HumanMessage("What is AI?")],
    "uploaded_files": [file1.pdf, file2.txt],
    "documents_processed": True,
    "num_chunks": 150,
}

# After Node 2 (retrieve_context) - RAG only
state = {
    "messages": [HumanMessage("What is AI?")],
    "query": "What is AI?",
    "retrieved_context": "AI is... [from document chunk 1] ... [from chunk 2]",
    "retrieved_docs": [Document(...), Document(...)],
}

# After Node 3 (generate_response) - RAG only
state = {
    "messages": [
        HumanMessage("What is AI?"),
        AIMessage("Based on the documents, AI is...")
    ],
    # ... all previous fields
}
```

### Graph Execution Examples

#### Example 1: Basic Chatbot

```python
# 1. Graph created
graph = GraphBuilder(model).setup_graph("Basic Chatbot")

# 2. Initial state
state = {"messages": [HumanMessage("Hello!")]}

# 3. Graph execution
result = graph.invoke(state)
# Result: {"messages": [HumanMessage("Hello!"), AIMessage("Hi there!")]}
```

#### Example 2: Chatbot with Tool

```python
# 1. Graph created with tools
graph = GraphBuilder(model).setup_graph("Chatbot with Tool")

# 2. Initial state
state = {"messages": [HumanMessage("What's the weather today?")]}

# 3. Graph execution (streaming)
for event in graph.stream(state):
    # Event 1: {"chatbot": {"messages": [ToolMessage(...)]}}
    # Event 2: {"tools": {"messages": [ToolMessage(result="Sunny, 72°F")]}}
    # Event 3: {"chatbot": {"messages": [AIMessage("Today's weather is sunny...")]}}
```

#### Example 3: RAG Chatbot

```python
# 1. Graph created
graph = GraphBuilder(model).setup_graph("RAG Chatbot", openai_api_key="...")

# 2. Initial state
state = {
    "messages": [HumanMessage("What was the revenue?")],
    "uploaded_files": [pdf_file]
}

# 3. Graph execution (streaming)
for event in graph.stream(state):
    # Event 1: {"process_documents": {"documents_processed": True, "num_chunks": 150}}
    # Event 2: {"retrieve_context": {"retrieved_context": "...", "query": "..."}}
    # Event 3: {"generate_response": {"messages": [AIMessage("Revenue was...")]}}
```

---

## 🎓 Key Concepts for Students

### 1. **What is LangGraph?**

LangGraph is a framework for building **stateful, multi-agent applications** with LLMs. Think of it as a workflow engine where:
- **Nodes** = Processing steps (functions)
- **Edges** = Connections between steps
- **State** = Shared data passed between nodes

### 2. **Stateful vs Stateless**

- **Stateless**: Each request is independent (traditional APIs)
- **Stateful**: Context is maintained across interactions (conversations)

**Example**:
```python
# Stateless (traditional)
def chatbot(message):
    return llm.generate(message)  # No memory

# Stateful (LangGraph)
state = {"messages": [...]}  # Maintains conversation history
result = graph.invoke(state)  # Uses previous messages
```

### 3. **Graph Nodes and Edges**

**Node**: A function that processes state
```python
def my_node(state: dict) -> dict:
    # Process state
    state["new_field"] = "value"
    return state  # Updated state
```

**Edge**: Connection between nodes
```python
graph.add_edge("node1", "node2")  # node1 → node2
```

**Conditional Edge**: Route based on condition
```python
def route_decision(state):
    if needs_tool(state):
        return "tools"
    return "end"

graph.add_conditional_edges("chatbot", route_decision)
```

### 4. **RAG (Retrieval-Augmented Generation)**

**Problem**: LLMs have limited knowledge and can't access documents.

**Solution**: RAG combines:
- **Retrieval**: Find relevant document chunks
- **Augmentation**: Add chunks to LLM prompt
- **Generation**: LLM generates answer using chunks

**Workflow**:
```
Documents → Split → Embed → Store (Vector DB)
                            ↓
User Query → Embed → Search → Retrieve Chunks
                            ↓
                  LLM Prompt (Query + Chunks) → Answer
```

### 5. **Embeddings and Vector Search**

**Embedding**: Numerical representation of text
- Similar texts have similar embeddings
- Enables semantic search (not just keyword matching)

**Vector Search**:
```
Query: "What is revenue?"
Document chunks:
  - "Revenue was $100M" → [0.2, 0.5, ...] (embedding)
  - "Profit was $50M" → [0.1, 0.3, ...]
  - "Employees count: 1000" → [0.9, 0.1, ...]

Search: Find chunks with embeddings closest to query embedding
Result: "Revenue was $100M" (most similar)
```

### 6. **Tool Integration Pattern**

1. **Bind tools to LLM**: `llm.bind_tools([search_tool])`
2. **LLM decides**: Tool call detected in response
3. **Route to tool node**: Conditional edge routes to tool execution
4. **Execute tool**: Tool node runs external API call
5. **Return to LLM**: Tool results fed back to LLM for final response

### 7. **Persistence Patterns**

**Vectorstore Persistence**:
- Save to disk after creation
- Load from disk on subsequent queries
- Avoids reprocessing documents

**State Persistence** (not implemented, but concept):
- LangGraph state is ephemeral by default
- Can persist state to database for multi-session conversations

---

## 🔍 Troubleshooting

### Common Issues

#### 1. **Import Errors**
```bash
# Solution: Ensure you're in the project directory and virtual environment is activated
pip install -r requirements.txt
```

#### 2. **API Key Errors**
- Ensure API keys are entered correctly
- Check API key validity
- Verify API quotas/billing

#### 3. **PDF Not Loading (RAG)**
```bash
# Solution: Ensure pypdf is installed
pip install pypdf
```

#### 4. **Vectorstore Not Found**
- First query after upload: Vectorstore is created
- Subsequent queries: Loads from disk automatically
- If still issues: Check `vectorstore_db/` directory permissions

#### 5. **"No context found" in RAG**
- Ensure PDF has extractable text (not scanned image)
- Check that documents were uploaded before querying
- Verify OpenAI API key is valid

---

## 📝 Project Structure Summary

### Files You'll Modify Most

1. **`graph/graph_builder.py`**: Add new graph types
2. **`nodes/*.py`**: Implement new node logic
3. **`state/state.py`**: Add state fields
4. **`ui/streamlitui/loadui.py`**: Modify UI inputs
5. **`ui/uiconfigfile.ini`**: Change options/configuration

### Files to Understand

1. **`graph/graph_builder.py`**: How graphs are constructed
2. **`nodes/rag_node.py`**: RAG pipeline implementation
3. **`RAG/rag_module.py`**: Document processing logic
4. **`main.py`**: Application orchestration

---

## 🎯 Learning Outcomes

After studying this project, students should understand:

1. ✅ **LangGraph Fundamentals**: Building stateful AI workflows
2. ✅ **Node Architecture**: Creating reusable processing nodes
3. ✅ **Conditional Routing**: Dynamic workflow routing
4. ✅ **RAG Implementation**: Document processing and retrieval
5. ✅ **Vector Databases**: Embeddings and similarity search
6. ✅ **Tool Integration**: External API integration patterns
7. ✅ **State Management**: Persisting and passing context
8. ✅ **UI Integration**: Streamlit for AI applications

---

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Groq API Documentation](https://console.groq.com/docs)

---

## 👥 Author & License

This project is designed for educational purposes to teach students about stateful Agentic AI applications using LangGraph.

**License**: Apache 2.0

---

## 🙏 Acknowledgments

- LangChain team for LangGraph framework
- Groq for fast LLM inference
- OpenAI for embedding models
- Tavily for search API

---

**Happy Learning! 🚀**

For questions or issues, please refer to the code comments or documentation links above.
