# 🏗️ Architecture & Execution Flow: LangGraph Chatbot with Tools (Tavily)

> **Comprehensive documentation of the architecture and execution flow of the Chatbot with Tools (Tavily Search) built with LangGraph**

---

## 📑 Table of Contents

1. [System Architecture Overview](#-system-architecture-overview)
2. [Component Architecture](#-component-architecture)
3. [Execution Flow Diagram](#-execution-flow-diagram)
4. [State Management](#-state-management)
5. [Data Flow Architecture](#-data-flow-architecture)
6. [Step-by-Step Execution](#-step-by-step-execution)
7. [Conditional Routing](#-conditional-routing)
8. [Tool Integration](#-tool-integration)
9. [Component Interactions](#-component-interactions)
10. [Performance Considerations](#-performance-considerations)

---

## 🏗️ System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT UI LAYER                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │  User Interface  │  │  API Key Input    │  │  Message Display  │    │
│  │  • Chat Input    │  │  • Tavily API Key │  │  • User Messages  │    │
│  │  • Messages      │  │  • Groq Key       │  │  • AI Responses   │    │
│  │  • Status       │  │  • Validation     │  │  • Tool Results   │    │
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
│  │                  CHATBOT WITH TOOLS WORKFLOW GRAPH                    │ │
│  │                                                                       │ │
│  │   START                                                               │ │
│  │     │                                                                 │ │
│  │     ▼                                                                 │ │
│  │  ┌──────────────────────┐                                            │ │
│  │  │     chatbot          │                                            │ │
│  │  │  • LLM Processing    │                                            │ │
│  │  │  • Tool Detection    │                                            │ │
│  │  │  • Response Gen      │                                            │ │
│  │  └──────────┬───────────┘                                            │ │
│  │             │                                                         │ │
│  │             ▼                                                         │ │
│  │  ┌──────────────────────┐                                            │ │
│  │  │ tools_condition     │                                            │ │
│  │  │ • Check tool_calls  │                                            │ │
│  │  │ • Route Decision    │                                            │ │
│  │  └──────────┬───────────┘                                            │ │
│  │             │                                                         │ │
│  │    ┌────────┴────────┐                                              │ │
│  │    │                 │                                              │ │
│  │    ▼                 ▼                                              │ │
│  │  ┌──────┐        ┌──────────┐                                       │ │
│  │  │ END  │        │  tools   │                                       │ │
│  │  │(No   │        │  • Exec │                                       │ │
│  │  │tools)│        │  Tools   │                                       │ │
│  │  └──────┘        └────┬────┘                                       │ │
│  │                       │                                              │ │
│  │                       │ (Loop back)                                  │ │
│  │                       ▼                                              │ │
│  │              ┌──────────────────────┐                                │ │
│  │              │     chatbot          │◄───────┐                      │ │
│  │              │  (with tool results)  │         │                      │ │
│  │              └───────────────────────┘         │                      │ │
│  │                                                │                      │ │
│  │                (Loop until no tools)           │                      │ │
│  │                                                │                      │ │
│  │                ┌────────────────────────────────┘                    │ │
│  │                │                                                      │ │
│  │                ▼                                                      │ │
│  │              END                                                      │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CORE COMPONENTS LAYER                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │ ChatbotWithTool │  │   ToolNode      │  │   State          │    │
│  │ Node            │  │  • Prebuilt      │  │  • TypedDict     │    │
│  │  • bind_tools   │  │  • Exec Tools   │  │  • State Mgmt    │    │
│  │  • invoke       │  │  • Format Res   │  │  • Validation     │    │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘    │
└───────────┼──────────────────────┼──────────────────────┼──────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOOLS LAYER                                       │
│  ┌──────────────────┐  ┌──────────────────┐                              │
│  │  TavilySearch   │  │  Search Tool     │                              │
│  │  • Web Search   │  │  • max_results=2 │                              │
│  │  • API Calls    │  │  • Result Format │                              │
│  └────────┬────────┘  └────────┬────────┘                              │
└───────────┼──────────────────────┼──────────────────────────────────────┘
            │                      │
            ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL SERVICES LAYER                           │
│  ┌──────────────────┐  ┌──────────────────┐                              │
│  │  Tavily API      │  │   Groq API       │                              │
│  │  • Web Search    │  │  • LLM Models    │                              │
│  │  • Real-time Data│  │  • Text Gen      │                              │
│  │  • Rate Limits   │  │  • Streaming     │                              │
│  └──────────────────┘  └──────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Component Architecture

### 1. **LoadStreamlitUI** (`loadui.py`)
**Purpose**: Manages Streamlit UI initialization and user input collection

**Responsibilities**:
- Initialize Streamlit page configuration
- Render sidebar with LLM and use case selection
- Handle Tavily API key input
- Collect user messages
- Validate user inputs

**Key Methods**:
- `load_streamlit_ui()`: Main method to load and configure UI
- `initialize_session()`: Initialize Streamlit session state

**Inputs**:
- User selections (LLM, model, use case)
- Tavily API key
- User messages

**Outputs**:
- `user_controls` dictionary with all user inputs

---

### 2. **GraphBuilder** (`graph_builder.py`)
**Purpose**: Builds and configures LangGraph workflows

**Responsibilities**:
- Initialize StateGraph with State TypedDict
- Add nodes to the graph
- Define edges (direct and conditional) between nodes
- Compile the graph for execution

**Key Methods**:
- `__init__(model)`: Initialize with LLM model
- `chatbot_with_tools_build_graph()`: Build chatbot with tools workflow graph
- `setup_graph(usecase)`: Setup graph based on use case

**Graph Structure for Chatbot with Tools**:
```python
START → chatbot → tools_condition → (tools if tool_calls) → chatbot → ... → END
```

**Node Addition**:
```python
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", tool_node)
```

**Edge Configuration**:
```python
# Direct edge from START to chatbot
graph_builder.add_edge(START, "chatbot")

# Conditional edge from chatbot - routes based on tool calls
graph_builder.add_conditional_edges("chatbot", tools_condition)

# Direct edge from tools back to chatbot (loop)
graph_builder.add_edge("tools", "chatbot")
```

**Conditional Routing**:
- Uses LangGraph's prebuilt `tools_condition` function
- Automatically routes to "tools" if `tool_calls` detected
- Routes to END if no `tool_calls`

---

### 3. **ChatbotWithToolNode** (`chatbot_with_Tool_node.py`)
**Purpose**: Orchestrates chatbot with tool integration

**Responsibilities**:
- Create chatbot node with tools bound
- Process user messages
- Detect tool calls
- Generate responses with or without tools

**Key Methods**:

#### **`create_chatbot(tools)`**
- Binds tools to LLM using `bind_tools()`
- Returns chatbot node function
- Enables LLM to decide when to call tools

**Implementation**:
```python
def create_chatbot(self, tools):
    llm_with_tools = self.llm.bind_tools(tools)
    
    def chatbot_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    return chatbot_node
```

**What `bind_tools()` Does**:
1. Injects tool schemas into LLM prompt
2. Enables LLM to generate `tool_calls` in response
3. Formats tool calls as structured data
4. Allows LLM to decide when tools are needed

**Output Format**:
- If tools needed: `AIMessage` with `tool_calls` attribute
- If no tools needed: `AIMessage` with regular `content`

---

### 4. **Tool Definition** (`serach_tool.py`)
**Purpose**: Defines and provides tools for the chatbot

**Responsibilities**:
- Define Tavily search tool
- Configure tool parameters
- Create ToolNode for graph integration

**Key Functions**:

#### **`get_tools()`**
```python
def get_tools():
    tools = [TavilySearchResults(max_results=2)]
    return tools
```

**TavilySearchResults**:
- LangChain integration for Tavily search API
- Performs real-time web searches
- Returns structured search results
- `max_results=2`: Limits results to top 2

#### **`create_tool_node(tools)`**
```python
def create_tool_node(tools):
    return ToolNode(tools=tools)
```

**ToolNode** (LangGraph Prebuilt):
- Executes tool calls from LLM
- Calls each tool with provided arguments
- Formats results as ToolMessage
- Returns messages for next step

---

### 5. **tools_condition** (LangGraph Prebuilt)
**Purpose**: Conditional routing based on tool calls

**Location**: `langgraph.prebuilt.tools_condition`

**Functionality**:
```python
def tools_condition(state: MessagesState) -> str:
    """
    Routes based on presence of tool_calls in last message.
    
    Returns:
        "tools" if tool_calls exist
        "__end__" if no tool_calls
    """
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "__end__"
```

**Routing Logic**:
1. Checks last message in state
2. Looks for `tool_calls` attribute
3. If tool_calls exist: route to "tools" node
4. If no tool_calls: route to END

**Integration**:
```python
graph_builder.add_conditional_edges("chatbot", tools_condition)
```

---

### 6. **State** (`state.py`)
**Purpose**: Defines the shared state structure for LangGraph

**Type Definition**:
```python
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # Required
    # Messages can be: HumanMessage, AIMessage, ToolMessage
```

**State Evolution Through Execution**:

**Initial State**:
```python
{
    "messages": [HumanMessage(content="What's the weather in New York?")]
}
```

**After First Chatbot Node** (if tools needed):
```python
{
    "messages": [
        HumanMessage(content="What's the weather in New York?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "tavily_search_results_json",
                    "args": {"query": "weather New York"},
                    "id": "call_123"
                }
            ]
        )
    ]
}
```

**After Tools Node**:
```python
{
    "messages": [
        HumanMessage(content="What's the weather in New York?"),
        AIMessage(..., tool_calls=[...]),
        ToolMessage(
            content='[{"title": "Weather NYC", "url": "...", "content": "..."}]',
            tool_call_id="call_123"
        )
    ]
}
```

**After Second Chatbot Node** (final response):
```python
{
    "messages": [
        HumanMessage(content="What's the weather in New York?"),
        AIMessage(..., tool_calls=[...]),
        ToolMessage(...),
        AIMessage(content="Based on the search results, the weather in New York is...")
    ]
}
```

---

### 7. **DisplayResultStreamlit** (`display_result.py`)
**Purpose**: Renders graph execution results in Streamlit UI

**Responsibilities**:
- Display user messages
- Execute graph with invoke
- Display all messages (user, AI, tool)
- Format tool results appropriately

**Chatbot with Tools Display**:
```python
initial_state = {"messages": [user_message]}
res = graph.invoke(initial_state)

for message in res['messages']:
    if type(message) == HumanMessage:
        # Display user message
    elif type(message) == ToolMessage:
        # Display tool execution result
    elif type(message) == AIMessage and message.content:
        # Display AI response
```

---

## 🔄 Execution Flow Diagram

### Complete Chatbot with Tools Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                              │
│  1. Select "Chatbot with Tool" use case                              │
│  2. Enter Tavily API key                                            │
│  3. Enter query message                                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    main.py: load_langgraph_agenticai_app()          │
│                                                                      │
│  1. LoadStreamlitUI.load_streamlit_ui()                             │
│     → Collects: usecase, tavily_api_key, user_message               │
│                                                                      │
│  2. GroqLLM(user_input).get_llm_model()                             │
│     → Initializes Groq LLM model                                     │
│                                                                      │
│  3. GraphBuilder(model).setup_graph("Chatbot with Tool")            │
│     → Builds chatbot with tools graph                                │
│                                                                      │
│  4. graph.compile()                                                 │
│     → Returns compiled LangGraph ready for execution                 │
│                                                                      │
│  5. DisplayResultStreamlit(...).display_result_on_ui()              │
│     → Executes graph and displays results                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│      GraphBuilder.chatbot_with_tools_build_graph()                  │
│                                                                      │
│  1. get_tools()                                                     │
│     → Returns [TavilySearchResults(max_results=2)]                   │
│                                                                      │
│  2. create_tool_node(tools)                                         │
│     → Creates ToolNode(tools)                                       │
│                                                                      │
│  3. ChatbotWithToolNode(llm).create_chatbot(tools)                 │
│     → Binds tools to LLM                                            │
│     → Returns chatbot_node function                                  │
│                                                                      │
│  4. Add Nodes:                                                       │
│     • chatbot: chatbot_node                                         │
│     • tools: tool_node                                              │
│                                                                      │
│  5. Add Edges:                                                       │
│     • START → chatbot                                               │
│     • chatbot → tools_condition (conditional)                       │
│     • tools → chatbot (loop)                                        │
│                                                                      │
│  6. Return compiled graph                                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│     DisplayResultStreamlit: graph.invoke(initial_state)            │
│                                                                      │
│  initial_state = {                                                   │
│      "messages": [user_message]                                      │
│  }                                                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
        ┌───────────────────┐  ┌──────────────────┐
        │  NODE: chatbot     │  │  Initial State:  │
        │  (First Execution) │  │  {messages:      │
        └────────┬──────────┘  │   [user_msg]}    │
                 │             └──────────────────┘
                 ▼
        ┌───────────────────────────────┐
        │ ChatbotNode (First Pass)     │
        │                               │
        │ 1. llm_with_tools.invoke(    │
        │    state["messages"])         │
        │                               │
        │ 2. LLM analyzes query         │
        │    • "What's weather in NYC?" │
        │    • Needs current data       │
        │    • Decides to use tool      │
        │                               │
        │ 3. LLM generates response:   │
        │    AIMessage(                │
        │      content="",             │
        │      tool_calls=[{           │
        │        "name": "tavily_...",  │
        │        "args": {             │
        │          "query": "weather   │
        │                   New York"  │
        │        }                     │
        │      }]                      │
        │    )                         │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ State after chatbot:          │
        │ {                             │
        │   "messages": [               │
        │     HumanMessage(...),        │
        │     AIMessage(                │
        │       tool_calls=[...]       │
        │     )                         │
        │   ]                           │
        │ }                             │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Conditional Edge:             │
        │  tools_condition(state)        │
        └───────────────┬───────────────┘
                        │
                        ├────────────────────┐
                        │                    │
            tool_calls  │                    │  no tool_calls
            detected    │                    │
                        ▼                    ▼
        ┌───────────────────┐      ┌──────────────────┐
        │  Route to: tools │      │  Route to: END   │
        └───────────┬──────┘      │  (Execution ends)│
                    │             └──────────────────┘
                    ▼
        ┌───────────────────────────────┐
        │  NODE: tools                   │
        └───────────┬───────────────────┘
                    │
                    ▼
        ┌───────────────────────────────┐
        │ ToolNode Execution            │
        │                               │
        │ 1. Extract tool_calls from    │
        │    last AIMessage             │
        │                               │
        │ 2. For each tool_call:         │
        │    • Extract tool name        │
        │    • Extract arguments         │
        │    • Find corresponding tool   │
        │                               │
        │ 3. Execute TavilySearch:      │
        │    tavily_search.invoke({     │
        │      "query": "weather New York"
        │    })                          │
        │                               │
        │ 4. Tavily API Call:            │
        │    → HTTP POST to Tavily API  │
        │    → Returns search results    │
        │    → Format as JSON           │
        │                               │
        │ 5. Create ToolMessage:         │
        │    ToolMessage(               │
        │      content="[{...}]",       │
        │      tool_call_id="call_123"  │
        │    )                          │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ State after tools:            │
        │ {                             │
        │   "messages": [               │
        │     HumanMessage(...),        │
        │     AIMessage(tool_calls),   │
        │     ToolMessage(              │
        │       content="[{...results...}]"
        │     )                         │
        │   ]                           │
        │ }                             │
        └───────────────┬───────────────┘
                        │
                        ▼ (Loop back to chatbot)
        ┌───────────────────────────────┐
        │  NODE: chatbot                 │
        │  (Second Pass)                 │
        └───────────┬───────────────────┘
                    │
                    ▼
        ┌───────────────────────────────┐
        │ ChatbotNode (Second Pass)     │
        │                               │
        │ 1. llm_with_tools.invoke(     │
        │    state["messages"])         │
        │                               │
        │ 2. LLM receives:              │
        │    • Original user query      │
        │    • Tool call it made         │
        │    • Tool results              │
        │                               │
        │ 3. LLM processes:              │
        │    • Reads tool results        │
        │    • Synthesizes information  │
        │    • Generates final answer    │
        │                               │
        │ 4. LLM generates response:    │
        │    AIMessage(                 │
        │      content="Based on the    │
        │               search results, │
        │               the weather in  │
        │               New York is...", │
        │      tool_calls=[]  # Empty   │
        │    )                         │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ State after second chatbot:   │
        │ {                             │
        │   "messages": [               │
        │     HumanMessage(...),        │
        │     AIMessage(tool_calls),   │
        │     ToolMessage(...),        │
        │     AIMessage(               │
        │       content="Final answer..."│
        │     )                         │
        │   ]                           │
        │ }                             │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Conditional Edge:             │
        │  tools_condition(state)        │
        └───────────────┬───────────────┘
                        │
                        │ (no tool_calls)
                        ▼
        ┌───────────────────────────────┐
        │  Route to: END                 │
        │  (Execution complete)          │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ DisplayResultStreamlit:        │
        │ Display all messages in UI     │
        │  • User message                │
        │  • Tool execution result        │
        │  • Final AI response           │
        └───────────────────────────────┘
```

---

## 📊 State Management

### State Initialization

**Source**: `display_result.py` → `DisplayResultStreamlit`

```python
initial_state = {"messages": [user_message]}
```

### State Propagation Through Nodes

LangGraph automatically propagates state between nodes. Each node:
1. Receives the complete state dictionary
2. Modifies/adds messages as needed
3. Returns the modified state
4. LangGraph merges returned state with existing state

**Key Mechanism**: `messages` field uses `add_messages` reducer
- Appends new messages to existing messages
- Maintains conversation history
- Preserves message order

### State Updates in Each Node

**Chatbot Node**:
```python
# Reads:
state["messages"]  # All previous messages

# Writes:
state["messages"] = state["messages"] + [new_ai_message]
# Uses add_messages reducer to append
```

**Tools Node**:
```python
# Reads:
state["messages"][-1].tool_calls  # Tool calls from last message

# Writes:
state["messages"] = state["messages"] + [new_tool_message]
# Appends ToolMessage for each tool call execution
```

### Message Types in State

**HumanMessage**:
- User input
- Content: String

**AIMessage**:
- LLM responses
- Content: String (can be empty if tool_calls present)
- tool_calls: List of tool call objects (if tools needed)

**ToolMessage**:
- Tool execution results
- Content: String (tool result)
- tool_call_id: ID matching tool_call in AIMessage

---

## 🌊 Data Flow Architecture

### Query Processing Flow

```
User Query String
         │
         ▼
HumanMessage(content=query)
         │
         ▼
State["messages"] = [HumanMessage(...)]
         │
         ▼
ChatbotNode.invoke(state["messages"])
         │
         ├─► llm_with_tools.invoke()
         │   └─► Groq API Call
         │       └─► LLM analyzes query
         │       └─► Decides if tools needed
         │
         ├─► If tools needed:
         │   └─► Returns AIMessage(
         │       content="",
         │       tool_calls=[{...}]
         │   )
         │
         └─► If no tools needed:
             └─► Returns AIMessage(
                 content="Direct answer..."
             )
         │
         ▼
State["messages"] = [HumanMessage, AIMessage]
```

### Tool Execution Flow

```
AIMessage with tool_calls
         │
         ▼
tools_condition(state)
         │
         └─► Detects tool_calls
             └─► Routes to "tools" node
         │
         ▼
ToolNode(state)
         │
         ├─► Extract tool_calls from AIMessage
         │
         ├─► For each tool_call:
         │   ├─► Extract tool name
         │   ├─► Extract arguments
         │   └─► Find tool by name
         │
         ▼
TavilySearchResults.invoke(tool_args)
         │
         ├─► Extract query from args
         │
         ├─► Format API request
         │
         └─► Tavily API Call
             └─► HTTP POST to Tavily API
                 └─► Returns search results JSON
         │
         ▼
Format Results
         │
         ├─► Create ToolMessage
         ├─► Set content = JSON results
         └─► Set tool_call_id = matching ID
         │
         ▼
State["messages"] = [...previous, ToolMessage]
```

### Response Generation Flow (Second Pass)

```
State["messages"] = [
    HumanMessage,
    AIMessage(tool_calls),
    ToolMessage(results)
]
         │
         ▼
ChatbotNode.invoke(state["messages"])
         │
         └─► llm_with_tools.invoke()
             └─► Groq API Call
                 └─► LLM receives:
                     • Original query
                     • Tool call it made
                     • Tool results
                 └─► LLM synthesizes answer
                 └─► Generates final response
         │
         ▼
AIMessage(content="Final answer...")
         │
         ▼
State["messages"] = [...previous, AIMessage(final)]
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
   - Collects Tavily API key
   - Collects user message

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
   graph = graph_builder.setup_graph("Chatbot with Tool")
   ```
   
   **Internal Flow**:
   - Calls `chatbot_with_tools_build_graph()`
   - Gets tools: `get_tools()`
   - Creates tool node: `create_tool_node(tools)`
   - Creates chatbot node: `ChatbotWithToolNode.create_chatbot(tools)`
   - Adds nodes to graph
   - Sets edges (direct and conditional)
   - Compiles graph

---

### Step 3: Graph Execution

**Location**: `display_result.py` → `display_result_on_ui()`

1. **Initial State Preparation**:
   ```python
   initial_state = {"messages": [user_message]}
   ```

2. **Invoke Execution**:
   ```python
   res = graph.invoke(initial_state)
   ```
   
   **Execution Flow**:
   - Graph executes until completion
   - May loop between chatbot and tools
   - Terminates when no tool calls detected

---

### Step 4: First Chatbot Node Execution

**Location**: `chatbot_with_Tool_node.py` → `chatbot_node()`

**Detailed Flow**:

1. **Receive State**:
   ```python
   state["messages"] = [HumanMessage(content=user_query)]
   ```

2. **Invoke LLM with Tools**:
   ```python
   llm_with_tools.invoke(state["messages"])
   ```
   - LLM receives user query
   - LLM has tool schemas (from `bind_tools()`)
   - LLM analyzes if tools needed

3. **LLM Decision**:
   - If tools needed: Generate AIMessage with `tool_calls`
   - If no tools needed: Generate AIMessage with `content`

4. **Update State**:
   ```python
   return {"messages": [new_ai_message]}
   ```
   - State now contains: [HumanMessage, AIMessage]

---

### Step 5: Conditional Routing

**Location**: LangGraph's `tools_condition`

**Detailed Flow**:

1. **Check Last Message**:
   ```python
   last_message = state["messages"][-1]
   ```

2. **Check for Tool Calls**:
   ```python
   if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
       return "tools"
   else:
       return "__end__"
   ```

3. **Route Decision**:
   - If tool_calls exist: Route to "tools" node
   - If no tool_calls: Route to END

---

### Step 6: Tools Node Execution

**Location**: LangGraph's `ToolNode`

**Detailed Flow**:

1. **Extract Tool Calls**:
   ```python
   last_message = state["messages"][-1]
   tool_calls = last_message.tool_calls
   ```

2. **For Each Tool Call**:
   ```python
   for tool_call in tool_calls:
       tool_name = tool_call["name"]
       tool_args = tool_call["args"]
   ```

3. **Find and Execute Tool**:
   ```python
   tool = find_tool_by_name(tool_name)  # TavilySearchResults
   result = tool.invoke(tool_args)
   ```

4. **Tavily API Call**:
   ```python
   # Internal to TavilySearchResults
   response = tavily_api.post("/search", {
       "query": tool_args["query"],
       "max_results": 2
   })
   ```

5. **Create ToolMessage**:
   ```python
   tool_message = ToolMessage(
       content=json.dumps(response),
       tool_call_id=tool_call["id"]
   )
   ```

6. **Update State**:
   ```python
   return {"messages": [tool_message]}
   ```
   - State now contains: [HumanMessage, AIMessage, ToolMessage]

---

### Step 7: Second Chatbot Node Execution (Loop Back)

**Location**: `chatbot_with_Tool_node.py` → `chatbot_node()`

**Detailed Flow**:

1. **Receive Updated State**:
   ```python
   state["messages"] = [
       HumanMessage(query),
       AIMessage(tool_calls),
       ToolMessage(results)
   ]
   ```

2. **Invoke LLM Again**:
   ```python
   llm_with_tools.invoke(state["messages"])
   ```
   - LLM receives full conversation history
   - Including tool call and results

3. **LLM Synthesizes Answer**:
   - Reads tool results
   - Synthesizes with original query
   - Generates final answer

4. **Generate Final Response**:
   ```python
   AIMessage(
       content="Based on the search results...",
       tool_calls=[]  # Empty - no more tools needed
   )
   ```

5. **Update State**:
   ```python
   return {"messages": [final_ai_message]}
   ```
   - State now contains all messages including final answer

---

### Step 8: Final Conditional Routing

**Location**: LangGraph's `tools_condition`

1. **Check Last Message**:
   ```python
   last_message = state["messages"][-1]
   # No tool_calls in final AIMessage
   ```

2. **Route to END**:
   ```python
   return "__end__"
   ```

3. **Execution Completes**

---

### Step 9: Display Results

**Location**: `display_result.py` → `display_result_on_ui()`

1. **Extract All Messages**:
   ```python
   for message in res['messages']:
       if type(message) == HumanMessage:
           # Display user message
       elif type(message) == ToolMessage:
           # Display tool execution
       elif type(message) == AIMessage:
           # Display AI response
   ```

2. **Render in UI**:
   - User message in chat
   - Tool execution results
   - Final AI response

---

## 🔀 Conditional Routing

### tools_condition Function

**Purpose**: Automatically route based on tool call detection

**Implementation** (LangGraph internal):
```python
def tools_condition(state: MessagesState) -> str:
    """
    Routes to 'tools' if tool_calls exist, else to END.
    
    Args:
        state: Current graph state
        
    Returns:
        "tools" or "__end__"
    """
    last_message = state["messages"][-1]
    
    # Check if last message has tool_calls
    if hasattr(last_message, 'tool_calls'):
        if last_message.tool_calls and len(last_message.tool_calls) > 0:
            return "tools"
    
    return "__end__"
```

### Routing Scenarios

**Scenario 1: Tool Calls Detected**
```
State:
  messages: [
    HumanMessage("What's weather in NYC?"),
    AIMessage(tool_calls=[{...}])
  ]

tools_condition → Returns "tools"
Route → tools node
```

**Scenario 2: No Tool Calls**
```
State:
  messages: [
    HumanMessage("What's weather in NYC?"),
    AIMessage(content="The weather is...", tool_calls=[])
  ]

tools_condition → Returns "__end__"
Route → END
```

### Conditional Edge Configuration

```python
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,  # Routing function
    {
        "tools": "tools",      # If returns "tools", go to tools node
        "__end__": END         # If returns "__end__", go to END
    }
)
```

---

## 🛠️ Tool Integration

### Tavily Search Tool

**Tool Definition**:
```python
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=2)]
```

**Tool Schema** (automatically generated):
```json
{
    "name": "tavily_search_results_json",
    "description": "A search engine. Useful for when you need to answer questions about current events.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string"
            }
        },
        "required": ["query"]
    }
}
```

**Tool Execution**:
```python
# LLM generates tool call:
tool_call = {
    "name": "tavily_search_results_json",
    "args": {"query": "weather New York"},
    "id": "call_123"
}

# ToolNode executes:
result = tavily_search.invoke({"query": "weather New York"})

# Tavily API returns:
[
    {
        "title": "Weather in New York",
        "url": "https://...",
        "content": "Current weather conditions..."
    },
    ...
]
```

### Tool Binding Process

**Step 1: Bind Tools to LLM**:
```python
llm_with_tools = llm.bind_tools(tools)
```

**What Happens**:
1. Tool schemas are injected into LLM prompt
2. LLM can now generate tool calls in structured format
3. LLM understands when tools are appropriate

**Step 2: LLM Decision**:
- LLM analyzes user query
- Determines if tool needed for current information
- If needed: Generates tool call
- If not: Generates direct response

**Step 3: Tool Execution**:
- ToolNode extracts tool calls
- Executes each tool
- Returns results as ToolMessage

**Step 4: LLM Synthesis**:
- LLM receives tool results
- Synthesizes information
- Generates final answer

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
                    ▼ graph.invoke()
        ┌───────────────────────┐
        │  LANGGRAPH GRAPH       │
        │  ┌─────────────────┐  │
        │  │  chatbot node    │  │
        │  └────────┬────────┘  │
        │           │            │
        │           ▼            │
        │  ┌─────────────────┐  │
        │  │tools_condition  │  │
        │  └────────┬────────┘  │
        │    ┌──────┴──────┐     │
        │    │             │     │
        │    ▼             ▼     │
        │  END          tools    │
        │              node      │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  ChatbotWithToolNode  │
        │  • bind_tools()       │
        │  • invoke()           │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  ToolNode             │
        │  • Extract tool_calls │
        │  • Execute tools      │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  TavilySearchResults  │
        │  • invoke()           │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Tavily API           │
        │  • Web Search         │
        └───────────────────────┘
```

### Method Call Chain

**Complete Execution Chain**:

```
main.load_langgraph_agenticai_app()
  └─► LoadStreamlitUI.load_streamlit_ui()
  └─► GroqLLM.get_llm_model()
  └─► GraphBuilder.setup_graph()
        └─► GraphBuilder.chatbot_with_tools_build_graph()
              ├─► get_tools()
              │     └─► TavilySearchResults(max_results=2)
              ├─► create_tool_node(tools)
              │     └─► ToolNode(tools=tools)
              └─► ChatbotWithToolNode.create_chatbot(tools)
                    └─► llm.bind_tools(tools)
  └─► DisplayResultStreamlit.display_result_on_ui()
        └─► graph.invoke(initial_state)
              ├─► ChatbotNode(state)
              │     └─► llm_with_tools.invoke(state["messages"])
              │           └─► Groq API Call
              │
              ├─► LangGraph: tools_condition(state)
              │     └─► Checks tool_calls
              │
              ├─► ToolNode(state) [if tools needed]
              │     ├─► Extract tool_calls
              │     ├─► TavilySearchResults.invoke(args)
              │     │     └─► Tavily API Call
              │     └─► Create ToolMessage
              │
              ├─► ChatbotNode(state) [second pass]
              │     └─► llm_with_tools.invoke(state["messages"])
              │           └─► Groq API Call (with tool results)
              │
              └─► LangGraph: tools_condition(state) [again]
                    └─► Routes to END (no more tools)
```

---

## ⚡ Performance Considerations

### Optimization Strategies

1. **Tool Result Limiting**:
   - `max_results=2`: Limits Tavily results to top 2
   - Reduces token usage in LLM context
   - Faster API responses

2. **Conditional Execution**:
   - Tools only executed when needed
   - LLM decides tool necessity
   - Avoids unnecessary API calls

3. **Message History Management**:
   - Full conversation history maintained
   - Enables context-aware responses
   - LLM can reference previous interactions

4. **Parallel Tool Execution**:
   - ToolNode can execute multiple tools in parallel
   - Currently single tool (Tavily) but extensible

### Bottlenecks

1. **API Calls**:
   - Groq API: ~1-2 seconds per call
   - Tavily API: ~0.5-1 second per call
   - Total: 2-4 seconds for full execution (with tools)

2. **Tool Call Detection**:
   - LLM must decide if tools needed
   - Additional processing time
   - Worthwhile for accurate routing

3. **Message Processing**:
   - Full message history sent to LLM each time
   - Larger context window usage
   - Necessary for conversation context

### Scalability

**Current Limitations**:
- Single tool execution (Tavily)
- Sequential tool execution
- Single LLM provider (Groq)

**Future Enhancements**:
- Multiple tools (parallel execution)
- Tool result caching
- Streaming responses
- Multiple LLM providers

---

## 📝 Summary

This architecture document provides a comprehensive overview of:

1. **System Architecture**: Multi-layered design from UI to external APIs
2. **Component Details**: Each component's purpose and responsibilities
3. **Execution Flow**: Step-by-step execution with conditional routing
4. **State Management**: How state and messages evolve through execution
5. **Data Flow**: Query processing, tool execution, and response generation
6. **Conditional Routing**: How `tools_condition` enables dynamic routing
7. **Tool Integration**: Tavily search tool integration and execution
8. **Component Interactions**: How components communicate
9. **Performance**: Optimization strategies and considerations

This architecture enables:
- **Dynamic Routing**: Conditional execution based on tool needs
- **Tool Integration**: Seamless external tool usage
- **Conversation Context**: Full history maintenance
- **Extensibility**: Easy to add more tools
- **User Experience**: Real-time tool execution and results display

---

**End of Architecture Documentation**
