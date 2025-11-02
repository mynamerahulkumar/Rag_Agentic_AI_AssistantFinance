from langgraph.graph import StateGraph, START,END, MessagesState
from langgraph.prebuilt import tools_condition,ToolNode
from langchain_core.prompts import ChatPromptTemplate
from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraphagenticai.nodes.chatbot_with_Tool_node import ChatbotWithToolNode
from src.langgraphagenticai.tools.serach_tool import get_tools,create_tool_node
from src.langgraphagenticai.nodes.rag_node import RAGNode




class GraphBuilder:

    def __init__(self,model):
        self.llm=model
        self.graph_builder=StateGraph(State)

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the `BasicChatbotNode` class 
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """
        self.basic_chatbot_node=BasicChatbotNode(self.llm)
        self.graph_builder.add_node("chatbot",self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_edge("chatbot",END)


    def chatbot_with_tools_build_graph(self):
        """
        Builds an advanced chatbot graph with tool integration.
        This method creates a chatbot graph that includes both a chatbot node 
        and a tool node. It defines tools, initializes the chatbot with tool 
        capabilities, and sets up conditional and direct edges between nodes. 
        The chatbot node is set as the entry point.
        """
        ## Define the tool and tool node

        tools=get_tools()
        tool_node=create_tool_node(tools)

        ##Define LLM
        llm = self.llm

        # Define chatbot node
        obj_chatbot_with_node = ChatbotWithToolNode(llm)
        chatbot_node = obj_chatbot_with_node.create_chatbot(tools)

        # Add nodes
        self.graph_builder.add_node("chatbot", chatbot_node)
        self.graph_builder.add_node("tools", tool_node)

        # Define conditional and direct edges
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_conditional_edges("chatbot", tools_condition)
        self.graph_builder.add_edge("tools","chatbot")

    def rag_build_graph(self, openai_api_key: str):
        """
        Builds a RAG graph with document processing, retrieval, and response generation.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
        """
        print("rag started--")
        rag_node = RAGNode(self.llm, openai_api_key)
        
        self.graph_builder.add_node("process_documents", rag_node.process_documents)
        self.graph_builder.add_node("retrieve_context", rag_node.retrieve_context)
        self.graph_builder.add_node("generate_response", rag_node.generate_response)
        
        self.graph_builder.set_entry_point("process_documents")
        self.graph_builder.add_edge("process_documents", "retrieve_context")
        self.graph_builder.add_edge("retrieve_context", "generate_response")
        self.graph_builder.add_edge("generate_response", END)
    
    def setup_graph(self, usecase: str, openai_api_key: str = None):
        """
        Sets up the graph for the selected use case.
        
        Args:
            usecase: The selected use case
            openai_api_key: OpenAI API key (required for RAG Chatbot)
        """
        print("usecase--",usecase)
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()

        if usecase == "Chatbot with Tool":
            self.chatbot_with_tools_build_graph()
            
        if usecase == "RAG Chatbot":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for RAG Chatbot")
            self.rag_build_graph(openai_api_key)
            
        return self.graph_builder.compile()
        
        
    




    

