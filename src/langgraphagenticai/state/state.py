from typing import Annotated, Literal, Optional, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

class State(TypedDict, total=False):
    """
    Represents the structure of the state used in the graph.
    """
    messages: Annotated[list, add_messages]
    uploaded_files: List  # Optional field for RAG document uploads
    query: str  # Optional field for RAG query
    retrieved_context: str  # Optional field for retrieved context
    retrieved_docs: List  # Optional field for retrieved documents
    documents_processed: bool  # Optional field to track document processing
    num_chunks: int  # Optional field for number of chunks
    error: str  # Optional field for errors