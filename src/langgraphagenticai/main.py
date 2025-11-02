import streamlit as st
import json
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit

# MAIN Function START
def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while 
    implementing exception handling for robustness.
    """
   
    # Load UI
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return

    # Text input for user message
    user_message = st.chat_input("Enter your message:")

    if user_message:
            try:
                # Configure LLM
                obj_llm_config = GroqLLM(user_controls_input=user_input)
                model = obj_llm_config.get_llm_model()
                
                if not model:
                    st.error("Error: LLM model could not be initialized.")
                    return

                # Initialize and set up the graph based on use case
                usecase = user_input.get('selected_usecase')
                if not usecase:
                    st.error("Error: No use case selected.")
                    return
                

                ### Graph Builder
                graph_builder=GraphBuilder(model)
                try:
                    # Get OpenAI API key and uploaded files for RAG Chatbot
                    openai_api_key = user_input.get('OPENAI_API_KEY', None) if usecase == "RAG Chatbot" else None
                    uploaded_files = user_input.get('uploaded_files', []) if usecase == "RAG Chatbot" else None
                    
                    graph = graph_builder.setup_graph(usecase, openai_api_key=openai_api_key)
                    DisplayResultStreamlit(usecase, graph, user_message, uploaded_files=uploaded_files).display_result_on_ui()
                except Exception as e:
                    st.error(f"Error: Graph setup failed - {e}")
                    return
                

            except Exception as e:
                 raise ValueError(f"Error Occurred with Exception : {e}")
            

        

   

    
