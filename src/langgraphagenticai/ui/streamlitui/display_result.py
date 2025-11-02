import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import json


class DisplayResultStreamlit:
    def __init__(self,usecase,graph,user_message,uploaded_files=None):
        self.usecase= usecase
        self.graph = graph
        self.user_message = user_message
        self.uploaded_files = uploaded_files

    def display_result_on_ui(self):
        usecase= self.usecase
        graph = self.graph
        user_message = self.user_message
        if usecase =="Basic Chatbot":
                for event in graph.stream({'messages':("user",user_message)}):
                    print(event.values())
                    for value in event.values():
                        print(value['messages'])
                        with st.chat_message("user"):
                            st.write(user_message)
                        with st.chat_message("assistant"):
                            st.write(value["messages"].content)

        elif usecase=="Chatbot with Tool":
             # Prepare state and invoke the graph
            initial_state = {"messages": [user_message]}
            res = graph.invoke(initial_state)
            for message in res['messages']:
                if type(message) == HumanMessage:
                    with st.chat_message("user"):
                        st.write(message.content)
                elif type(message)==ToolMessage:
                    with st.chat_message("ai"):
                        st.write("Tool Call Start")
                        st.write(message.content)
                        st.write("Tool Call End")
                elif type(message)==AIMessage and message.content:
                    with st.chat_message("assistant"):
                        st.write(message.content)

        elif usecase == "RAG Chatbot":
            # Display user message
            with st.chat_message("user"):
                st.write(user_message)
            
            # Prepare initial state with uploaded files and user message
            initial_state = {
                "messages": [HumanMessage(content=user_message)],
                "uploaded_files": self.uploaded_files if self.uploaded_files else []
            }
            
            # Check if files are uploaded
            if not self.uploaded_files or len(self.uploaded_files) == 0:
                with st.chat_message("assistant"):
                    st.error("⚠️ Please upload at least one document (PDF or TXT) before asking questions.")
                return
            
            # Stream the graph to show progress at each step
            status_placeholder = st.empty()
            
            try:
                # Stream through the graph to see each step
                final_result = None
                for event in graph.stream(initial_state):
                    # Print event to console for debugging
                    print("Event:", event)
                    
                    for node_name, node_output in event.items():
                        print(f"Node: {node_name}, Output: {node_output}")
                        
                        # Show progress for each step
                        if node_name == "process_documents":
                            status_placeholder.info("📄 **Step 1/3**: Processing documents and creating vector store...")
                            if node_output.get('documents_processed'):
                                num_chunks = node_output.get('num_chunks', 0)
                                status_placeholder.success(f"✅ Step 1 Complete: Processed {num_chunks} chunks")
                            elif node_output.get('error'):
                                error_msg = node_output['error']
                                status_placeholder.error(f"❌ Step 1 Error: {error_msg}")
                                # Show detailed error
                                st.error(f"**Document Processing Failed:** {error_msg}")
                                if 'pypdf' in error_msg.lower() or 'package not found' in error_msg.lower():
                                    st.warning("💡 **Solution:** Please install pypdf by running: `pip install pypdf`")
                                st.stop()
                        
                        elif node_name == "retrieve_context":
                            status_placeholder.info("🔍 **Step 2/3**: Retrieving relevant context from documents...")
                            if node_output.get('retrieved_context'):
                                context_len = len(node_output.get('retrieved_context', ''))
                                status_placeholder.success(f"✅ Step 2 Complete: Retrieved context ({context_len} chars)")
                            elif node_output.get('error'):
                                error_msg = node_output['error']
                                status_placeholder.error(f"❌ Step 2 Error: {error_msg}")
                                st.error(f"**Context Retrieval Failed:** {error_msg}")
                                # If it's because vectorstore doesn't exist, show helpful message
                                if 'vector store not initialized' in error_msg.lower():
                                    st.error("⚠️ Documents were not processed successfully in Step 1. Check the console logs for details.")
                                st.stop()
                        
                        elif node_name == "generate_response":
                            status_placeholder.info("🤖 **Step 3/3**: Generating response with LLM...")
                            final_result = node_output
                
                # Display final response
                if final_result and final_result.get('messages'):
                    status_placeholder.empty()  # Clear status messages
                    with st.chat_message("assistant"):
                        response_message = final_result['messages'][0]
                        if hasattr(response_message, 'content'):
                            st.write(response_message.content)
                        else:
                            st.write(str(response_message))
                elif final_result and final_result.get('error'):
                    status_placeholder.error(f"❌ Error: {final_result['error']}")
                    with st.chat_message("assistant"):
                        st.error(f"Error: {final_result['error']}")
                        
            except Exception as e:
                status_placeholder.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc())
                print(f"Exception traceback:\n{traceback.format_exc()}")
             
