import streamlit as st
from langgraph_agents.summarized_memory_agent import SummarizedMemoryAgent
from langgraph_agents.web_search_agent import WebSearchAgent
import os

# os.environ["LANGCHAIN_PROJECT"] = "groq-agent"

class StreamlitApp:
    """A Streamlit application for interacting with the LangGraph agent."""
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.title = "LangGraph Agent Demo"
        self.agent = None
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "agent" not in st.session_state:
            st.session_state.agent = None
    
    def setup_sidebar(self):
        """Set up the sidebar with model selection."""
        with st.sidebar:
            st.title("Configuration")
        
            # Agent type selection
            agent_type = st.selectbox(
                "Select Agent Variant",
                ["SummarizedMemoryAgent", "WebSearchAgent"],
                index=0,
                key="agent_type"
            )
            
            # Model options based on agent type
            if agent_type == "SummarizedMemoryAgent":
                model_options = ["llama3-8b-8192", "llama-3.2-3b-preview"]
                default_index = 0
            elif agent_type == "WebSearchAgent":
                model_options = ["qwen-2.5-32b"]
                default_index = 0
            
            model_name = st.selectbox(
                "Select LLM Model",
                model_options,
                index=default_index
            )
        
            if st.button("Initialize Agent"):
                with st.spinner("Initializing agent..."):
                    if agent_type == "SummarizedMemoryAgent":
                        self.agent = SummarizedMemoryAgent(model_name=model_name)
                    elif agent_type == "WebSearchAgent":
                        self.agent = WebSearchAgent(model_name=model_name)
                
                    st.session_state.agent = self.agent
                    st.success(f"Agent initialized with {model_name} model!")
    
    def display_messages(self):
        """Display the conversation history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    def handle_user_input(self):
        """Handle user input and agent responses."""
        if prompt := st.chat_input("What would you like help with?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Check if agent is initialized
            if st.session_state.agent is None:
                with st.chat_message("assistant"):
                    st.write("Please initialize the agent from the sidebar first.")
                st.session_state.messages.append({"role": "assistant", "content": "Please initialize the agent from the sidebar first."})
                return
            
            # Process with agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_container = st.empty()
                    try:
                        result = st.session_state.agent.run(prompt, "1")
                        
                        response = result["messages"][-1].content
                        response_container.write(response)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        response_container.write(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    def run(self):
        """Run the Streamlit application."""
        st.title(self.title)
        
        self.setup_sidebar()
        self.display_messages()
        self.handle_user_input()


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()