import streamlit as st
from agent import LangGraphAgent

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
            model_name = st.selectbox(
                "Select Ollama Model",
                ["llama3-8b-8192", "llama-3.2-1b-preview", "llama-3.2-3b-preview"],
                index=0
            )
            
            if st.button("Initialize Agent"):
                with st.spinner("Initializing agent..."):
                    self.agent = LangGraphAgent(model_name=model_name)
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
                        # Run the agent
                        result = st.session_state.agent.run(prompt, "1")
                        
                        # Display the final result
                        response = result["messages"][-1].content
                        response_container.write(response)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        response_container.write(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    def run(self):
        """Run the Streamlit application."""
        st.title(self.title)
        
        # Set up the sidebar
        self.setup_sidebar()
        
        # Display the conversation history
        self.display_messages()
        
        # Handle user input
        self.handle_user_input()


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()