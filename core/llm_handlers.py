# core/llm_handlers.py
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import MessagesState

class ModelHandlers:
    """Reusable model handlers for LangGraph agents."""
    
    @staticmethod
    def call_model(state: MessagesState, llm, summary_field="summary"):
        """Generic model calling function that handles summaries.
        
        Args:
            state: The current state with messages
            llm: The language model to use
            summary_field: The field name where summary is stored
            
        Returns:
            Updated state with model response
        """
        # Get summary if it exists
        summary = state.get(summary_field, "")

        # If there is summary, then we add it
        if summary:
            # Add summary to system message
            system_message = f"Summary of conversation earlier: {summary}"

            # Append summary to any newer messages
            messages = [SystemMessage(content=system_message)] + state["messages"]
        
        else:
            messages = state["messages"]
        
        response = llm.invoke(messages)
        return {"messages": response}
    
    @staticmethod
    def summarize_conversation(state: MessagesState, llm, summary_field="summary", keep_messages=2):
        """Generic conversation summarization function.
        
        Args:
            state: The current state with messages
            llm: The language model to use
            summary_field: The field name where summary is stored
            keep_messages: Number of most recent messages to keep
            
        Returns:
            Updated state with summary and pruned messages
        """
        # First, we get any existing summary
        summary = state.get(summary_field, "")

        # Create our summarization prompt 
        if summary:
            # A summary already exists
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
            
        else:
            summary_message = "Create a summary of the conversation above:"

        # Add prompt to our history
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = llm.invoke(messages)
        
        # Delete all but the N most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-keep_messages]]
        return {summary_field: response.content, "messages": delete_messages}
    
    @staticmethod
    def should_continue(state: MessagesState, threshold=6, summary_node="summarize_conversation", end="END"):
        """Generic conditional function to decide whether to continue or summarize.
        
        Args:
            state: The current state with messages
            threshold: Number of messages that trigger summarization
            summary_node: Name of the summarization node
            end: END constant from LangGraph
            
        Returns:
            Next node to execute
        """
        messages = state["messages"]
        
        # If there are more than threshold messages, then we summarize the conversation
        if len(messages) > threshold:
            return summary_node
        
        # Otherwise we can just end
        return end