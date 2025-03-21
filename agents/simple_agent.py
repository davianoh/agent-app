# agents/web_search_agent.py
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage
from langgraph.graph import END

from core.base_agent import BaseAgent, BaseState
from core.tools import search_web
from core.llm_handlers import ModelHandlers

class SimpleAgent(BaseAgent):
    """A LangGraph agent with web search capabilities."""
    
    def __init__(self, model_name: str = "qwen-2.5-32b"):
        """Initialize the SimpleAgent.
        
        Args:
            model_name: The name of the model to use
        """
        # Pass the search_web function as a tool
        super().__init__(model_name=model_name)
        self.setup_graph()
    
    def setup_graph(self):
        """Set up the LangGraph state machine for web search agent."""
        # Create basic graph structure
        super().setup_graph()
        
        # Add summarization node with bound method
        summarize_fn = lambda state: ModelHandlers.summarize_conversation(state, self.llm)
        self.workflow.add_node("summarize_conversation", summarize_fn)
        
        # Add conditional edge with continuation check
        should_continue_fn = lambda state: ModelHandlers.should_continue(
            state, 
            threshold=6, 
            summary_node="summarize_conversation",
            end=END
        )
        self.workflow.add_conditional_edges("conversation", should_continue_fn)
        
        # Add edge from summarization to end
        self.workflow.add_edge("summarize_conversation", END)
        
        # Re-compile the graph with the new nodes and edges
        self.graph = self.workflow.compile(checkpointer=self.memory)