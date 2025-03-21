# core/base_agent.py
from typing import Dict, List, Any, Optional, Callable
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_groq import ChatGroq
import os

from core.llm_handlers import ModelHandlers

class BaseState(MessagesState):
    summary: str

class BaseAgent:
    """Base class for LangGraph agents with common functionality."""
    
    def __init__(self, 
                 model_name: str,
                 tools: Optional[List] = None,
                 state_class=BaseState):
        """Initialize the base LangGraph agent.
        
        Args:
            model_name: The name of the LLM model to use
            tools: Optional list of tools to use
            state_class: The state class to use
        """
        self.model_name = model_name
        self.tools = tools or []
        self.state_class = state_class
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=os.environ['GROQ_API_KEY'], 
            model_name=self.model_name
        )
        
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)
        
        # Create memory and workflow (to be filled by subclasses)
        self.memory = None
        self.workflow = None
        self.graph = None
    
    def setup_graph(self):
        """Base graph setup method to be extended by subclasses."""
        # Create the state graph
        self.workflow = StateGraph(self.state_class)
        
        # Add basic nodes (to be extended)
        self.workflow.add_node("conversation", 
                              lambda state: ModelHandlers.call_model(state, self.llm))
        
        # Basic edges
        self.workflow.add_edge(START, "conversation")
        
        # If there are tools, set up tool handling
        if self.tools:
            self.workflow.add_node("tools", ToolNode(self.tools))
            self.workflow.add_conditional_edges("conversation", tools_condition)
            self.workflow.add_edge("tools", "conversation")
        
        # Finalize
        self.memory = MemorySaver()
        self.graph = self.workflow.compile(checkpointer=self.memory)
    
    def run(self, task: str, thread_id: str) -> Dict[str, Any]:
        """Run the agent on a given task.
        
        Args:
            task: The task to run the agent on
            thread_id: Thread identifier for persistence
            
        Returns:
            The final state of the agent
        """
        if not self.graph:
            raise ValueError("Graph not compiled. Call setup_graph() first.")
            
        config = {"configurable": {"thread_id": thread_id}}
        task = {"messages": [HumanMessage(content=task)]}

        # Run the graph and get the final state
        result = self.graph.invoke(task, config)
        return result