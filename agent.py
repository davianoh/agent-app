import os
from typing import Dict, List, Any, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    """Type definition for the agent state."""
    messages: List[Any]
    current_step: str
    task: str
    result: str


class LangGraphAgent:
    """A simple LangGraph agent using Ollama LLM."""
    
    def __init__(self, model_name: str = "llama3"):
        """Initialize the LangGraph agent.
        
        Args:
            model_name: The name of the Ollama model to use.
        """
        self.model_name = model_name
        self.llm = Ollama(model=model_name)
        self.setup_graph()
    
    def setup_graph(self):
        """Set up the LangGraph state machine."""
        # Create the state graph
        self.workflow = StateGraph(AgentState)
        
        # Add nodes to the graph
        self.workflow.add_node("understand_task", self.understand_task)
        self.workflow.add_node("execute_task", self.execute_task)
        self.workflow.add_node("summarize_result", self.summarize_result)
        
        # Add edges to connect the nodes
        self.workflow.add_edge("understand_task", "execute_task")
        self.workflow.add_edge("execute_task", "summarize_result")
        self.workflow.add_edge("summarize_result", END)
        
        # Set the entry point
        self.workflow.set_entry_point("understand_task")
        
        # Compile the graph
        self.graph = self.workflow.compile()
    
    def understand_task(self, state: AgentState) -> AgentState:
        """Understand the task and plan the execution.
        
        Args:
            state: The current state of the agent.
            
        Returns:
            The updated state.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that understands user tasks and breaks them down into steps."),
            ("human", "I need help with the following task: {task}"),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        understanding = chain.invoke({"task": state["task"]})
        
        # Update the state
        new_state = state.copy()
        new_state["messages"] = state["messages"] + [
            HumanMessage(content=state["task"]),
            AIMessage(content=understanding)
        ]
        new_state["current_step"] = "understand_task"
        
        return new_state
    
    def execute_task(self, state: AgentState) -> AgentState:
        """Execute the task based on the understanding.
        
        Args:
            state: The current state of the agent.
            
        Returns:
            The updated state.
        """
        last_message = state["messages"][-1].content
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that executes tasks based on the provided understanding."),
            ("human", "I need you to execute this task: {task}"),
            ("human", "Your understanding of the task: {understanding}"),
            ("human", "Please provide the execution result."),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        execution_result = chain.invoke({
            "task": state["task"],
            "understanding": last_message
        })
        
        # Update the state
        new_state = state.copy()
        new_state["messages"] = state["messages"] + [
            AIMessage(content=execution_result)
        ]
        new_state["current_step"] = "execute_task"
        
        return new_state
    
    def summarize_result(self, state: AgentState) -> AgentState:
        """Summarize the execution result.
        
        Args:
            state: The current state of the agent.
            
        Returns:
            The updated state.
        """
        execution_result = state["messages"][-1].content
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that summarizes execution results concisely."),
            ("human", "Original task: {task}"),
            ("human", "Execution result: {result}"),
            ("human", "Please provide a concise summary of the results."),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        summary = chain.invoke({
            "task": state["task"],
            "result": execution_result
        })
        
        # Update the state
        new_state = state.copy()
        new_state["messages"] = state["messages"] + [
            AIMessage(content=summary)
        ]
        new_state["current_step"] = "summarize_result"
        new_state["result"] = summary
        
        return new_state
    
    def run(self, task: str) -> Dict[str, Any]:
        """Run the agent on a given task.
        
        Args:
            task: The task to run the agent on.
            
        Returns:
            The final state of the agent.
        """
        # Initialize the state
        initial_state = {
            "messages": [],
            "current_step": "",
            "task": task,
            "result": ""
        }
        
        # Run the graph and get the final state
        result = self.graph.invoke(initial_state)
        return result


if __name__ == "__main__":
    # Example usage
    agent = LangGraphAgent()
    result = agent.run("Explain the concept of quantum computing in simple terms.")
    print(result["result"])