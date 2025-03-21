from pydantic import BaseModel, Field
from typing import Dict, List, Any, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import tools_condition, ToolNode

from dotenv import load_dotenv
import os
load_dotenv()

class State(MessagesState):
    summary: str

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class WebSearchAgent:
    """A simple LangGraph agent using Ollama LLM."""
    
    def __init__(self, model_name: str = "qwen-2.5-32b"):
        """Initialize the LangGraph agent.
        
        Args:
            model_name: The name of the Ollama model to use.
        """

        self.model_name = model_name
        self.tools = [self.search_web]
        self.llm = ChatGroq(
                groq_api_key=os.environ['GROQ_API_KEY'], 
                model_name=self.model_name
        ).bind_tools(self.tools)
        
        self.setup_graph()
    
    def setup_graph(self):
        """Set up the LangGraph state machine."""
        # Create the state graph
        self.workflow = StateGraph(State)
        
        # Add nodes to the graph
        self.workflow.add_node("conversation", self.call_model)
        self.workflow.add_node(self.summarize_conversation)
        self.workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges to connect the nodes
        self.workflow.add_edge(START, "conversation")
        self.workflow.add_conditional_edges("conversation", tools_condition)
        self.workflow.add_edge("tools", "conversation")
        self.workflow.add_conditional_edges("conversation", self.should_continue)
        self.workflow.add_edge("summarize_conversation", END)
        
        # Compile the graph
        self.memory = MemorySaver()
        self.graph = self.workflow.compile(checkpointer=self.memory)
    
    # this is a tool function
    def search_web(self, query: str) -> str:
        """ Retrieve additional context and information from web search using the query. 
        
        Args: 
            query: string data type of the search query to retrieve information from.
        """

        # Search
        tavily_search = TavilySearchResults(
            max_results=3, 
            tavily_api_key=os.environ['TAVILY_API_KEY']
        )

        search_docs = tavily_search.invoke(query)

        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ]
        )

        return formatted_search_docs    
    
    
    
    def call_model(self, state: State) -> State: 
        
        # Get summary if it exists
        summary = state.get("summary", "")

        # If there is summary, then we add it
        if summary:
            
            # Add summary to system message
            system_message = f"Summary of conversation earlier: {summary}"

            # Append summary to any newer messages
            messages = [SystemMessage(content=system_message)] + state["messages"]
        
        else:
            messages = state["messages"]
        
        response = self.llm.invoke(messages)
        return {"messages": response}
    
    def summarize_conversation(self, state: State) -> State: 
        
        # First, we get any existing summary
        summary = state.get("summary", "")

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
        response = self.llm.invoke(messages)
        
        # Delete all but the 2 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}
    
    def should_continue(self, state: State) -> State:
        
        """Return the next node to execute."""
        
        messages = state["messages"]
        
        # If there are more than six messages, then we summarize the conversation
        if len(messages) > 6:
            return "summarize_conversation"
        
        # Otherwise we can just end
        return END
    
    
    def run(self, task: str, thread_id: str) -> Dict[str, Any]:
        """Run the agent on a given task.
        
        Args:
            task: The task to run the agent on.
            
        Returns:
            The final state of the agent.
        """

        config = {"configurable": {"thread_id": thread_id}}
        task = {"messages": [HumanMessage(content=task)]}

        # Run the graph and get the final state
        result = self.graph.invoke(task, config)
        return result


if __name__ == "__main__":
    # Example usage
    agent = WebSearchAgent()
    result = agent.run("What's the capital of Indonesia?", "1")
    print(result['messages'])
