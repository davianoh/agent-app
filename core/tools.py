# core/tools.py
from typing import Dict, List, Any
from langchain_community.tools.tavily_search import TavilySearchResults
import os

def search_web(query: str) -> str:
    """ Retrieve additional context and information from web search using the query. 
    
    Args: 
        query: string data type of the search query to retrieve information from.
    """

    # Handle the case where query is passed as a dictionary from tool calling
    if isinstance(query, dict) and "query" in query:
        query = query["query"]

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