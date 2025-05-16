from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage

def map_to_langchain_messages(conversation: List[Dict[str, str]]):
    """
    Convert a list of conversation messages to LangChain message format.
    
    Args:
        conversation (List[Dict[str, str]]): List of conversation messages with 'role' and 'content'
        
    Returns:
        List: Conversation history in LangChain format
    """
    return [
        AIMessage(content=message["content"]) if message["role"] == "ai" 
        else HumanMessage(content=message["content"]) 
        for message in conversation
    ] 