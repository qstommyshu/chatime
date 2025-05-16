from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_pinecone import PineconeVectorStore

def get_context_retriever_chain(vector_store: PineconeVectorStore):
    """
    Create a history-aware retriever chain.
    
    Args:
        vector_store (PineconeVectorStore): The vector store to retrieve from
        
    Returns:
        Chain: A history-aware retriever chain
    """
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever(
        # Increase k to search for more relevant information
        search_kwargs={"k": 10, "namespace": ""}
    )
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    return create_history_aware_retriever(llm, retriever, prompt) 