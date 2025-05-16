from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableMap
from models.schema import StructuredAnswer
from config import Config

def get_conversational_rag_chain(retriever_chain):
    """
    Create a conversational RAG chain with structured output.
    
    Args:
        retriever_chain: The retriever chain to get context
        
    Returns:
        Chain: A conversational RAG chain
    """
    llm = ChatOpenAI(model=Config.OPENAI_MODEL).with_structured_output(StructuredAnswer)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}\n\n"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    # Map input to prompt and run LLM
    rag_chain = (
        RunnableMap({
            "context": retriever_chain,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"]
        })
        | prompt
        | llm
    )

    return rag_chain 