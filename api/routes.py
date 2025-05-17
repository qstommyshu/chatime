from flask import Blueprint, jsonify, request
from services.retriever import get_context_retriever_chain
from services.rag_chain import get_conversational_rag_chain
from utils.history import map_to_langchain_messages
from services.vector_store import initialize_vector_store

# Initialize the vector store
vector_store = initialize_vector_store()

# Create a Blueprint for API routes
api_bp = Blueprint('api', __name__)

@api_bp.route('/health')
def health():
    return {"status": "ok"}


@api_bp.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint that processes user questions and returns AI responses.
    
    Request body:
    {
        "history": List of conversation messages,
        "question": User's question
    }
    
    Returns:
        JSON response with answer and sources
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        history = data.get('history', [])
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Convert history to LangChain format
        langchain_history = map_to_langchain_messages(history)
        
        # Get retriever chain
        retriever_chain = get_context_retriever_chain(vector_store)
        
        # Get RAG chain
        rag_chain = get_conversational_rag_chain(retriever_chain)
        
        # Generate response
        response = rag_chain.invoke({
            'chat_history': langchain_history,
            'input': question
        })
        
        # Return the response as JSON
        return jsonify(response.dict()), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500 