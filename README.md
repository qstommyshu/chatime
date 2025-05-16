# ChatTime

A conversational RAG (Retrieval-Augmented Generation) application powered by LangChain and Pinecone.

## Project Structure

```
chatime/
├── app.py               # Main application entry point
├── config.py            # Configuration settings
├── api/                 # API endpoints
│   └── routes.py        # Flask routes
├── models/              # Data models
│   └── schema.py        # Pydantic models
├── services/            # Business logic services
│   ├── vector_store.py  # Pinecone initialization
│   ├── retriever.py     # Retrieval chain
│   └── rag_chain.py     # RAG chain implementation
├── utils/               # Utility functions
│   └── history.py       # Chat history utils
└── requirements.txt     # Project dependencies
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/chatime.git
cd chatime
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV_REGION=your_pinecone_region
```

4. Run the application:

```bash
python app.py
```

## API Endpoints

### Chat Endpoint

- **URL**: `/chat`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "history": [
      { "role": "human", "content": "Hello" },
      { "role": "ai", "content": "Hi there! How can I help you?" }
    ],
    "question": "What information do you have about ChatTime?"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "ChatTime is a conversational RAG application...",
    "sources": ["https://example.com/source1", "https://example.com/source2"]
  }
  ```
