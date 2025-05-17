from dotenv import load_dotenv
import os

load_dotenv()
class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV_REGION = os.getenv("PINECONE_ENV_REGION")
    OPENAI_MODEL = "gpt-4o-mini"
    SERVER_PORT = 5001
