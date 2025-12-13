"""
Configuration settings for the Graph-RAG pipeline
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not NEO4J_PASSWORD:
    raise ValueError("⚠️  NEO4J_PASSWORD must be set in .env file")

# Embedding Model
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

# LLM Models
AVAILABLE_LLM_MODELS = [
    'gemma-2-9b-it',
    'llama-3.1-8b-instruct',
    'mistral-7b-instruct'
]

# HuggingFace Token (if needed)
HF_TOKEN = os.getenv('HF_TOKEN', None)

# Retrieval Settings
DEFAULT_MAX_RESULTS = 10
DEFAULT_EMBEDDING_TOP_K = 5