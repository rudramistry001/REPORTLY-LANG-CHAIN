import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app from server module
from server.langchain_server import app

# This is required for Vercel serverless functions 