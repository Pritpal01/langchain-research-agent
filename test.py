import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file (optional, if you're using .env)
load_dotenv()

# Set API key (use your actual API key here if not using .env)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "your_api_key_here"
genai.configure(api_key=GOOGLE_API_KEY)

# List models
models = genai.list_models()

print("Available Gemini Models for your API Key:\n")
for model in models:
    print(f"- {model.name}")
