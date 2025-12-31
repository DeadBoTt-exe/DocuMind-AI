"""
Lists all Gemini models available for the configured API key using the
latest Google GenAI SDK. Useful for selecting compatible models and
debugging API access issues.
"""


import os
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

models = client.models.list()

print("\nAVAILABLE MODELS:\n")
for m in models:
    print(m.name)
