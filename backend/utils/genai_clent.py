# genai_client.py
from google import genai
import os, dotenv; dotenv.load_dotenv()

_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))  # created once

def get_client():
    return _client
