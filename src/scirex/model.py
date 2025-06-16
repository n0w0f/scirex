import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv("../../.env")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")


class GeminiModel:
    """Wrapper for Gemini API."""

    def __init__(self, model_name: str = "gemini-2.5-pro-preview-06-05", **config):
        self.client = genai.Client()
        self.model_name = model_name
        self.config = types.GenerateContentConfig(**config)

    def generate(self, prompt: str) -> str:
        """Generate response from Gemini."""
        response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=self.config)
        return response.text
