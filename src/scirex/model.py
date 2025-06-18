import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from scirex.response import GeminiResponse

# Load environment variables from .env file
load_dotenv("/Users/n0w0f/git/n0w0f_2025/SciResearchBench/.env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API Key not set.")


class GeminiModel:
    """Wrapper for Gemini API."""

    def __init__(self, model_name: str = "gemini-2.5-pro-preview-06-05", **config):
        self.client = genai.Client()
        self.model_name = model_name
        self.config = types.GenerateContentConfig(**config)

    def generate(self, prompt: str, return_full_response: bool = False) -> str | GeminiResponse:
        """
        Generate response from Gemini.

        Args:
            prompt: Input prompt
            return_full_response: If True, returns GeminiResponse object with all metadata
                                If False, returns just text (default behavior for backward compatibility)

        Returns:
            str if return_full_response=False
            GeminiResponse if return_full_response=True
        """
        raw_response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=self.config)

        if return_full_response:
            return GeminiResponse.from_api_response(raw_response)
        else:
            return raw_response.text
