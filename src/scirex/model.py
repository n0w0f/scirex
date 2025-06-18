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

    def generate(self, prompt: str, return_thoughts: bool = False) -> str | tuple[str, str | None]:
        """
        Generate response from Gemini.

        Args:
            prompt: Input prompt
            return_thoughts: If True, returns (text, thought_summary) tuple
                           If False, returns just text (default behavior)

        Returns:
            str if return_thoughts=False
            tuple[str, str | None] if return_thoughts=True
        """
        response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=self.config)

        # Handle responses that might contain thought parts
        if hasattr(response.candidates[0].content, "parts") and len(response.candidates[0].content.parts) > 1:
            # Multiple parts - extract text and thoughts
            text_content = ""
            thought_summary = None

            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if hasattr(part, "thought") and part.thought:
                    thought_summary = part.text
                else:
                    text_content = part.text

            if return_thoughts:
                return text_content, thought_summary
            else:
                return text_content
        else:
            # Single part response (standard case)
            text_content = response.text
            if return_thoughts:
                return text_content, None
            else:
                return text_content
