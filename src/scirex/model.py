from google import genai
from google.genai import types


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
