import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from scirex.response import GeminiResponse

# Load environment variables from .env file
load_dotenv("../../.env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API Key not set.")


class GeminiModel:
    """Wrapper for Gemini API with multimodal support."""

    def __init__(self, model_name: str = "gemini-2.5-pro-preview-06-05", delay: int = 2, **config):
        self.client = genai.Client()
        self.model_name = model_name
        self.config = types.GenerateContentConfig(**config)
        self.delay = delay

    def generate(self, prompt: str | list, return_full_response: bool = False) -> str | GeminiResponse:
        """
        Generate response from Gemini with multimodal support.

        Args:
            prompt: Input prompt - can be:
                   - str: Simple text prompt (backward compatibility)
                   - list: Multimodal content with text and images
            return_full_response: If True, returns GeminiResponse object with all metadata
                                If False, returns just text (default behavior for backward compatibility)

        Returns:
            str if return_full_response=False
            GeminiResponse if return_full_response=True
        """
        # Add delay
        if self.delay > 0:
            import time

            time.sleep(self.delay)

        # Prepare content for API
        content = self._prepare_content(prompt)

        # Generate response
        raw_response = self.client.models.generate_content(model=self.model_name, contents=content, config=self.config)

        if return_full_response:
            return GeminiResponse.from_api_response(raw_response)
        else:
            return raw_response.text

    def _prepare_content(self, prompt: str | list) -> str | list:
        """
        Prepare content for Gemini API.

        Args:
            prompt: String or list of content parts

        Returns:
            Properly formatted content for Gemini API
        """
        if isinstance(prompt, str):
            # Simple text prompt - return as is
            return prompt

        if isinstance(prompt, list):
            # Multimodal content - convert to Gemini API format
            api_content = []

            for part in prompt:
                if isinstance(part, str):
                    # Text part
                    api_content.append(part)
                elif isinstance(part, dict) and part.get("type") == "image":
                    # Image part - convert to types.Part
                    image_part = types.Part.from_bytes(data=part["data"], mime_type=part["mime_type"])
                    api_content.append(image_part)
                else:
                    # Unknown part type - treat as text
                    api_content.append(str(part))

            return api_content

        # Fallback - convert to string
        return str(prompt)

    def generate_multimodal(self, task, prompt_text: str, return_full_response: bool = False) -> str | GeminiResponse:
        """
        Generate response for a multimodal task.

        Args:
            task: Task object with multimodal content
            prompt_text: The prompt template text (e.g., from PromptTemplate)
            return_full_response: Whether to return full response metadata

        Returns:
            Generated response text or GeminiResponse object
        """
        if not task.is_multimodal:
            # Text-only task - use simple prompt
            return self.generate(prompt_text, return_full_response)

        # Get resolved multimodal content
        multimodal_parts = task.resolve_multimodal_content()

        # Combine prompt text with multimodal content
        # The prompt_text typically contains instructions, the multimodal content contains the question
        combined_content = []

        # Add the multimodal content first
        for part in multimodal_parts:
            combined_content.append(part)

        # Add prompt instructions (if they're not already included)
        if prompt_text and not any(prompt_text in str(part) for part in multimodal_parts if isinstance(part, str)):
            combined_content.append(f"\n\n{prompt_text}")

        return self.generate(combined_content, return_full_response)
