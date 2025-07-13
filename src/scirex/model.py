import base64
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from scirex.response import GeminiResponse

load_dotenv()


class APIKeyError(ValueError):
    def __init__(self):
        super().__init__("Google API key must be provided or set in the environment as GOOGLE_API_KEY.")


class GeminiModel:
    """Wrapper for Gemini API with multimodal support."""

    def __init__(
        self, model_name: str = "gemini-2.5-pro-preview-06-05", api_key: str | None = None, delay: int = 2, **config
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise APIKeyError()
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

        # Prepare contents using ternary operator
        contents = self._prepare_multimodal_content(prompt) if isinstance(prompt, list) else prompt

        # Generate response
        raw_response = self.client.models.generate_content(model=self.model_name, contents=contents, config=self.config)

        if return_full_response:
            return GeminiResponse.from_api_response(raw_response)
        else:
            return raw_response.text

    def _base64_to_image_part(self, base64_data: str, mime_type: str = "image/png"):
        # Remove data URL prefix if present
        if base64_data.startswith("data:"):
            _, base64_content = base64_data.split(",", 1)
            data_prefix = base64_data.split(",")[0]
            if ";" in data_prefix and ":" in data_prefix:
                mime_type = data_prefix.split(";")[0].split(":", 1)[1]
        else:
            base64_content = base64_data

        image_bytes = base64.b64decode(base64_content)
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    def _prepare_multimodal_content(self, content_parts: list[Any]):
        prepared_content = []
        for part in content_parts:
            if isinstance(part, str):
                prepared_content.append(part)
            elif isinstance(part, dict) and part.get("type") == "image":
                base64_data = part.get("data", "")
                image_part = self._base64_to_image_part(base64_data)
                prepared_content.append(image_part)
            else:
                prepared_content.append(str(part))
        return prepared_content

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

    def test_multimodal_capability(self) -> bool:
        """
        Test if the model supports multimodal input.

        Returns:
            True if multimodal is supported, False otherwise
        """
        try:
            # Create a small test image (1x1 red pixel PNG)
            test_base64 = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            )

            # Decode base64 to bytes
            import base64

            image_bytes = base64.b64decode(test_base64)

            # Create image part
            test_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

            # Try a simple multimodal request
            test_content = [test_part, "What do you see?"]
            response = self.client.models.generate_content(
                model=self.model_name, contents=test_content, config=self.config
            )

            return len(response.text.strip()) > 0
        except Exception as e:
            print(f"Multimodal test failed: {e}")
            return False
