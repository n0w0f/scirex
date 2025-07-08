from typing import Any

from scirex.task import Task


class PromptTemplate:
    """Configurable prompt templates with multimodal support."""

    def __init__(self, custom_templates: dict[str, str] | None = None):
        self.templates = {
            "mcq": """Answer the following multiple choice question about chemistry.

Question: {question}

Options:
{options}

Instructions: Respond with only the letter(s) of the correct answer(s).
If multiple answers are correct, separate them with commas.

Answer:""",
            "numeric": """Answer the following chemistry question with a numerical value.

Question: {question}

Instructions: Respond with only the numerical answer. Do not include units.

Answer:""",
            "parse_mcq": """Extract the answer from the following response to a multiple choice question.
The answer should be one or more letters (A, B, C, etc.) separated by commas if multiple.

Response: {response}

Instructions: Return only the letter(s) that represent the answer. If no clear answer is found, return 'UNCLEAR'.

Answer:""",
            "parse_numeric": """Extract the numerical answer from the following response.

Response: {response}

Instructions: Return only the number. If no clear numerical answer is found, return 'UNCLEAR'.

Answer:""",
            # New multimodal templates
            "mcq_multimodal": """Answer the following multiple choice question about chemistry.

{content}

Options:
{options}

Instructions: Respond with only the letter(s) of the correct answer(s).
If multiple answers are correct, separate them with commas.

Answer:""",
            "numeric_multimodal": """Answer the following chemistry question with a numerical value.

{content}

Instructions: Respond with only the numerical answer. Do not include units.

Answer:""",
        }

        if custom_templates:
            self.templates.update(custom_templates)

    def format_mcq_prompt(self, task: Task) -> str | list[Any]:
        """Format MCQ prompt with multimodal support."""
        options = []
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for i, (option, _score) in enumerate(task.target.items()):
            options.append(f"{letters[i]}. {option}")

        if task.is_multimodal:
            return self._format_multimodal_mcq_prompt(task, options)
        else:
            return self.templates["mcq"].format(question=task.question, options="\n".join(options))

    def format_numeric_prompt(self, task: Task) -> str | list[Any]:
        """Format numeric prompt with multimodal support."""
        if task.is_multimodal:
            return self._format_multimodal_numeric_prompt(task)
        else:
            return self.templates["numeric"].format(question=task.question)

    def _format_multimodal_mcq_prompt(self, task: Task, options: list[str]) -> list[Any]:
        """Format multimodal MCQ prompt."""
        # Get the resolved multimodal content
        content_parts = task.resolve_multimodal_content()

        # Create the prompt structure
        prompt_parts = []

        # Add initial instruction
        prompt_parts.append("Answer the following multiple choice question about chemistry.\n")

        # Add the multimodal content
        prompt_parts.extend(content_parts)

        # Add options
        options_text = f"\n\nOptions:\n{chr(10).join(options)}\n\nInstructions: Respond with only the letter(s) of the correct answer(s).\nIf multiple answers are correct, separate them with commas.\n\nAnswer:"
        prompt_parts.append(options_text)

        return prompt_parts

    def _format_multimodal_numeric_prompt(self, task: Task) -> list[Any]:
        """Format multimodal numeric prompt."""
        # Get the resolved multimodal content
        content_parts = task.resolve_multimodal_content()

        # Create the prompt structure
        prompt_parts = []

        # Add initial instruction
        prompt_parts.append("Answer the following chemistry question with a numerical value.\n")

        # Add the multimodal content
        prompt_parts.extend(content_parts)

        # Add final instructions
        instruction_text = "\n\nInstructions: Respond with only the numerical answer. Do not include units.\n\nAnswer:"
        prompt_parts.append(instruction_text)

        return prompt_parts

    def format_parse_prompt(self, response: str, answer_type: str) -> str:
        """Format parsing prompt (remains text-only)."""
        template_key = f"parse_{answer_type}"
        return self.templates[template_key].format(response=response)

    def _flatten_content_for_text(self, content_parts: list[Any]) -> str:
        """
        Flatten multimodal content to text-only for parsing prompts.
        Used when we need a text representation of multimodal content.
        """
        text_parts = []
        for part in content_parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "image":
                text_parts.append("[IMAGE]")  # Placeholder for image
            else:
                text_parts.append(str(part))

        return " ".join(text_parts)

    def get_content_for_parsing(self, task: Task) -> str:
        """
        Get text content suitable for response parsing.
        For multimodal tasks, returns a text representation.
        """
        if task.is_multimodal:
            return task.get_text_content()
        else:
            return task.question

    def preview_prompt(self, task: Task, prompt_type: str = None) -> str:
        """
        Generate a text preview of what the prompt would look like.
        Useful for debugging and visualization.
        """
        if prompt_type is None:
            prompt_type = task.answer_type

        if prompt_type == "mcq":
            prompt = self.format_mcq_prompt(task)
        elif prompt_type == "numeric":
            prompt = self.format_numeric_prompt(task)
        else:
            return f"Unknown prompt type: {prompt_type}"

        if isinstance(prompt, list):
            # Convert multimodal content to text preview
            return self._flatten_content_for_text(prompt)
        else:
            return prompt
