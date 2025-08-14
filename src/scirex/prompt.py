from typing import Any

from scirex.task import Task


class PromptTemplate:
    """Enhanced configurable prompt templates with exact match support."""

    def __init__(self, custom_templates: dict[str, str] | None = None):
        self.templates = {
            "mcq": """Answer the following multiple choice question.

Question: {question}

Options:
{options}

Instructions: Think through the problem step by step, then provide your final answer in the format \boxed{letter(s)}.
If multiple answers are correct, separate them with commas inside the box.

Answer:""",
            "numeric": """Answer the following question with a numerical value.

Question: {question}

Instructions: Think through the problem step by step, then provide your final numerical answer in the format \boxed{number}. Do not include units in the box.

Answer:""",
            "exact_str_match": """Answer the following question with the exact answer.

Question: {question}

Instructions: Think through the problem step by step, then provide your exact final answer in the format \boxed{answer}.

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
            "parse_exact_str_match": """Extract the exact answer from the following response.

Response: {response}

Instructions: Return only the direct answer, removing any extra explanation or formatting. If no clear answer is found, return 'UNCLEAR'.

Answer:""",
            "mcq_multimodal": """Answer the following multiple choice question.

{content}

Options:
{options}

Instructions: Respond with only the letter(s) of the correct answer(s).
If multiple answers are correct, separate them with commas.

Answer:""",
            "numeric_multimodal": """Answer the following question with a numerical value.

{content}

Instructions: Respond with only the numerical answer. Do not include units.

Answer:""",
            "exact_str_match_multimodal": """Answer the following question with the exact answer.

{content}

Instructions: Provide the exact answer as requested. Be precise and concise.

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

    def format_exact_match_prompt(self, task: Task) -> str | list[Any]:
        """Format exact string match prompt with multimodal support."""
        if task.is_multimodal:
            return self._format_multimodal_exact_match_prompt(task)
        else:
            return self.templates["exact_str_match"].format(question=task.question)

    def _format_multimodal_mcq_prompt(self, task: Task, options: list[str]) -> list[Any]:
        """Format multimodal MCQ prompt."""
        content_parts = task.resolve_multimodal_content()
        prompt_parts = []

        # Add initial instruction
        prompt_parts.append("Answer the following multiple choice question.\n")

        # Add the multimodal content
        prompt_parts.extend(content_parts)

        # Add options
        options_text = f"\n\nOptions:\n{chr(10).join(options)}\n\nInstructions: Respond with only the letter(s) of the correct answer(s).\nIf multiple answers are correct, separate them with commas.\n\nAnswer:"
        prompt_parts.append(options_text)

        return prompt_parts

    def _format_multimodal_numeric_prompt(self, task: Task) -> list[Any]:
        """Format multimodal numeric prompt."""
        content_parts = task.resolve_multimodal_content()
        prompt_parts = []

        # Add initial instruction
        prompt_parts.append("Answer the following question with a numerical value.\n")

        # Add the multimodal content
        prompt_parts.extend(content_parts)

        # Add final instructions
        instruction_text = "\n\nInstructions: Respond with only the numerical answer. Do not include units.\n\nAnswer:"
        prompt_parts.append(instruction_text)

        return prompt_parts

    def _format_multimodal_exact_match_prompt(self, task: Task) -> list[Any]:
        """Format multimodal exact string match prompt."""
        content_parts = task.resolve_multimodal_content()
        prompt_parts = []

        # Add initial instruction
        prompt_parts.append("Answer the following question with the exact answer.\n")

        # Add the multimodal content
        prompt_parts.extend(content_parts)

        # Add final instructions
        instruction_text = "\n\nInstructions: Provide the exact answer as requested. Be precise and concise.\n\nAnswer:"
        prompt_parts.append(instruction_text)

        return prompt_parts

    def format_parse_prompt(self, response: str, answer_type: str) -> str:
        """Format parsing prompt for all answer types."""
        template_key = f"parse_{answer_type}"
        if template_key not in self.templates:
            # Fallback to exact string match for unknown types
            template_key = "parse_exact_str_match"
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
        elif prompt_type == "exact_str_match":
            prompt = self.format_exact_match_prompt(task)
        else:
            return f"Unknown prompt type: {prompt_type}"

        if isinstance(prompt, list):
            # Convert multimodal content to text preview
            return self._flatten_content_for_text(prompt)
        else:
            return prompt

    def get_supported_answer_types(self) -> list[str]:
        """Get list of supported answer types."""
        return ["mcq", "numeric", "exact_str_match"]

    def add_custom_template(self, answer_type: str, template: str, multimodal_template: str = None):
        """Add a custom template for a new answer type."""
        self.templates[answer_type] = template
        if multimodal_template:
            self.templates[f"{answer_type}_multimodal"] = multimodal_template

        # Also add a parse template if not exists
        parse_key = f"parse_{answer_type}"
        if parse_key not in self.templates:
            self.templates[parse_key] = f"""Extract the answer from the following response.

Response: {{response}}

Instructions: Return only the answer. If no clear answer is found, return 'UNCLEAR'.

Answer:"""
