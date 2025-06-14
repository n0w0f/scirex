from scirex.task import Task
class PromptTemplate:
    """Configurable prompt templates."""

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
        }

        if custom_templates:
            self.templates.update(custom_templates)

    def format_mcq_prompt(self, task: Task) -> str:
        """Format MCQ prompt."""
        options = []
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for i, (option, score) in enumerate(task.target.items()):
            options.append(f"{letters[i]}. {option}")

        return self.templates["mcq"].format(question=task.question, options="\n".join(options))

    def format_numeric_prompt(self, task: Task) -> str:
        """Format numeric prompt."""
        return self.templates["numeric"].format(question=task.question)

    def format_parse_prompt(self, response: str, answer_type: str) -> str:
        """Format parsing prompt."""
        template_key = f"parse_{answer_type}"
        return self.templates[template_key].format(response=response)
