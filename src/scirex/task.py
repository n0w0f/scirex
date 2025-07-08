import ast
import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


def parse_json_like_string(s: str) -> dict:
    """
    Convert a JSON-like string with single quotes to valid JSON and parse it.
    """
    try:
        python_dict = ast.literal_eval(s)
        json_string = json.dumps(python_dict)
        return json.loads(json_string)
    except (SyntaxError, ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing string: {e}")
        return None


@dataclass
class Task:
    """Simple task representation with multimodal support."""

    uuid: str
    name: str
    description: str
    question: str  # For backward compatibility and resolved multimodal content
    answer_type: str  # "mcq" or "numeric"
    target: float | dict[str, float]  # target value or target_scores for MCQ
    keywords: list[str]

    # New multimodal fields
    is_multimodal: bool = False
    input_template: Optional[str] = None  # Template like "{type1} {entry1} is an image..."
    qentries_modality: Optional[Dict[str, Any]] = None  # Modality entries
    selected_entries: Optional[List[str]] = None  # Which entries are used
    resolved_content: Optional[List[Any]] = None  # Resolved content for API calls

    def resolve_multimodal_content(self) -> List[Any]:
        """
        Resolve template placeholders with actual content.
        Returns list of content parts suitable for Gemini API.
        """
        if not self.is_multimodal or not self.input_template or not self.qentries_modality:
            return [self.question]

        if self.resolved_content is not None:
            return self.resolved_content

        # Start with the template
        resolved_text = self.input_template
        content_parts = []

        # Find all placeholders in the template
        import re

        placeholders = re.findall(r"\{(\w+)\}", self.input_template)

        # Track which parts need to be replaced with images
        image_placeholders = {}

        # Process each modality category
        for category, entries in self.qentries_modality.items():
            for placeholder, entry_data in entries.items():
                if placeholder in placeholders:
                    entry_type = entry_data.get("type", "text")
                    entry_value = entry_data.get("value", "")

                    if entry_type == "text":
                        # Replace text placeholders directly
                        resolved_text = resolved_text.replace(f"{{{placeholder}}}", entry_value)
                    elif entry_type == "image":
                        # Mark image placeholders for special handling
                        image_placeholders[placeholder] = entry_value
                        # For now, replace with a marker
                        resolved_text = resolved_text.replace(f"{{{placeholder}}}", f"[IMAGE_{placeholder}]")

        # If we have images, we need to create a mixed content list
        if image_placeholders:
            # Split text by image markers and create content parts
            parts = []
            current_text = resolved_text

            for placeholder, base64_data in image_placeholders.items():
                marker = f"[IMAGE_{placeholder}]"
                if marker in current_text:
                    # Split at the marker
                    before, after = current_text.split(marker, 1)

                    # Add text before image (if any)
                    if before.strip():
                        parts.append(before.strip())

                    # Add image part
                    parts.append({"type": "image", "data": base64_data})

                    current_text = after

            # Add remaining text (if any)
            if current_text.strip():
                parts.append(current_text.strip())

            content_parts = parts
        else:
            # No images, just return the resolved text
            content_parts = [resolved_text]

        self.resolved_content = content_parts
        return content_parts

    def get_text_content(self) -> str:
        """Get the text-only version of the content for parsing."""
        if not self.is_multimodal:
            return self.question

        content_parts = self.resolve_multimodal_content()
        # Extract only text parts
        text_parts = []
        for part in content_parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "image":
                text_parts.append("[IMAGE]")  # Placeholder for image in text

        return " ".join(text_parts)


class Dataset:
    """Loads and manages benchmark tasks with multimodal support."""

    def __init__(self, dataset_name: str, subset: str | None = None):
        self.dataset_name = dataset_name
        self.subset = subset
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> list[Task]:
        """Load tasks from HuggingFace dataset with multimodal support."""
        dataset = load_dataset(self.dataset_name, self.subset)["train"]
        tasks = []

        for row in dataset:
            for example in row["examples"]:
                # Parse example if it's a string
                if isinstance(example, str):
                    example = parse_json_like_string(example)

                if example is None:
                    continue

                # Detect if this is a multimodal task
                is_multimodal = "qentries_modality" in example

                if is_multimodal:
                    # Handle multimodal format
                    tasks.extend(self._parse_multimodal_task(example, row))
                else:
                    # Handle original text-only format
                    task = self._parse_text_task(example, row)
                    if task:
                        tasks.append(task)

        return tasks

    def _parse_multimodal_task(self, example: dict, row: dict) -> List[Task]:
        """Parse a multimodal task."""
        tasks = []

        input_template = example.get("input", "")
        qentries_modality = example.get("qentries_modality", {})
        selected_entries = example.get("selected_entries", [])

        # Create a Task object
        task_data = {
            "uuid": row["uuid"],
            "name": row["name"],
            "description": row["description"],
            "keywords": row["keywords"],
            "is_multimodal": True,
            "input_template": input_template,
            "qentries_modality": qentries_modality,
            "selected_entries": selected_entries,
        }

        # Handle target_scores and target
        answer_type = None
        target = None

        if example.get("target") is not None:
            target = example["target"]
            if isinstance(target, str):
                try:
                    target = float(target)
                except ValueError:
                    print(f"Failed to parse numeric target: {target}")
                    return []
            answer_type = "numeric"

        if example.get("target_scores") is not None:
            target_scores = example["target_scores"]
            if isinstance(target_scores, str):
                try:
                    target_scores = json.loads(target_scores)
                except json.JSONDecodeError:
                    print(f"Failed to parse target_scores: {target_scores}")
                    return []
            answer_type = "mcq"
            target = target_scores

        if answer_type is None or target is None:
            print(f"Could not determine answer type or target for multimodal task")
            return []

        task_data["answer_type"] = answer_type
        task_data["target"] = target

        # Create a temporary task to resolve content for the question field
        temp_task = Task(question="", **task_data)
        resolved_text = temp_task.get_text_content()
        task_data["question"] = resolved_text

        task = Task(**task_data)
        tasks.append(task)

        return tasks

    def _parse_text_task(self, example: dict, row: dict) -> Optional[Task]:
        """Parse a text-only task (original format)."""
        cleaned_example = {"input": example["input"]}
        answer_type = None
        target = None

        # Handle target_scores and target with proper JSON parsing
        if example.get("target") is not None:
            target = example["target"]
            if isinstance(target, str):
                try:
                    target = float(target)
                except ValueError:
                    print(f"Failed to parse numeric target: {target}")
                    return None
            cleaned_example["target"] = target
            answer_type = "numeric"

        if example.get("target_scores") is not None:
            target_scores = example["target_scores"]
            if isinstance(target_scores, str):
                try:
                    target_scores = json.loads(target_scores)
                except json.JSONDecodeError:
                    print(f"Failed to parse target_scores: {target_scores}")
                    return None
            cleaned_example["target_scores"] = target_scores
            answer_type = "mcq"
            target = target_scores

        if answer_type is None or target is None:
            return None

        task = Task(
            uuid=row["uuid"],
            name=row["name"],
            description=row["description"],
            question=cleaned_example["input"],
            answer_type=answer_type,
            target=target,
            keywords=row["keywords"],
            is_multimodal=False,
        )

        return task

    def get_tasks_by_type(self, answer_type: str) -> list[Task]:
        """Get tasks filtered by answer type."""
        return [task for task in self.tasks if task.answer_type == answer_type]

    def get_multimodal_tasks(self) -> list[Task]:
        """Get only multimodal tasks."""
        return [task for task in self.tasks if task.is_multimodal]

    def get_text_tasks(self) -> list[Task]:
        """Get only text-only tasks."""
        return [task for task in self.tasks if not task.is_multimodal]
