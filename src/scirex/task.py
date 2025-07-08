import ast
import base64
import json
from dataclasses import dataclass
from typing import Any, Optional, Union

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
    input_template: str | None = None  # Template like "{type1} {entry1} is an image..."
    qentries_modality: dict[str, Any] | None = None  # Modality entries
    selected_entries: list[str] | None = None  # Which entries are used
    resolved_content: list[Any] | None = None  # Resolved content for API calls

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

    def resolve_multimodal_content(self) -> list[Any]:
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

    def _parse_base64_image(self, data_url: str) -> Optional[dict[str, Any]]:
        """
        Parse base64 encoded image from data URL.
        Expected format: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...'
        """
        try:
            if not data_url.startswith("data:"):
                return None

            # Split the data URL
            header, data = data_url.split(",", 1)

            # Extract MIME type
            mime_type = header.split(";")[0].split(":")[1]

            # Decode base64 data
            image_bytes = base64.b64decode(data)

            return {"data": image_bytes, "mime_type": mime_type}
        except Exception as e:
            print(f"Error parsing base64 image: {e}")
            return None


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

                # Check if this is a multimodal task
                is_multimodal = "qentries_modality" in example

                if is_multimodal:
                    # Handle multimodal task
                    task = self._create_multimodal_task(row, example)
                else:
                    # Handle traditional text-only task
                    task = self._create_text_task(row, example)

                if task:
                    tasks.append(task)

        return tasks

    def _create_multimodal_task(self, row: dict, example: dict) -> Optional[Task]:
        """Create a multimodal task from the new format."""
        try:
            # Extract basic info
            input_template = example["input"]
            qentries_modality = example["qentries_modality"]
            selected_entries = example.get("selected_entries", [])

            # Determine answer type and target
            if example.get("target") is not None:
                target = example["target"]
                if isinstance(target, str):
                    target = float(target)
                answer_type = "numeric"
            elif example.get("target_scores") is not None:
                target_scores = example["target_scores"]
                if isinstance(target_scores, str):
                    target_scores = json.loads(target_scores)
                target = target_scores
                answer_type = "mcq"
            else:
                print(f"No target found for multimodal task {row['uuid']}")
                return None

            # Create task object
            task = Task(
                uuid=row["uuid"],
                name=row["name"],
                description=row["description"],
                question=input_template,  # Store template as question for now
                answer_type=answer_type,
                target=target,
                keywords=row["keywords"],
                is_multimodal=True,
                input_template=input_template,
                qentries_modality=qentries_modality,
                selected_entries=selected_entries,
            )

            # Resolve the template to create the final question text
            content_parts = task.resolve_multimodal_content()
            # For now, create a text representation (images will be handled separately in model)
            text_parts = [part for part in content_parts if isinstance(part, str)]
            task.question = " ".join(text_parts) if text_parts else input_template

            return task

        except Exception as e:
            print(f"Error creating multimodal task {row['uuid']}: {e}")
            return None

    def _create_text_task(self, row: dict, example: dict) -> Optional[Task]:
        """Create a traditional text-only task (backward compatibility)."""
        try:
            # Clean up example data (following original logic)
            cleaned_example = {"input": example["input"]}

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

            elif example.get("target_scores") is not None:
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
            else:
                print(f"No target found for text task {row['uuid']}")
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

        except Exception as e:
            print(f"Error creating text task {row['uuid']}: {e}")
            return None

    def get_tasks_by_type(self, answer_type: str) -> list[Task]:
        """Get tasks filtered by answer type."""
        return [task for task in self.tasks if task.answer_type == answer_type]

    def get_multimodal_tasks(self) -> list[Task]:
        """Get only multimodal tasks."""
        return [task for task in self.tasks if task.is_multimodal]

    def get_text_tasks(self) -> list[Task]:
        """Get only text-only tasks."""
        return [task for task in self.tasks if not task.is_multimodal]
