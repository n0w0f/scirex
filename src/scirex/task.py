import ast
import base64
import json
import re
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from loguru import logger


def parse_json_like_string(s: str) -> dict:
    """
    Convert a JSON-like string with single quotes to valid JSON and parse it.
    """
    try:
        # First try ast.literal_eval for Python literals
        python_dict = ast.literal_eval(s)
        json_string = json.dumps(python_dict)
        return json.loads(json_string)
    except (SyntaxError, ValueError, json.JSONDecodeError):
        try:
            # Try direct JSON parsing
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                # Try replacing single quotes with double quotes
                s_fixed = s.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
                return json.loads(s_fixed)
            except json.JSONDecodeError as e:
                logger.info(f"Error parsing string: {e}")
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
        placeholders = re.findall(r"\{(\w+)\}", self.input_template)

        # Track which parts need to be replaced with images
        image_placeholders = {}

        # Process each modality category
        for _category, entries in self.qentries_modality.items():
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

    def _parse_base64_image(self, data_url: str) -> dict[str, Any] | None:
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
            logger.info(f"Error parsing base64 image: {e}")
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
            # Debug: logger.info the first few examples to understand the format
            if len(tasks) == 0:
                logger.info(f"First row keys: {list(row.keys())}")
                if row["examples"]:
                    logger.info(f"First example preview: {str(row['examples'][0])[:100]}...")

            for example in row["examples"]:
                original_example = example

                # Handle different example formats
                if isinstance(example, str):
                    # Parse string examples (some datasets have this format)
                    example = parse_json_like_string(example)
                    if example is None:
                        logger.info(f"Skipping failed parse for row {row.get('uuid', 'unknown')}")
                        continue
                elif isinstance(example, dict):
                    pass
                else:
                    # Skip if example is not a string or dict
                    logger.info(f"Skipping unknown example type: {type(example)} - {str(original_example)[:100]}...")
                    continue

                # Now example should be a valid dictionary
                # Detect if this is a multimodal task
                is_multimodal = (
                    isinstance(example, dict)
                    and "qentries_modality" in example
                    and example["qentries_modality"] is not None
                )

                if is_multimodal:
                    # Handle multimodal format
                    multimodal_tasks = self._parse_multimodal_task(example, row)
                    tasks.extend(multimodal_tasks)
                else:
                    # Handle original text-only format
                    task = self._parse_text_task(example, row)
                    if task:
                        tasks.append(task)

        logger.info(f"Loaded {len(tasks)} tasks successfully")
        return tasks

    def _parse_multimodal_task(self, example: dict, row: dict) -> list[Task]:
        """Parse a multimodal task from the new format."""
        tasks = []

        try:
            # Extract basic info
            input_template = example["input"]
            qentries_modality = example["qentries_modality"]
            selected_entries = example.get("selected_entries", [])

            # Determine answer type and target
            if example.get("target") is not None:
                target = example["target"]
                if isinstance(target, str):
                    try:
                        target = float(target)
                    except ValueError:
                        logger.info(f"Failed to parse numeric target: {target}")
                        return []
                answer_type = "numeric"
            elif example.get("target_scores") is not None:
                target_scores = example["target_scores"]
                if isinstance(target_scores, str):
                    try:
                        target_scores = json.loads(target_scores)
                    except json.JSONDecodeError:
                        logger.info(f"Failed to parse target_scores: {target_scores}")
                        return []
                target = target_scores
                answer_type = "mcq"
            else:
                logger.info(f"No target found for multimodal task {row['uuid']}")
                return []

            # Create task object
            task = Task(
                uuid=row["uuid"],
                name=row["name"],
                description=row["description"],
                question="",  # Will be filled below
                answer_type=answer_type,
                target=target,
                keywords=row["keywords"],
                is_multimodal=True,
                input_template=input_template,
                qentries_modality=qentries_modality,
                selected_entries=selected_entries,
            )

            # Resolve the template to create the final question text
            task.question = task.get_text_content()

            tasks.append(task)

        except Exception as e:
            logger.info(f"Error creating multimodal task {row['uuid']}: {e}")

        return tasks

    def _parse_text_task(self, example: dict, row: dict) -> Task | None:
        """Parse a traditional text-only task (backward compatibility)."""
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
                        logger.info(f"Failed to parse numeric target: {target}")
                        return None
                cleaned_example["target"] = target
                answer_type = "numeric"

            elif example.get("target_scores") is not None:
                target_scores = example["target_scores"]
                if isinstance(target_scores, str):
                    try:
                        target_scores = json.loads(target_scores)
                    except json.JSONDecodeError:
                        logger.info(f"Failed to parse target_scores: {target_scores}")
                        return None
                cleaned_example["target_scores"] = target_scores
                answer_type = "mcq"
                target = target_scores
            else:
                logger.info(f"No target found for text task {row['uuid']}")
                return None

            return Task(
                uuid=row["uuid"],
                name=row["name"],
                description=row["description"],
                question=cleaned_example["input"],
                answer_type=answer_type,
                target=target,
                keywords=row["keywords"],
                is_multimodal=False,
            )

        except Exception as e:
            logger.info(f"Error creating text task {row['uuid']}: {e}")
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
