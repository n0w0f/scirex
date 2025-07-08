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
    input_template: Optional[str] = None  # Template with placeholders like "{type1} {entry1}"
    qentries_modality: Optional[dict[str, Any]] = None  # Modality entries
    selected_entries: Optional[list[str]] = None  # Which entries are used

    def resolve_multimodal_content(self) -> list[Union[str, dict[str, Any]]]:
        """
        Resolve multimodal content for API consumption.
        Returns list of content parts (strings for text, dicts for images).
        """
        if not self.is_multimodal:
            return [self.question]

        # Start with the input template
        resolved_template = self.input_template
        image_parts = []

        # Process each selected entry
        if self.qentries_modality and self.selected_entries:
            for entry_group in self.selected_entries:
                if entry_group in self.qentries_modality:
                    entries = self.qentries_modality[entry_group]

                    # Replace placeholders in template
                    for placeholder, entry_data in entries.items():
                        placeholder_key = f"{{{placeholder}}}"

                        if entry_data["type"] == "text":
                            # Replace text placeholders directly
                            resolved_template = resolved_template.replace(placeholder_key, entry_data["value"])
                        elif entry_data["type"] == "image":
                            # For images, replace with a marker and collect image data
                            image_marker = f"[IMAGE_{len(image_parts)}]"
                            resolved_template = resolved_template.replace(placeholder_key, image_marker)

                            # Parse base64 image
                            image_data = self._parse_base64_image(entry_data["value"])
                            if image_data:
                                image_parts.append({
                                    "marker": image_marker,
                                    "data": image_data["data"],
                                    "mime_type": image_data["mime_type"],
                                })

        # Build final content list
        content_parts = []

        if image_parts:
            # Split text by image markers and interleave with images
            text_parts = resolved_template
            for image_part in image_parts:
                if image_part["marker"] in text_parts:
                    before, after = text_parts.split(image_part["marker"], 1)
                    if before.strip():
                        content_parts.append(before.strip())
                    content_parts.append({
                        "type": "image",
                        "data": image_part["data"],
                        "mime_type": image_part["mime_type"],
                    })
                    text_parts = after

            # Add remaining text
            if text_parts.strip():
                content_parts.append(text_parts.strip())
        else:
            # No images, just return the resolved text
            content_parts.append(resolved_template)

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
