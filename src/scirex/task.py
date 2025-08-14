import ast
import base64
import json
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Union

import PIL.Image
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


def pil_image_to_base64(image: PIL.Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    base64_string = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{format.lower()};base64,{base64_string}"


@dataclass
class Task:
    """Enhanced task representation with template and multimodal support."""

    uuid: str
    name: str
    description: str
    question: str  # For backward compatibility and resolved content
    answer_type: str  # "mcq", "numeric", or "exact_str_match"
    target: Union[float, dict[str, float], str]  # target value, target_scores for MCQ, or exact answer string
    keywords: list[str]

    # Original multimodal fields (for backward compatibility)
    is_multimodal: bool = False
    input_template: str | None = None
    qentries_modality: dict[str, Any] | None = None
    selected_entries: list[str] | None = None
    resolved_content: list[Any] | None = None

    # New template-based fields
    question_template: str | None = None
    template_input: dict[str, Any] | None = None
    image_data: str | None = None  # Base64 encoded image data
    preferred_score: str | None = None

    def get_text_content(self) -> str:
        """Get the text-only version of the content for parsing."""
        if not self.is_multimodal:
            return self.question

        if self.question_template and self.template_input:
            # New template-based format
            return self._resolve_template_text()
        else:
            # Original format
            content_parts = self.resolve_multimodal_content()
            text_parts = []
            for part in content_parts:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "image":
                    text_parts.append("[IMAGE]")

            return " ".join(text_parts)

    def _resolve_template_text(self) -> str:
        """Resolve template placeholders with text-only content."""
        if not self.question_template or not self.template_input:
            return self.question

        resolved_text = self.question_template

        # Replace placeholders
        for key, value in self.template_input.items():
            placeholder = f"{{{{{key}}}}}"
            if key == "entry1" and self.image_data:
                # Replace image placeholder with text description
                resolved_text = resolved_text.replace(placeholder, "[IMAGE]")
            else:
                resolved_text = resolved_text.replace(placeholder, str(value))

        return resolved_text

    def resolve_multimodal_content(self) -> list[Any]:
        """
        Resolve template placeholders with actual content.
        Returns list of content parts suitable for API calls.
        """
        if not self.is_multimodal:
            return [self.question]

        if self.resolved_content is not None:
            return self.resolved_content

        # Handle new template-based format
        if self.question_template and self.template_input:
            return self._resolve_template_multimodal()

        # Handle original format
        elif self.input_template and self.qentries_modality:
            return self._resolve_legacy_multimodal()

        return [self.question]

    def _resolve_template_multimodal(self) -> list[Any]:
        """Resolve new template-based multimodal content."""
        if not self.question_template or not self.template_input:
            return [self.question]

        content_parts = []
        template = self.question_template

        # Find all placeholders
        placeholders = re.findall(r"\{\{(\w+)\}\}", template)

        # Track processed placeholders to avoid double processing
        processed = set()
        current_template = template

        for placeholder in placeholders:
            if placeholder in processed:
                continue

            placeholder_pattern = f"{{{{{placeholder}}}}}"

            if placeholder == "entry1" and self.image_data:
                # Split at image placeholder
                parts = current_template.split(placeholder_pattern, 1)
                if len(parts) == 2:
                    before, after = parts

                    # Add text before image
                    if before.strip():
                        content_parts.append(before.strip())

                    # Add image
                    content_parts.append({"type": "image", "data": self.image_data})

                    current_template = after
                    processed.add(placeholder)
            elif placeholder in self.template_input:
                # Replace text placeholder
                value = str(self.template_input[placeholder])
                current_template = current_template.replace(placeholder_pattern, value)
                processed.add(placeholder)

        # Add remaining text
        if current_template.strip():
            content_parts.append(current_template.strip())

        # If no multimodal content was found, return as text
        if not any(isinstance(part, dict) for part in content_parts):
            return [" ".join(str(part) for part in content_parts)]

        self.resolved_content = content_parts
        return content_parts

    def _resolve_legacy_multimodal(self) -> list[Any]:
        """Resolve original multimodal format (for backward compatibility)."""
        # This is the existing logic from the original code
        resolved_text = self.input_template
        content_parts = []
        placeholders = re.findall(r"\{(\w+)\}", self.input_template)
        image_placeholders = {}

        for _category, entries in self.qentries_modality.items():
            for placeholder, entry_data in entries.items():
                if placeholder in placeholders:
                    entry_type = entry_data.get("type", "text")
                    entry_value = entry_data.get("value", "")

                    if entry_type == "text":
                        resolved_text = resolved_text.replace(f"{{{placeholder}}}", entry_value)
                    elif entry_type == "image":
                        image_placeholders[placeholder] = entry_value
                        resolved_text = resolved_text.replace(f"{{{placeholder}}}", f"[IMAGE_{placeholder}]")

        if image_placeholders:
            parts = []
            current_text = resolved_text

            for placeholder, base64_data in image_placeholders.items():
                marker = f"[IMAGE_{placeholder}]"
                if marker in current_text:
                    before, after = current_text.split(marker, 1)
                    if before.strip():
                        parts.append(before.strip())
                    parts.append({"type": "image", "data": base64_data})
                    current_text = after

            if current_text.strip():
                parts.append(current_text.strip())
            content_parts = parts
        else:
            content_parts = [resolved_text]

        self.resolved_content = content_parts
        return content_parts


class Dataset:
    """Loads and manages benchmark tasks with enhanced multimodal support."""

    def __init__(self, dataset_name: str, subset: str | None = None):
        self.dataset_name = dataset_name
        self.subset = subset
        self.tasks = self._load_tasks()

    def _detect_schema_format(self, row: dict) -> str:
        """Detect which schema format the dataset uses."""
        if "examples" in row:
            return "legacy"  # Original format with examples array
        elif "question_template" in row and "answer" in row:
            return "template"  # New template-based format
        else:
            return "unknown"

    def _load_tasks(self) -> list[Task]:
        """Load tasks from HuggingFace dataset with enhanced multimodal support."""
        dataset = load_dataset(self.dataset_name, self.subset)["train"]
        tasks = []

        for row in dataset:
            # Detect schema format
            schema_format = self._detect_schema_format(row)

            if len(tasks) == 0:
                logger.info(f"Detected schema format: {schema_format}")
                logger.info(f"First row keys: {list(row.keys())}")

            if schema_format == "legacy":
                # Handle original format
                tasks.extend(self._parse_legacy_format(row))
            elif schema_format == "template":
                # Handle new template-based format
                task = self._parse_template_format(row)
                if task:
                    tasks.append(task)
            else:
                logger.warning(f"Unknown schema format for row {row.get('uuid', 'unknown')}")

        logger.info(f"Loaded {len(tasks)} tasks successfully")
        return tasks

    def _parse_legacy_format(self, row: dict) -> list[Task]:
        """Parse original format with examples array."""
        tasks = []

        for example in row["examples"]:
            if isinstance(example, str):
                example = parse_json_like_string(example)
                if example is None:
                    continue

            # Detect if multimodal
            is_multimodal = (
                isinstance(example, dict)
                and "qentries_modality" in example
                and example["qentries_modality"] is not None
            )

            if is_multimodal:
                multimodal_tasks = self._parse_multimodal_task(example, row)
                tasks.extend(multimodal_tasks)
            else:
                task = self._parse_text_task(example, row)
                if task:
                    tasks.append(task)

        return tasks

    def _parse_template_format(self, row: dict) -> Task | None:
        """Parse new template-based format."""
        try:
            # Parse template input
            template_input = {}
            if row.get("question_template_input"):
                template_input = parse_json_like_string(row["question_template_input"])
                if template_input is None:
                    logger.warning(f"Failed to parse template_input for {row.get('row_uuid', 'unknown')}")
                    return None

            # Determine answer type based on preferred_score
            preferred_score = row.get("preferred_score", "")
            if preferred_score == "exact_str_match":
                answer_type = "exact_str_match"
                target = row["answer"]
            else:
                # Try to determine if it's numeric or MCQ based on the answer
                answer = row["answer"]
                try:
                    target = float(answer)
                    answer_type = "numeric"
                except (ValueError, TypeError):
                    # Assume it's exact string match for non-numeric answers
                    answer_type = "exact_str_match"
                    target = str(answer)

            # Handle image data
            image_data = None
            is_multimodal = False
            if "image" in row and row["image"] is not None:
                is_multimodal = True
                # Convert PIL image to base64
                if hasattr(row["image"], "save"):  # Check if it's a PIL Image
                    image_data = pil_image_to_base64(row["image"])

            # Resolve question text
            question_text = row.get("question_template", "")
            if template_input:
                # Create a temporary resolved version for backward compatibility
                temp_question = question_text
                for key, value in template_input.items():
                    placeholder = f"{{{{{key}}}}}"
                    if key == "entry1" and image_data:
                        temp_question = temp_question.replace(placeholder, "[IMAGE]")
                    else:
                        temp_question = temp_question.replace(placeholder, str(value))
                question_text = temp_question

            return Task(
                uuid=row.get("row_uuid", ""),
                name=row.get("dataset_name", ""),
                description=row.get("dataset_description", ""),
                question=question_text,
                answer_type=answer_type,
                target=target,
                keywords=row.get("keywords", []),
                is_multimodal=is_multimodal,
                question_template=row.get("question_template"),
                template_input=template_input,
                image_data=image_data,
                preferred_score=preferred_score,
            )

        except Exception as e:
            logger.error(f"Error creating template-based task {row.get('row_uuid', 'unknown')}: {e}")
            return None

    def _parse_multimodal_task(self, example: dict, row: dict) -> list[Task]:
        """Parse multimodal task from original format (for backward compatibility)."""
        # This is the existing logic from the original code
        tasks = []
        try:
            input_template = example["input"]
            qentries_modality = example["qentries_modality"]
            selected_entries = example.get("selected_entries", [])

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

            task = Task(
                uuid=row["uuid"],
                name=row["name"],
                description=row["description"],
                question="",
                answer_type=answer_type,
                target=target,
                keywords=row["keywords"],
                is_multimodal=True,
                input_template=input_template,
                qentries_modality=qentries_modality,
                selected_entries=selected_entries,
            )

            task.question = task.get_text_content()
            tasks.append(task)

        except Exception as e:
            logger.info(f"Error creating multimodal task {row['uuid']}: {e}")

        return tasks

    def _parse_text_task(self, example: dict, row: dict) -> Task | None:
        """Parse traditional text-only task (backward compatibility)."""
        try:
            cleaned_example = {"input": example["input"]}

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

    def get_exact_match_tasks(self) -> list[Task]:
        """Get only exact string match tasks."""
        return [task for task in self.tasks if task.answer_type == "exact_str_match"]
