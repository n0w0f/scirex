import ast
import json
from dataclasses import dataclass

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
    """Simple task representation."""

    uuid: str
    name: str
    description: str
    question: str
    answer_type: str  # "mcq" or "numeric"
    target: float | dict[str, float]  # target value or target_scores for MCQ
    keywords: list[str]


class Dataset:
    """Loads and manages benchmark tasks."""

    def __init__(self, dataset_name: str, subset: str | None = None):
        self.dataset_name = dataset_name
        self.subset = subset
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> list[Task]:
        """Load tasks from HuggingFace dataset."""
        dataset = load_dataset(self.dataset_name, self.subset)["train"]
        tasks = []

        for row in dataset:
            for example in row["examples"]:
                # Parse example if it's a string (like in your old code)
                if isinstance(example, str):
                    example = parse_json_like_string(example)

                # Clean up example data (following your old logic)
                cleaned_example = {"input": example["input"]}

                # Handle target_scores and target with proper JSON parsing
                if example.get("target") is not None:
                    target = example["target"]
                    if isinstance(target, str):
                        try:
                            target = float(target)
                        except ValueError:
                            print(f"Failed to parse numeric target: {target}")
                            continue
                    cleaned_example["target"] = target
                    answer_type = "numeric"

                if example.get("target_scores") is not None:
                    target_scores = example["target_scores"]
                    if isinstance(target_scores, str):
                        try:
                            target_scores = json.loads(target_scores)  # Parse JSON string
                        except json.JSONDecodeError:
                            print(f"Failed to parse target_scores: {target_scores}")
                            continue
                    cleaned_example["target_scores"] = target_scores
                    answer_type = "mcq"
                    target = target_scores

                task = Task(
                    uuid=row["uuid"],
                    name=row["name"],
                    description=row["description"],
                    question=cleaned_example["input"],
                    answer_type=answer_type,
                    target=target,
                    keywords=row["keywords"],
                )
                tasks.append(task)

        return tasks

    def get_tasks_by_type(self, answer_type: str) -> list[Task]:
        """Get tasks filtered by answer type."""
        return [task for task in self.tasks if task.answer_type == answer_type]
