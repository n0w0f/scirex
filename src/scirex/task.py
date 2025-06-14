from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class Task:
    """Simple task representation."""

    uuid: str
    name: str
    description: str
    question: str
    answer_type: str  # "mcq" or "numeric"
    target: float | dict[str, float] # target value or target_scores for MCQ
    keywords: list[str]


class Dataset:
    """Loads and manages benchmark tasks."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> list[Task]:
        """Load tasks from HuggingFace dataset."""
        dataset = load_dataset(self.dataset_name, split="train")
        tasks = []

        for row in dataset:
            for example in row["examples"]:
                # Determine answer type
                answer_type = "mcq" if "target_scores" in example else "numeric"
                target = example.get("target_scores", example.get("target"))

                task = Task(
                    uuid=row["uuid"],
                    name=row["name"],
                    description=row["description"],
                    question=example["input"],
                    answer_type=answer_type,
                    target=target,
                    keywords=row["keywords"],
                )
                tasks.append(task)

        return tasks

    def get_tasks_by_type(self, answer_type: str) -> list[Task]:
        """Get tasks filtered by answer type."""
        return [task for task in self.tasks if task.answer_type == answer_type]
