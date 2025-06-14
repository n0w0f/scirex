import re
from dataclasses import dataclass
from typing import Any

from datasets import Dataset

from scirex.model import GeminiModel
from scirex.prompt import PromptTemplate
from scirex.task import Task


class LLMParser:
    """LLM-based answer parser."""

    def __init__(self, model: GeminiModel, prompt_template: PromptTemplate):
        self.model = model
        self.prompt_template = prompt_template

    def parse_mcq_answer(self, response: str) -> list[str]:
        """Parse MCQ answer using LLM."""
        parse_prompt = self.prompt_template.format_parse_prompt(response, "mcq")
        parsed = self.model.generate(parse_prompt).strip()

        if parsed == "UNCLEAR":
            return []

        # Extract letters
        letters = re.findall(r"[A-Z]", parsed.upper())
        return letters

    def parse_numeric_answer(self, response: str) -> float | None:
        """Parse numeric answer using LLM."""
        parse_prompt = self.prompt_template.format_parse_prompt(response, "numeric")
        parsed = self.model.generate(parse_prompt).strip()

        if parsed == "UNCLEAR":
            return None

        try:
            return float(parsed)
        except ValueError:
            return None


class Evaluator:
    """Evaluates model performance."""

    def evaluate_mcq(self, predicted: list[str], target_scores: dict[str, float]) -> dict[str, float]:
        """Evaluate MCQ prediction."""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        correct_answers = []

        # Find correct answers (score > 0)
        for i, (_option, score) in enumerate(target_scores.items()):
            if score > 0:
                correct_answers.append(letters[i])

        # Calculate metrics
        predicted_set = set(predicted)
        correct_set = set(correct_answers)

        # Accuracy: exact match
        accuracy = float(predicted_set == correct_set)

        # Precision: TP / (TP + FP)
        tp = len(predicted_set & correct_set)
        precision = tp / len(predicted_set) if predicted_set else 0.0

        # Recall: TP / (TP + FN)
        recall = tp / len(correct_set) if correct_set else 0.0

        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def evaluate_numeric(self, predicted: float | None, target: float, tolerance: float = 0.01) -> dict[str, float]:
        """Evaluate numeric prediction."""
        if predicted is None:
            return {"accuracy": 0.0, "mae": float("inf")}

        mae = abs(predicted - target)
        accuracy = float(mae <= tolerance * abs(target))

        return {"accuracy": accuracy, "mae": mae}


@dataclass
class BenchmarkResult:
    """Results from benchmark evaluation."""

    task: Task
    response: str
    parsed_answer: list[str] | float | None
    metrics: dict[str, float]
    success: bool


class Benchmark:
    """Main benchmark orchestrator."""

    def __init__(self, model: GeminiModel, prompt_template: PromptTemplate | None = None, tolerance: float = 0.01):
        self.model = model
        self.prompt_template = prompt_template or PromptTemplate()
        self.parser = LLMParser(model, self.prompt_template)
        self.evaluator = Evaluator()
        self.tolerance = tolerance

    def run_single_task(self, task: Task) -> BenchmarkResult:
        """Run benchmark on a single task."""
        try:
            # Generate prompt
            if task.answer_type == "mcq":
                prompt = self.prompt_template.format_mcq_prompt(task)
            else:
                prompt = self.prompt_template.format_numeric_prompt(task)

            # Get model response
            response = self.model.generate(prompt)

            # Parse answer
            if task.answer_type == "mcq":
                parsed_answer = self.parser.parse_mcq_answer(response)
                metrics = self.evaluator.evaluate_mcq(parsed_answer, task.target)
            else:
                parsed_answer = self.parser.parse_numeric_answer(response)
                metrics = self.evaluator.evaluate_numeric(parsed_answer, task.target, self.tolerance)

            success = True

        except Exception as e:
            print(f"Error processing task {task.uuid}: {e}")
            response = ""
            parsed_answer = None
            metrics = {"accuracy": 0.0}
            success = False

        return BenchmarkResult(
            task=task, response=response, parsed_answer=parsed_answer, metrics=metrics, success=success
        )

    def run_benchmark(self, dataset: Dataset, max_tasks: int | None = None) -> list[BenchmarkResult]:
        """Run benchmark on dataset."""
        tasks = dataset.tasks[:max_tasks] if max_tasks else dataset.tasks
        results = []

        for i, task in enumerate(tasks):
            print(f"Processing task {i + 1}/{len(tasks)}: {task.name}")
            result = self.run_single_task(task)
            results.append(result)

        return results

    def compute_summary_metrics(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Compute summary metrics across all results."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {"total_tasks": len(results), "successful_tasks": 0}

        # Separate by task type
        mcq_results = [r for r in successful_results if r.task.answer_type == "mcq"]
        numeric_results = [r for r in successful_results if r.task.answer_type == "numeric"]

        summary = {
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "success_rate": len(successful_results) / len(results),
        }

        # MCQ metrics
        if mcq_results:
            mcq_metrics = {}
            for metric in ["accuracy", "precision", "recall", "f1"]:
                values = [r.metrics[metric] for r in mcq_results]
                mcq_metrics[metric] = sum(values) / len(values)
            summary["mcq"] = mcq_metrics

        # Numeric metrics
        if numeric_results:
            numeric_metrics = {}
            for metric in ["accuracy", "mae"]:
                values = [r.metrics[metric] for r in numeric_results if r.metrics[metric] != float("inf")]
                if values:
                    numeric_metrics[metric] = sum(values) / len(values)
            summary["numeric"] = numeric_metrics

        return summary
