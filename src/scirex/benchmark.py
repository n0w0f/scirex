import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset

from scirex.model import GeminiModel
from scirex.prompt import PromptTemplate
from scirex.response import GeminiResponse, TokenUsage  # Add these imports
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
    response: str  # Keep for backward compatibility
    parsed_answer: list[str] | float | None
    metrics: dict[str, float]
    success: bool

    # Rich response data
    full_response: Optional[GeminiResponse] = None

    @property
    def thought_summary(self) -> str | None:
        """Backward compatibility property."""
        return self.full_response.thought_summary if self.full_response else None

    @property
    def token_usage(self) -> TokenUsage | None:
        """Easy access to token usage."""
        return self.full_response.token_usage if self.full_response else None

    @property
    def total_tokens(self) -> int:
        """Easy access to total token count."""
        if self.full_response and self.full_response.token_usage:
            return self.full_response.token_usage.total_token_count
        return 0


class Benchmark:
    """Main benchmark orchestrator."""

    def __init__(
        self,
        model: GeminiModel,
        prompt_template: PromptTemplate | None = None,
        tolerance: float = 0.01,
        include_thoughts: bool = False,
    ):
        self.model = model
        self.prompt_template = prompt_template or PromptTemplate()
        self.parser = LLMParser(model, self.prompt_template)
        self.evaluator = Evaluator()
        self.tolerance = tolerance
        self.include_thoughts = include_thoughts

    def run_single_task(self, task: Task) -> BenchmarkResult:
        """Run benchmark on a single task."""
        try:
            # Generate prompt
            if task.answer_type == "mcq":
                prompt = self.prompt_template.format_mcq_prompt(task)
            else:
                prompt = self.prompt_template.format_numeric_prompt(task)

            # Get full model response with all metadata
            full_response = self.model.generate(prompt, return_full_response=True)
            response = full_response.text

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
            full_response = None

        return BenchmarkResult(
            task=task,
            response=response,
            parsed_answer=parsed_answer,
            metrics=metrics,
            success=success,
            full_response=full_response if success else None,
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

    def save_results_to_json(
        self, results: list[BenchmarkResult], output_path: str | Path, include_summary: bool = True
    ) -> None:
        """Save benchmark results to JSON file."""
        output_path = Path(output_path)

        # Convert results to serializable format
        serializable_results = []
        total_tokens = 0
        total_cost_estimate = 0.0

        for result in results:
            result_dict = {
                "task": {
                    "uuid": result.task.uuid,
                    "name": result.task.name,
                    "answer_type": result.task.answer_type,
                    "target": result.task.target,
                    # Add other task fields as needed
                },
                "response": result.response,
                "parsed_answer": result.parsed_answer,
                "metrics": result.metrics,
                "success": result.success,
                # Rich response data
                "full_response": result.full_response.to_dict() if result.full_response else None,
                # Convenience fields (for easy access without drilling into full_response)
                "thought_summary": result.thought_summary,
                "total_tokens": result.total_tokens,
                "model_version": result.full_response.model_version if result.full_response else None,
                "finish_reason": result.full_response.finish_reason if result.full_response else None,
            }

            # Track totals for summary
            if result.full_response:
                total_tokens += result.total_tokens
                total_cost_estimate += result.full_response.get_cost_estimate()

            serializable_results.append(result_dict)

        # Prepare output data
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_results": len(results),
                "include_thoughts": self.include_thoughts,
                "tolerance": self.tolerance,
                "total_tokens_used": total_tokens,
                "estimated_total_cost": round(total_cost_estimate, 4),
            },
            "results": serializable_results,
        }

        # Add summary metrics if requested
        if include_summary:
            summary_metrics = self.compute_summary_metrics(results)
            summary_metrics["token_stats"] = self._compute_token_stats(results)
            output_data["summary_metrics"] = summary_metrics

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")
        print(f"Total tokens used: {total_tokens:,}")
        print(f"Estimated cost: ${total_cost_estimate:.4f}")

    def _compute_token_stats(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Compute token usage statistics (basic version for compatibility)."""
        token_stats = {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "total_thought_tokens": 0,
            "avg_tokens_per_task": 0,
            "max_tokens_single_task": 0,
            "min_tokens_single_task": 0,
        }

        # If your BenchmarkResult doesn't have token tracking, return empty stats
        if not hasattr(results[0], "total_tokens") if results else True:
            return token_stats

        valid_results = [r for r in results if hasattr(r, "total_tokens") and r.total_tokens > 0]

        if not valid_results:
            return token_stats

        for result in valid_results:
            token_stats["total_tokens"] += getattr(result, "total_tokens", 0)
            # Add other token types if available
            token_stats["max_tokens_single_task"] = max(
                token_stats["max_tokens_single_task"], getattr(result, "total_tokens", 0)
            )
            if token_stats["min_tokens_single_task"] == 0:
                token_stats["min_tokens_single_task"] = getattr(result, "total_tokens", 0)
            else:
                token_stats["min_tokens_single_task"] = min(
                    token_stats["min_tokens_single_task"], getattr(result, "total_tokens", 0)
                )

        if valid_results:
            token_stats["avg_tokens_per_task"] = token_stats["total_tokens"] / len(valid_results)

        return token_stats

    def load_results_from_json(self, input_path: str | Path) -> tuple[list[BenchmarkResult], dict[str, Any]]:
        """Load benchmark results from JSON file."""
        pass
