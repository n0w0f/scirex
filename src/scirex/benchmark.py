import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset

from scirex.model import GeminiModel
from scirex.prompt import PromptTemplate
from scirex.response import GeminiResponse, TokenUsage
from scirex.task import Task


class LLMParser:
    """LLM-based answer parser with multimodal support."""

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
    """Results from benchmark evaluation with multimodal support."""

    task: Task
    response: str  # Keep for backward compatibility
    parsed_answer: list[str] | float | None
    metrics: dict[str, float]
    success: bool

    # Rich response data
    full_response: GeminiResponse | None = None

    # Multimodal specific fields
    is_multimodal: bool = False
    content_type: str = "text"  # "text", "multimodal", etc.

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
    """Main benchmark orchestrator with multimodal support."""

    def __init__(
        self,
        model: GeminiModel,
        prompt_template: PromptTemplate | None = None,
        tolerance: float = 0.01,
        include_thoughts: bool = False,
        test_multimodal: bool = True,
    ):
        self.model = model
        self.prompt_template = prompt_template or PromptTemplate()
        self.parser = LLMParser(model, self.prompt_template)
        self.evaluator = Evaluator()
        self.tolerance = tolerance
        self.include_thoughts = include_thoughts

        # Test multimodal capability if requested
        self.multimodal_supported = False
        if test_multimodal:
            self.multimodal_supported = self.model.test_multimodal_capability()
            print(f"Multimodal support: {'✓' if self.multimodal_supported else '✗'}")

    def run_single_task(self, task: Task) -> BenchmarkResult:
        """Run benchmark on a single task with multimodal support."""
        try:
            # Check if task is multimodal but model doesn't support it
            if task.is_multimodal and not self.multimodal_supported:
                print(f"Warning: Task {task.uuid} is multimodal but model doesn't support it. Using text-only version.")
                task_content = task.get_text_content()
                prompt = self._format_text_prompt(task, task_content)
                content_type = "text_fallback"
            elif task.is_multimodal:
                # Use multimodal prompt
                prompt = self._format_multimodal_prompt(task)
                content_type = "multimodal"
            else:
                # Use text-only prompt
                prompt = self._format_text_prompt(task, task.question)
                content_type = "text"

            # Get full model response with all metadata
            full_response = self.model.generate(prompt, return_full_response=True)
            response = full_response.text

            # Parse answer using text representation for parsing
            text_for_parsing = task.get_text_content() if task.is_multimodal else task.question

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
            content_type = "error"

        return BenchmarkResult(
            task=task,
            response=response,
            parsed_answer=parsed_answer,
            metrics=metrics,
            success=success,
            full_response=full_response if success else None,
            is_multimodal=task.is_multimodal,
            content_type=content_type,
        )

    def _format_multimodal_prompt(self, task: Task) -> list[Any]:
        """Format prompt for multimodal task."""
        if task.answer_type == "mcq":
            return self.prompt_template.format_mcq_prompt(task)
        else:
            return self.prompt_template.format_numeric_prompt(task)

    def _format_text_prompt(self, task: Task, text_content: str) -> str:
        """Format prompt for text-only task."""
        # Create a temporary task with the text content for formatting
        temp_task = Task(
            uuid=task.uuid,
            name=task.name,
            description=task.description,
            question=text_content,
            answer_type=task.answer_type,
            target=task.target,
            keywords=task.keywords,
            is_multimodal=False,
        )

        if task.answer_type == "mcq":
            return self.prompt_template.format_mcq_prompt(temp_task)
        else:
            return self.prompt_template.format_numeric_prompt(temp_task)

    def run_benchmark(self, dataset: Dataset, max_tasks: int | None = None) -> list[BenchmarkResult]:
        """Run benchmark on dataset with multimodal support."""
        tasks = dataset.tasks[:max_tasks] if max_tasks else dataset.tasks
        results = []

        # Print dataset statistics
        total_tasks = len(tasks)
        multimodal_tasks = len([t for t in tasks if t.is_multimodal])
        text_tasks = total_tasks - multimodal_tasks

        print(f"Dataset statistics:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Text-only tasks: {text_tasks}")
        print(f"  Multimodal tasks: {multimodal_tasks}")

        for i, task in enumerate(tasks):
            task_type = "multimodal" if task.is_multimodal else "text"
            print(f"Processing task {i + 1}/{len(tasks)} ({task_type}): {task.name}")
            result = self.run_single_task(task)
            results.append(result)

        return results

    def compute_summary_metrics(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Compute summary metrics across all results with multimodal breakdown."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {"total_tasks": len(results), "successful_tasks": 0}

        # Separate by task type
        mcq_results = [r for r in successful_results if r.task.answer_type == "mcq"]
        numeric_results = [r for r in successful_results if r.task.answer_type == "numeric"]

        # Separate by modality
        text_results = [r for r in successful_results if not r.is_multimodal]
        multimodal_results = [r for r in successful_results if r.is_multimodal]

        summary = {
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "text_tasks": len(text_results),
            "multimodal_tasks": len(multimodal_results),
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

        # Modality-specific metrics
        if text_results:
            text_accuracy = sum(r.metrics["accuracy"] for r in text_results) / len(text_results)
            summary["text_accuracy"] = text_accuracy

        if multimodal_results:
            multimodal_accuracy = sum(r.metrics["accuracy"] for r in multimodal_results) / len(multimodal_results)
            summary["multimodal_accuracy"] = multimodal_accuracy

        return summary

    def save_results_to_json(
        self, results: list[BenchmarkResult], output_path: str | Path, include_summary: bool = True
    ) -> None:
        """Save benchmark results to JSON file with multimodal metadata."""
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
                    "is_multimodal": result.task.is_multimodal,
                    "input_template": result.task.input_template if result.task.is_multimodal else None,
                },
                "response": result.response,
                "parsed_answer": result.parsed_answer,
                "metrics": result.metrics,
                "success": result.success,
                "is_multimodal": result.is_multimodal,
                "content_type": result.content_type,
                # Rich response data
                "full_response": result.full_response.to_dict() if result.full_response else None,
                # Convenience fields
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
                "multimodal_supported": self.multimodal_supported,
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
        """Compute token usage statistics with multimodal breakdown."""
        token_stats = {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "total_thought_tokens": 0,
            "avg_tokens_per_task": 0,
            "max_tokens_single_task": 0,
            "min_tokens_single_task": 0,
            "text_task_tokens": 0,
            "multimodal_task_tokens": 0,
        }

        valid_results = [r for r in results if hasattr(r, "total_tokens") and r.total_tokens > 0]

        if not valid_results:
            return token_stats

        for result in valid_results:
            tokens = getattr(result, "total_tokens", 0)
            token_stats["total_tokens"] += tokens

            # Track by modality
            if result.is_multimodal:
                token_stats["multimodal_task_tokens"] += tokens
            else:
                token_stats["text_task_tokens"] += tokens

            # Track min/max
            token_stats["max_tokens_single_task"] = max(token_stats["max_tokens_single_task"], tokens)
            if token_stats["min_tokens_single_task"] == 0:
                token_stats["min_tokens_single_task"] = tokens
            else:
                token_stats["min_tokens_single_task"] = min(token_stats["min_tokens_single_task"], tokens)

        if valid_results:
            token_stats["avg_tokens_per_task"] = token_stats["total_tokens"] / len(valid_results)

        return token_stats

    def load_results_from_json(self, input_path: str | Path) -> tuple[list[BenchmarkResult], dict[str, Any]]:
        """Load benchmark results from JSON file."""
        pass
