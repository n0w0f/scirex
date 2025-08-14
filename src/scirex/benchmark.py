import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
from loguru import logger

from scirex.model import GeminiModel
from scirex.prompt import PromptTemplate
from scirex.response import GeminiResponse, TokenUsage
from scirex.task import Task


class LLMParser:
    """Enhanced LLM-based answer parser with boxed format support."""

    def __init__(self, model: GeminiModel, prompt_template: PromptTemplate):
        self.model = model
        self.prompt_template = prompt_template

    def _extract_from_boxed(self, response: str) -> str | None:
        """
        Extract content from \\boxed{} format.

        Args:
            response: The response text to search

        Returns:
            Content inside the box if found, None otherwise
        """
        # Look for \boxed{content} pattern
        # Handle both \boxed{} and \\boxed{} formats
        patterns = [
            r"\\boxed\{([^}]*)\}",  # \boxed{content}
            r"\boxed\{([^}]*)\}",  # boxed{content} (missing backslash)
            r"\\boxed\{([^}]+)\}",  # Ensure non-empty content
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # Return the last match (most likely the final answer)
                return matches[-1].strip()

        return None

    def parse_mcq_answer(self, response: str) -> list[str]:
        """Parse MCQ answer, first trying boxed format, then LLM fallback."""
        # First try to extract from boxed format
        boxed_content = self._extract_from_boxed(response)

        if boxed_content:
            # Extract letters from boxed content
            letters = re.findall(r"[A-Z]", boxed_content.upper())
            if letters:
                return letters

        # Fallback to LLM parsing
        parse_prompt = self.prompt_template.format_parse_prompt(response, "mcq")
        parsed = self.model.generate(parse_prompt).strip()

        if parsed == "UNCLEAR":
            return []

        # Extract letters
        letters = re.findall(r"[A-Z]", parsed.upper())
        return letters

    def parse_numeric_answer(self, response: str) -> float | None:
        """Parse numeric answer, first trying boxed format, then LLM fallback."""
        # First try to extract from boxed format
        boxed_content = self._extract_from_boxed(response)

        if boxed_content:
            # Try to parse as number
            try:
                # Clean the content (remove common non-numeric chars)
                cleaned = re.sub(r"[^\d\.\-\+eE]", "", boxed_content)
                if cleaned:
                    return float(cleaned)
            except ValueError:
                pass

        # Fallback to LLM parsing
        parse_prompt = self.prompt_template.format_parse_prompt(response, "numeric")
        parsed = self.model.generate(parse_prompt).strip()

        if parsed == "UNCLEAR":
            return None

        try:
            return float(parsed)
        except ValueError:
            return None

    def parse_exact_match_answer(self, response: str) -> str | None:
        """Parse exact string match answer, first trying boxed format, then LLM fallback."""
        # First try to extract from boxed format
        boxed_content = self._extract_from_boxed(response)

        if boxed_content:
            return boxed_content.strip()

        # Fallback to LLM parsing
        parse_prompt = self.prompt_template.format_parse_prompt(response, "exact_str_match")
        parsed = self.model.generate(parse_prompt).strip()

        if parsed == "UNCLEAR":
            return None

        return parsed

    def get_boxed_extraction_stats(self, responses: list[str]) -> dict[str, Any]:
        """
        Get statistics on how often boxed format was found and used.
        Useful for monitoring prompt compliance.
        """
        total_responses = len(responses)
        boxed_found = 0

        for response in responses:
            if self._extract_from_boxed(response):
                boxed_found += 1

        return {
            "total_responses": total_responses,
            "boxed_found": boxed_found,
            "boxed_rate": boxed_found / total_responses if total_responses > 0 else 0.0,
            "fallback_rate": (total_responses - boxed_found) / total_responses if total_responses > 0 else 0.0,
        }

    def debug_parsing(self, response: str, answer_type: str) -> dict[str, Any]:
        """
        Debug parsing process - shows both boxed extraction and LLM parsing results.
        Useful for troubleshooting parsing issues.
        """
        boxed_content = self._extract_from_boxed(response)

        # Get LLM parsing result
        parse_prompt = self.prompt_template.format_parse_prompt(response, answer_type)
        llm_parsed = self.model.generate(parse_prompt).strip()

        debug_info = {
            "original_response": response,
            "answer_type": answer_type,
            "boxed_found": boxed_content is not None,
            "boxed_content": boxed_content,
            "llm_parsed": llm_parsed,
            "used_method": "boxed" if boxed_content else "llm",
        }

        # Get final parsed result
        if answer_type == "mcq":
            final_result = self.parse_mcq_answer(response)
        elif answer_type == "numeric":
            final_result = self.parse_numeric_answer(response)
        elif answer_type == "exact_str_match":
            final_result = self.parse_exact_match_answer(response)
        else:
            final_result = None

        debug_info["final_result"] = final_result

        return debug_info


class Evaluator:
    """Enhanced evaluator with exact string match support."""

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

    def evaluate_exact_match(self, predicted: str | None, target: str) -> dict[str, float]:
        """Evaluate exact string match prediction."""
        if predicted is None:
            return {"accuracy": 0.0, "exact_match": 0.0}

        # Clean both strings for comparison
        predicted_clean = predicted.strip()
        target_clean = target.strip()

        # Exact match
        exact_match = float(predicted_clean == target_clean)

        # Case-insensitive match
        case_insensitive_match = float(predicted_clean.lower() == target_clean.lower())

        # Contains match (predicted contains target or vice versa)
        contains_match = float(
            target_clean.lower() in predicted_clean.lower() or predicted_clean.lower() in target_clean.lower()
        )

        return {
            "accuracy": exact_match,  # Use exact match as primary accuracy metric
            "exact_match": exact_match,
            "case_insensitive_match": case_insensitive_match,
            "contains_match": contains_match,
        }


@dataclass
class BenchmarkResult:
    """Enhanced results from benchmark evaluation."""

    task: Task
    response: str  # Keep for backward compatibility
    parsed_answer: list[str] | float | str | None  # Updated to include string answers
    metrics: dict[str, float]
    success: bool

    # Rich response data
    full_response: GeminiResponse | None = None

    # Enhanced fields
    is_multimodal: bool = False
    content_type: str = "text"  # "text", "multimodal", "text_fallback", etc.
    answer_type: str = "unknown"  # "mcq", "numeric", "exact_str_match"

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
    """Enhanced benchmark orchestrator with exact string match support."""

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
            logger.info(f"Multimodal support: {'✓' if self.multimodal_supported else '✗'}")

    def run_single_task(self, task: Task) -> BenchmarkResult:
        """Run benchmark on a single task with enhanced answer type support."""
        try:
            # Check if task is multimodal but model doesn't support it
            if task.is_multimodal and not self.multimodal_supported:
                logger.info(
                    f"Warning: Task {task.uuid} is multimodal but model doesn't support it. Using text-only version."
                )
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

            # Parse answer based on task type
            if task.answer_type == "mcq":
                parsed_answer = self.parser.parse_mcq_answer(response)
                metrics = self.evaluator.evaluate_mcq(parsed_answer, task.target)
            elif task.answer_type == "numeric":
                parsed_answer = self.parser.parse_numeric_answer(response)
                metrics = self.evaluator.evaluate_numeric(parsed_answer, task.target, self.tolerance)
            elif task.answer_type == "exact_str_match":
                parsed_answer = self.parser.parse_exact_match_answer(response)
                metrics = self.evaluator.evaluate_exact_match(parsed_answer, task.target)
            else:
                logger.warning(f"Unknown answer type '{task.answer_type}' for task {task.uuid}")
                parsed_answer = None
                metrics = {"accuracy": 0.0}

            success = True

        except Exception as e:
            logger.error(f"Error processing task {task.uuid}: {e}")
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
            answer_type=task.answer_type,
        )

    def _format_multimodal_prompt(self, task: Task) -> list[Any]:
        """Format prompt for multimodal task."""
        if task.answer_type == "mcq":
            return self.prompt_template.format_mcq_prompt(task)
        elif task.answer_type == "numeric":
            return self.prompt_template.format_numeric_prompt(task)
        elif task.answer_type == "exact_str_match":
            return self.prompt_template.format_exact_match_prompt(task)
        else:
            # Fallback to text prompt
            return self.prompt_template.format_exact_match_prompt(task)

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
        elif task.answer_type == "numeric":
            return self.prompt_template.format_numeric_prompt(temp_task)
        elif task.answer_type == "exact_str_match":
            return self.prompt_template.format_exact_match_prompt(temp_task)
        else:
            # Fallback
            return self.prompt_template.format_exact_match_prompt(temp_task)

    def run_benchmark(self, dataset: Dataset, max_tasks: int | None = None) -> list[BenchmarkResult]:
        """Run benchmark on dataset with enhanced answer type support."""
        tasks = dataset.tasks[:max_tasks] if max_tasks else dataset.tasks
        results = []

        # Enhanced dataset statistics
        total_tasks = len(tasks)
        multimodal_tasks = len([t for t in tasks if t.is_multimodal])
        text_tasks = total_tasks - multimodal_tasks

        # Count by answer type
        mcq_tasks = len([t for t in tasks if t.answer_type == "mcq"])
        numeric_tasks = len([t for t in tasks if t.answer_type == "numeric"])
        exact_match_tasks = len([t for t in tasks if t.answer_type == "exact_str_match"])

        logger.info(f"Dataset statistics:")
        logger.info(f"  Total tasks: {total_tasks}")
        logger.info(f"  Text-only tasks: {text_tasks}")
        logger.info(f"  Multimodal tasks: {multimodal_tasks}")
        logger.info(f"  MCQ tasks: {mcq_tasks}")
        logger.info(f"  Numeric tasks: {numeric_tasks}")
        logger.info(f"  Exact match tasks: {exact_match_tasks}")

        for i, task in enumerate(tasks):
            task_type = "multimodal" if task.is_multimodal else "text"
            logger.info(f"Processing task {i + 1}/{len(tasks)} ({task_type}, {task.answer_type}): {task.name}")
            result = self.run_single_task(task)
            results.append(result)

        return results

    def compute_summary_metrics(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Compute enhanced summary metrics across all results."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {"total_tasks": len(results), "successful_tasks": 0}

        # Separate by task type
        mcq_results = [r for r in successful_results if r.answer_type == "mcq"]
        numeric_results = [r for r in successful_results if r.answer_type == "numeric"]
        exact_match_results = [r for r in successful_results if r.answer_type == "exact_str_match"]

        # Separate by modality
        text_results = [r for r in successful_results if not r.is_multimodal]
        multimodal_results = [r for r in successful_results if r.is_multimodal]

        summary = {
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "text_tasks": len(text_results),
            "multimodal_tasks": len(multimodal_results),
            "mcq_tasks": len(mcq_results),
            "numeric_tasks": len(numeric_results),
            "exact_match_tasks": len(exact_match_results),
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

        # Exact match metrics
        if exact_match_results:
            exact_match_metrics = {}
            for metric in ["accuracy", "exact_match", "case_insensitive_match", "contains_match"]:
                values = [r.metrics.get(metric, 0.0) for r in exact_match_results]
                exact_match_metrics[metric] = sum(values) / len(values)
            summary["exact_str_match"] = exact_match_metrics

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
        """Save enhanced benchmark results to JSON file."""
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
                    "preferred_score": result.task.preferred_score,
                    "question_template": result.task.question_template,
                },
                "response": result.response,
                "parsed_answer": result.parsed_answer,
                "metrics": result.metrics,
                "success": result.success,
                "is_multimodal": result.is_multimodal,
                "content_type": result.content_type,
                "answer_type": result.answer_type,
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

        logger.info(f"Results saved to {output_path}")
        logger.info(f"Total tokens used: {total_tokens:,}")
        logger.info(f"Estimated cost: ${total_cost_estimate:.4f}")

    def _compute_token_stats(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Compute enhanced token usage statistics."""
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
            "mcq_task_tokens": 0,
            "numeric_task_tokens": 0,
            "exact_match_task_tokens": 0,
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

            # Track by answer type
            if result.answer_type == "mcq":
                token_stats["mcq_task_tokens"] += tokens
            elif result.answer_type == "numeric":
                token_stats["numeric_task_tokens"] += tokens
            elif result.answer_type == "exact_str_match":
                token_stats["exact_match_task_tokens"] += tokens

            # Track min/max
            token_stats["max_tokens_single_task"] = max(token_stats["max_tokens_single_task"], tokens)
            if token_stats["min_tokens_single_task"] == 0:
                token_stats["min_tokens_single_task"] = tokens
            else:
                token_stats["min_tokens_single_task"] = min(token_stats["min_tokens_single_task"], tokens)

        if valid_results:
            token_stats["avg_tokens_per_task"] = token_stats["total_tokens"] / len(valid_results)

        return token_stats
