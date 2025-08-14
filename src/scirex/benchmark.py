import re
from dataclasses import dataclass
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
        """Extract content from \\boxed{} format."""
        patterns = [
            r"\\boxed\{([^}]*)\}",  # \boxed{content}
            r"\boxed\{([^}]*)\}",  # boxed{content} (missing backslash)
            r"\\boxed\{([^}]+)\}",  # Ensure non-empty content
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[-1].strip()

        return None

    def parse_mcq_answer(self, response: str) -> list[str]:
        """Parse MCQ answer, first trying boxed format, then LLM fallback."""
        boxed_content = self._extract_from_boxed(response)

        if boxed_content:
            letters = re.findall(r"[A-Z]", boxed_content.upper())
            if letters:
                return letters

        try:
            parse_prompt = self.prompt_template.format_parse_prompt(response, "mcq")
            parsed = self.model.generate(parse_prompt).strip()
            if parsed == "UNCLEAR":
                return []
            letters = re.findall(r"[A-Z]", parsed.upper())
            return letters
        except Exception as e:
            logger.error(f"Error in MCQ parsing: {e}")
            return []

    def parse_numeric_answer(self, response: str) -> float | None:
        """Parse numeric answer, first trying boxed format, then LLM fallback."""
        boxed_content = self._extract_from_boxed(response)

        if boxed_content:
            try:
                cleaned = re.sub(r"[^\d\.\-\+eE]", "", boxed_content)
                if cleaned:
                    return float(cleaned)
            except ValueError:
                pass

        try:
            parse_prompt = self.prompt_template.format_parse_prompt(response, "numeric")
            parsed = self.model.generate(parse_prompt).strip()
            if parsed == "UNCLEAR":
                return None
            return float(parsed)
        except Exception as e:
            logger.error(f"Error in numeric parsing: {e}")
            return None

    def parse_exact_match_answer(self, response: str) -> str | None:
        """Parse exact string match answer, first trying boxed format, then LLM fallback."""
        boxed_content = self._extract_from_boxed(response)

        if boxed_content:
            return boxed_content.strip()

        try:
            parse_prompt = self.prompt_template.format_parse_prompt(response, "exact_str_match")
            parsed = self.model.generate(parse_prompt).strip()
            if parsed == "UNCLEAR":
                return None
            return parsed
        except Exception as e:
            logger.error(f"Error in exact match parsing: {e}")
            return None


class Evaluator:
    """Enhanced evaluator with exact string match support."""

    def evaluate_mcq(self, predicted: list[str], target_scores: dict[str, float]) -> dict[str, float]:
        """Evaluate MCQ prediction."""
        try:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            correct_answers = []

            for i, (_option, score) in enumerate(target_scores.items()):
                if score > 0:
                    correct_answers.append(letters[i])

            predicted_set = set(predicted)
            correct_set = set(correct_answers)

            accuracy = float(predicted_set == correct_set)
            tp = len(predicted_set & correct_set)
            precision = tp / len(predicted_set) if predicted_set else 0.0
            recall = tp / len(correct_set) if correct_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        except Exception as e:
            logger.error(f"Error in MCQ evaluation: {e}")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    def evaluate_numeric(self, predicted: float | None, target: float, tolerance: float = 0.01) -> dict[str, float]:
        """Evaluate numeric prediction."""
        try:
            if predicted is None:
                return {"accuracy": 0.0, "mae": float("inf")}

            mae = abs(predicted - target)
            accuracy = float(mae <= tolerance * abs(target))

            return {"accuracy": accuracy, "mae": mae}
        except Exception as e:
            logger.error(f"Error in numeric evaluation: {e}")
            return {"accuracy": 0.0, "mae": float("inf")}

    def evaluate_exact_match(self, predicted: str | None, target: str) -> dict[str, float]:
        """Evaluate exact string match prediction."""
        try:
            if predicted is None:
                return {"accuracy": 0.0, "exact_match": 0.0}

            predicted_clean = predicted.strip()
            target_clean = target.strip()

            exact_match = float(predicted_clean == target_clean)
            case_insensitive_match = float(predicted_clean.lower() == target_clean.lower())
            contains_match = float(
                target_clean.lower() in predicted_clean.lower() or predicted_clean.lower() in target_clean.lower()
            )

            return {
                "accuracy": exact_match,
                "exact_match": exact_match,
                "case_insensitive_match": case_insensitive_match,
                "contains_match": contains_match,
            }
        except Exception as e:
            logger.error(f"Error in exact match evaluation: {e}")
            return {"accuracy": 0.0, "exact_match": 0.0}


@dataclass
class BenchmarkResult:
    """Enhanced results from benchmark evaluation."""

    task: Task
    response: str
    parsed_answer: list[str] | float | str | None
    metrics: dict[str, float]
    success: bool
    full_response: GeminiResponse | None = None
    is_multimodal: bool = False
    content_type: str = "text"
    answer_type: str = "unknown"

    @property
    def thought_summary(self) -> str | None:
        return self.full_response.thought_summary if self.full_response else None

    @property
    def token_usage(self) -> TokenUsage | None:
        return self.full_response.token_usage if self.full_response else None

    @property
    def total_tokens(self) -> int:
        if self.full_response and self.full_response.token_usage:
            return self.full_response.token_usage.total_token_count
        return 0


class Benchmark:
    """Enhanced benchmark orchestrator with better error handling."""

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
            try:
                self.multimodal_supported = self.model.test_multimodal_capability()
                logger.info(f"Multimodal support: {'✓' if self.multimodal_supported else '✗'}")
            except Exception as e:
                logger.warning(f"Multimodal test failed: {e}")
                self.multimodal_supported = False

    def run_single_task(self, task: Task) -> BenchmarkResult:
        """Run benchmark on a single task with enhanced error handling."""
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
                prompt = self._format_multimodal_prompt(task)
                content_type = "multimodal"
            else:
                prompt = self._format_text_prompt(task, task.question)
                content_type = "text"

            # Get full model response with all metadata
            full_response = self.model.generate(prompt, return_full_response=True)
            response = full_response.text

            # Parse answer based on task type
            parsed_answer = None
            metrics = {"accuracy": 0.0}

            if task.answer_type == "mcq":
                parsed_answer = self.parser.parse_mcq_answer(response)
                if not isinstance(task.target, dict):
                    logger.error(f"MCQ task target should be dict, got {type(task.target)}: {task.target}")
                    metrics = {"accuracy": 0.0}
                else:
                    metrics = self.evaluator.evaluate_mcq(parsed_answer, task.target)

            elif task.answer_type == "numeric":
                parsed_answer = self.parser.parse_numeric_answer(response)
                if not isinstance(task.target, (int, float)):
                    logger.error(f"Numeric task target should be number, got {type(task.target)}: {task.target}")
                    metrics = {"accuracy": 0.0}
                else:
                    metrics = self.evaluator.evaluate_numeric(parsed_answer, float(task.target), self.tolerance)

            elif task.answer_type == "exact_str_match":
                parsed_answer = self.parser.parse_exact_match_answer(response)
                target_str = str(task.target)
                metrics = self.evaluator.evaluate_exact_match(parsed_answer, target_str)

            else:
                logger.warning(f"Unknown answer type '{task.answer_type}' for task {task.uuid}")
                parsed_answer = None
                metrics = {"accuracy": 0.0}

            logger.debug(f"Parsed answer: {parsed_answer}")
            logger.debug(f"Metrics: {metrics}")

            success = True

        except KeyError as e:
            logger.error(f"KeyError processing task {task.uuid}: missing key '{e}'")
            logger.error(f"Task attributes: {[attr for attr in dir(task) if not attr.startswith('_')]}")
            logger.error(f"Task target: {getattr(task, 'target', 'NOT_FOUND')}")
            logger.error(f"Task answer_type: {getattr(task, 'answer_type', 'NOT_FOUND')}")
            response = ""
            parsed_answer = None
            metrics = {"accuracy": 0.0}
            success = False
            full_response = None
            content_type = "error"

        except Exception as e:
            logger.error(f"Error processing task {task.uuid}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
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
        try:
            if task.answer_type == "mcq":
                return self.prompt_template.format_mcq_prompt(task)
            elif task.answer_type == "numeric":
                return self.prompt_template.format_numeric_prompt(task)
            elif task.answer_type == "exact_str_match":
                return self.prompt_template.format_exact_match_prompt(task)
            else:
                return self.prompt_template.format_exact_match_prompt(task)
        except Exception as e:
            logger.error(f"Error formatting multimodal prompt: {e}")
            return [str(task.question)]

    def _format_text_prompt(self, task: Task, text_content: str) -> str:
        """Format prompt for text-only task."""
        try:
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
                return self.prompt_template.format_exact_match_prompt(temp_task)
        except Exception as e:
            logger.error(f"Error formatting text prompt: {e}")
            return str(task.question)

    def run_benchmark(self, dataset: Dataset, max_tasks: int | None = None) -> list[BenchmarkResult]:
        """Run benchmark on dataset with enhanced error handling."""
        tasks = dataset.tasks[:max_tasks] if max_tasks else dataset.tasks
        results = []

        logger.info(f"Running benchmark on {len(tasks)} tasks")

        for i, task in enumerate(tasks):
            task_type = "multimodal" if task.is_multimodal else "text"
            logger.info(f"Processing task {i + 1}/{len(tasks)} ({task_type}, {task.answer_type}): {task.name}")
            result = self.run_single_task(task)
            results.append(result)

            if result.success:
                logger.info(f"✅ Task completed - Accuracy: {result.metrics.get('accuracy', 0):.3f}")
            else:
                logger.warning(f"❌ Task failed")

        return results

    def compute_summary_metrics(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Compute summary metrics across all results."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {"total_tasks": len(results), "successful_tasks": 0}

        summary = {
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "success_rate": len(successful_results) / len(results),
        }

        # Overall accuracy
        overall_accuracy = sum(r.metrics.get("accuracy", 0) for r in successful_results) / len(successful_results)
        summary["overall_accuracy"] = overall_accuracy

        return summary
