"""
Complete benchmark runner script for all dataset subsets.
Runs benchmarks and saves results as both pickle and JSON files.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from google.genai import types
from loguru import logger

from scirex.benchmark import Benchmark
from scirex.model import GeminiModel
from scirex.prompt import PromptTemplate
from scirex.task import Dataset


def setup_directories(base_output_dir: str = "benchmark_results") -> Path:
    """Create directory structure for results."""
    base_path = Path(base_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped run directory
    run_dir = base_path / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_dir / "pickle_results").mkdir(exist_ok=True)
    (run_dir / "json_results").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    return run_dir


def create_model(
    model_name: str = "gemini", temperature: float = 0.0, thinking_budget: int = 0, delay: int = 2
) -> tuple[GeminiModel, bool]:
    """Create GeminiModel with thinking configuration."""

    if thinking_budget == 0:
        # No thinking
        model = GeminiModel(
            model_name=model_name,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=0),
            delay=delay,
        )
        include_thoughts = False
    else:
        # With thinking
        model = GeminiModel(
            model_name=model_name,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=thinking_budget),
            delay=delay,
        )
        include_thoughts = True

    return model, include_thoughts


def benchmark_single_subset(
    dataset_name: str,
    subset: str,
    model: GeminiModel,
    include_thoughts: bool,
    output_dir: Path,
    max_tasks: int | None = None,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Benchmark a single dataset subset and save results."""

    logger.info(f"Starting benchmark for {dataset_name}/{subset}")

    try:
        # Load dataset
        logger.info(f"Loading dataset: {dataset_name}/{subset}")
        dataset = Dataset(dataset_name, subset)

        if not dataset.tasks:
            logger.warning(f"No tasks found in {dataset_name}/{subset}")
            return {"error": "No tasks found", "subset": subset}

        logger.info(f"Loaded {len(dataset.tasks)} tasks")

        # Initialize benchmark
        prompt_template = PromptTemplate()
        benchmark = Benchmark(
            model=model,
            prompt_template=prompt_template,
            tolerance=tolerance,
            include_thoughts=include_thoughts,
            test_multimodal=False,
        )

        # Run benchmark
        logger.info(f"Running benchmark on {max_tasks or 'all'} tasks...")
        results = benchmark.run_benchmark(dataset, max_tasks=max_tasks)

        # Compute summary metrics
        summary_metrics = benchmark.compute_summary_metrics(results)

        # Save pickle file (complete results object)
        pickle_path = output_dir / "pickle_results" / f"{subset}_results.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved pickle results to {pickle_path}")

        # Save JSON file (metrics and metadata)
        json_path = output_dir / "json_results" / f"{subset}_metrics.json"
        benchmark.save_results_to_json(results, json_path, include_summary=True)
        logger.info(f"Saved JSON metrics to {json_path}")

        # Return summary for main log
        return {
            "subset": subset,
            "status": "success",
            "total_tasks": len(results),
            "successful_tasks": len([r for r in results if r.success]),
            "accuracy": summary_metrics.get("text_accuracy") or summary_metrics.get("multimodal_accuracy", 0),
            "total_tokens": sum(r.total_tokens for r in results if r.total_tokens > 0),
            "estimated_cost": sum(r.full_response.get_cost_estimate() for r in results if r.full_response),
            "files": {"pickle": str(pickle_path), "json": str(json_path)},
        }

    except Exception as e:
        logger.error(f"Error benchmarking {subset}: {e}")
        return {"subset": subset, "status": "error", "error": str(e)}


def run_all_benchmarks(
    configs: list[str],
    dataset_name: str = "n0w0f/scirex-text",
    model_name: str = "gemini",
    temperature: float = 0.0,
    thinking_budget: int = 0,
    delay: int = 2,
    max_tasks: int | None = None,
    tolerance: float = 0.01,
    output_dir: str = "benchmark_results",
) -> None:
    """Run benchmarks on all dataset subsets."""

    # Setup directories
    run_dir = setup_directories(output_dir)

    # Setup logging
    log_file = run_dir / "logs" / "benchmark_run.log"
    logger.add(log_file, rotation="10 MB")

    logger.info(f"Starting benchmark run with {len(configs)} subsets")
    logger.info(f"Model: {model_name}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Thinking budget: {thinking_budget}")
    logger.info(f"Max tasks per subset: {max_tasks or 'All'}")
    logger.info(f"Output directory: {run_dir}")

    # Create model
    model, include_thoughts = create_model(model_name, temperature, thinking_budget, delay)

    # Track overall progress
    all_results = []
    total_cost = 0.0
    total_tokens = 0
    successful_subsets = 0
    failed_subsets = 0

    for i, subset in enumerate(configs, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing subset {i}/{len(configs)}: {subset}")
        logger.info(f"{'=' * 60}")

        # Benchmark this subset
        result = benchmark_single_subset(
            dataset_name=dataset_name,
            subset=subset,
            model=model,
            include_thoughts=include_thoughts,
            output_dir=run_dir,
            max_tasks=max_tasks,
            tolerance=tolerance,
        )

        all_results.append(result)

        # Track totals
        if result["status"] == "success":
            successful_subsets += 1
            total_cost += result.get("estimated_cost", 0)
            total_tokens += result.get("total_tokens", 0)

            logger.info(f"✅ {subset} completed successfully")
            logger.info(f"   Tasks: {result['successful_tasks']}/{result['total_tasks']}")
            logger.info(f"   Accuracy: {result['accuracy']:.3f}")
            logger.info(f"   Tokens: {result['total_tokens']:,}")
            logger.info(f"   Cost: ${result['estimated_cost']:.4f}")
        else:
            failed_subsets += 1
            logger.error(f"❌ {subset} failed: {result.get('error', 'Unknown error')}")

    # Save overall summary
    summary = {
        "run_metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "temperature": temperature,
            "thinking_budget": thinking_budget,
            "max_tasks_per_subset": max_tasks,
            "tolerance": tolerance,
            "dataset_name": dataset_name,
            "total_subsets": len(configs),
            "successful_subsets": successful_subsets,
            "failed_subsets": failed_subsets,
        },
        "overall_stats": {
            "total_tokens_used": total_tokens,
            "total_estimated_cost": round(total_cost, 4),
            "avg_cost_per_subset": round(total_cost / successful_subsets, 4) if successful_subsets > 0 else 0,
            "avg_tokens_per_subset": round(total_tokens / successful_subsets, 0) if successful_subsets > 0 else 0,
        },
        "subset_results": all_results,
    }

    # Save summary
    summary_path = run_dir / "run_summary.json"
    import json

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Final report
    logger.info(f"\n{'=' * 80}")
    logger.info("🏁 BENCHMARK RUN COMPLETED")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total subsets processed: {len(configs)}")
    logger.info(f"Successful: {successful_subsets}")
    logger.info(f"Failed: {failed_subsets}")
    logger.info(f"Total tokens used: {total_tokens:,}")
    logger.info(f"Total estimated cost: ${total_cost:.4f}")
    logger.info(f"Results saved in: {run_dir}")
    logger.info(f"Summary saved to: {summary_path}")

    if failed_subsets > 0:
        logger.warning(f"\n⚠️  {failed_subsets} subsets failed. Check logs for details.")
        failed_configs = [r["subset"] for r in all_results if r["status"] == "error"]
        logger.warning(f"Failed subsets: {failed_configs}")


def run_quick_test(num_subsets: int = 3, max_tasks: int = 2) -> None:
    """Run a quick test on a few subsets with limited tasks."""

    configs = ["fsm_traversal_text_len5", "decay_chain_text_len5", "peak_sorting_text_n5"][:num_subsets]

    logger.info(f"Running quick test on {len(configs)} subsets with {max_tasks} tasks each")

    run_all_benchmarks(
        configs=configs,
        max_tasks=max_tasks,
        delay=1,  # Faster for testing
        output_dir="benchmark_test_results",
    )


if __name__ == "__main__":
    # All dataset subsets
    configs = [
        "decay_chain_text_len10",
        "decay_chain_text_len25",
        "decay_chain_text_len5",
        "decay_chain_text_len50",
        "diffusion_path_text_grid10",
        "diffusion_path_text_grid25",
        "diffusion_path_text_grid5",
        "diffusion_path_text_grid50",
        "fsm_traversal_text_len10",
        "fsm_traversal_text_len25",
        "fsm_traversal_text_len5",
        "fsm_traversal_text_len50",
        "kinematic_motion_text_n10",
        "kinematic_motion_text_n25",
        "kinematic_motion_text_n5",
        "kinematic_motion_text_n50",
        "knights_knaves_text_n10",
        "knights_knaves_text_n15",
        "knights_knaves_text_n5",
        "particle_energy_text_2d_n10",
        "particle_energy_text_2d_n25",
        "particle_energy_text_2d_n5",
        "particle_energy_text_2d_n50",
        "particle_energy_text_3d_n10",
        "particle_energy_text_3d_n25",
        "particle_energy_text_3d_n5",
        "particle_energy_text_3d_n50",
        "peak_sorting_text_n10",
        "peak_sorting_text_n100",
        "peak_sorting_text_n25",
        "peak_sorting_text_n5",
        "peak_sorting_text_n50",
        "tree_traversal_text_n10",
        "tree_traversal_text_n25",
        "tree_traversal_text_n5",
        "tree_traversal_text_n50",
    ]

    # Choose what to run:

    # Option 1: Quick test (uncomment to use)
    # run_quick_test(num_subsets=3, max_tasks=2)

    run_all_benchmarks(
        configs=configs,
        dataset_name="n0w0f/scirex-text",
        model_name="gemini-2.5-flash",
        temperature=0.0,
        thinking_budget=0,  # No thinking tokens
        delay=1,
        max_tasks=None,  # Limit to 5 tasks per subset for testing
        tolerance=0.01,
        output_dir="benchmark_results",
    )

    # Option 3: Full benchmark on all tasks (uncomment when ready)
    # run_all_benchmarks(
    #     configs=configs,
    #     max_tasks=None,  # All tasks
    #     delay=2,
    #     output_dir="full_benchmark_results"
    # )
