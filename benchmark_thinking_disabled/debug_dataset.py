"""
Quick fix to run a single subset with debugging to identify the issue.
Run this before running the full benchmark to test the fix.
"""

from scirex.task import Dataset
from scirex.benchmark import Benchmark  # Use the fixed benchmark
from scirex.model import GeminiModel
from scirex.prompt import PromptTemplate
from google.genai import types
from loguru import logger


def test_single_subset_with_debug():
    """Test a single problematic subset with full debugging."""

    dataset_name = "n0w0f/scirex-text"
    subset = "decay_chain_text_len10"

    print(f"=== Testing {subset} with Enhanced Debugging ===\n")

    try:
        # Step 1: Load dataset with debugging
        print("Step 1: Loading dataset...")
        dataset = Dataset(dataset_name, subset)

        if not dataset.tasks:
            print("❌ No tasks loaded!")
            return False

        print(f"✅ Successfully loaded {len(dataset.tasks)} tasks")

        # Examine first task
        task = dataset.tasks[0]
        print(f"\nFirst task details:")
        print(f"  UUID: {task.uuid}")
        print(f"  Name: {task.name}")
        print(f"  Answer type: {task.answer_type}")
        print(f"  Target: {task.target}")
        print(f"  Target type: {type(task.target)}")
        print(f"  Is multimodal: {task.is_multimodal}")
        print(f"  Question: {task.question[:200]}...")

        # Step 2: Test model creation with minimum thinking budget
        print(f"\nStep 2: Creating model with minimum thinking budget...")
        model = GeminiModel(
            model_name="gemini-2.5-flash",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=1000),  # Minimum budget
            delay=1,  # Faster for testing
        )
        print(f"✅ Model created successfully")

        # Step 3: Test benchmark creation
        print(f"\nStep 3: Creating benchmark...")
        prompt_template = PromptTemplate()
        benchmark = Benchmark(
            model=model, prompt_template=prompt_template, tolerance=0.01, include_thoughts=False, test_multimodal=False
        )
        print(f"✅ Benchmark created successfully")

        # Step 4: Test single task
        print(f"\nStep 4: Testing single task...")
        result = benchmark.run_single_task(task)

        if result.success:
            print(f"✅ Task completed successfully!")
            print(f"  Expected: {result.task.target}")
            print(f"  Predicted: {result.parsed_answer}")
            print(f"  Accuracy: {result.metrics.get('accuracy', 'N/A')}")
            print(f"  Tokens used: {result.total_tokens}")
        else:
            print(f"❌ Task failed!")
            return False

        # Step 5: Test small benchmark
        print(f"\nStep 5: Testing small benchmark (2 tasks)...")
        results = benchmark.run_benchmark(dataset, max_tasks=2)

        successful = [r for r in results if r.success]
        print(f"✅ Benchmark completed: {len(successful)}/{len(results)} tasks successful")

        if successful:
            avg_accuracy = sum(r.metrics.get("accuracy", 0) for r in successful) / len(successful)
            print(f"  Average accuracy: {avg_accuracy:.3f}")
            total_tokens = sum(r.total_tokens for r in successful)
            print(f"  Total tokens: {total_tokens}")

        return True

    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multiple_subsets_quick():
    """Test multiple subsets with just 1 task each to identify patterns."""

    test_subsets = [
        "decay_chain_text_len10",  # The problematic one
        "fsm_traversal_text_len10",  # Known working one
        "peak_sorting_text_n10",  # Different type
        "decay_chain_text_len5",  # Same family, different size
    ]

    print(f"=== Quick Test of Multiple Subsets ===\n")

    # Create model once with minimum thinking budget
    model = GeminiModel(
        model_name="gemini-2.5-flash",
        temperature=0.0,
        thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=1000),
        delay=1,
    )

    results = {}

    for subset in test_subsets:
        print(f"Testing {subset}...")
        try:
            # Load dataset
            dataset = Dataset("n0w0f/scirex-text", subset)

            if not dataset.tasks:
                results[subset] = {"status": "❌ No tasks loaded", "error": "No tasks"}
                continue

            # Create benchmark
            benchmark = Benchmark(model, test_multimodal=False)

            # Test one task
            result = benchmark.run_single_task(dataset.tasks[0])

            if result.success:
                results[subset] = {
                    "status": "✅ Success",
                    "tasks_loaded": len(dataset.tasks),
                    "answer_type": dataset.tasks[0].answer_type,
                    "target": str(dataset.tasks[0].target),
                    "predicted": str(result.parsed_answer),
                    "accuracy": result.metrics.get("accuracy", "N/A"),
                }
            else:
                results[subset] = {
                    "status": "❌ Task failed",
                    "tasks_loaded": len(dataset.tasks),
                    "answer_type": dataset.tasks[0].answer_type,
                    "error": "Task execution failed",
                }

        except Exception as e:
            results[subset] = {"status": f"❌ Error: {str(e)[:100]}", "error": str(e)}

    # Summary
    print(f"\n=== Results Summary ===")
    for subset, result in results.items():
        print(f"{subset}: {result['status']}")
        if "tasks_loaded" in result:
            print(f"  Tasks: {result['tasks_loaded']}, Type: {result.get('answer_type', 'N/A')}")
            if "target" in result:
                print(f"  Expected: {result['target']}, Got: {result.get('predicted', 'N/A')}")

    # Count successes
    successes = sum(1 for r in results.values() if "✅" in r.get("status", ""))
    print(f"\nSuccess rate: {successes}/{len(test_subsets)}")

    return results


if __name__ == "__main__":
    print("Running diagnostic tests with fixed code...\n")

    # Enable debug logging
    logger.add("debug.log", level="DEBUG", rotation="1 MB")

    # Test the specific problematic case first
    success = test_single_subset_with_debug()

    if success:
        print(f"\n✅ Single subset test passed! The fix works.")
        print(f"\nNow testing multiple subsets...")

        test_multiple_subsets_quick()

        print(f"\n🎉 If all tests passed, you can now run the full benchmark!")

    else:
        print(f"\n❌ Single subset test failed. Check the debug.log for more details.")
        print(f"The error might be in the prompt formatting or evaluation logic.")
