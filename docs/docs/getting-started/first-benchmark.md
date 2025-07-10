# Your First Benchmark

A step-by-step tutorial to run your first scientific benchmark with SciRex.

## Prerequisites

- [SciRex installed](installation.md)
- Google API key configured
- Basic Python knowledge

## Step 1: Import SciRex

```python
from scirex.task import Dataset
from scirex.model import GeminiModel
from scirex.benchmark import Benchmark
from scirex.prompt import PromptTemplate
```

## Step 2: Setup Your Model

```python
# Initialize Gemini model
model = GeminiModel(
    model_name="gemini-2.5-flash",  # Fast model for testing
    delay=1  # Reduce delay for faster development
)
```

## Step 3: Create a Benchmark

```python
# Initialize benchmark with custom settings
benchmark = Benchmark(
    model=model,
    tolerance=0.01,  # For numeric tasks
    include_thoughts=False,  # Disable to save tokens
    test_multimodal=True  # Test if model supports images
)
```

## Step 4: Load a Dataset

```python
# Option A: Load existing dataset
dataset = Dataset("jablonkagroup/MaCBench", "isomers")

# Option B: Create custom task (example)
from scirex.task import Task

custom_task = Task(
    uuid="demo-001",
    name="Basic Chemistry",
    description="Test basic chemistry knowledge",
    question="What is the molecular formula of water?",
    answer_type="mcq",
    target={"H2O": 1.0, "CO2": 0.0, "CH4": 0.0},
    keywords=["chemistry", "basic"]
)

# Create dataset from custom tasks
class MockDataset:
    def __init__(self, tasks):
        self.tasks = tasks

dataset = MockDataset([custom_task])
```

## Step 5: Run the Benchmark

```python
# Run on a small subset first
results = benchmark.run_benchmark(dataset, max_tasks=3)

print(f"Completed {len(results)} tasks")
print(f"Success rate: {len([r for r in results if r.success])}/{len(results)}")
```

## Step 6: Analyze Results

```python
# Get detailed metrics
summary = benchmark.compute_summary_metrics(results)
print(f"Overall accuracy: {summary['success_rate']:.2%}")

# Look at individual results
for i, result in enumerate(results[:3]):
    print(f"\nTask {i+1}: {result.task.name}")
    print(f"  Question: {result.task.question[:100]}...")
    print(f"  Answer: {result.parsed_answer}")
    print(f"  Accuracy: {result.metrics['accuracy']:.2f}")
    print(f"  Tokens used: {result.total_tokens}")
```

## Step 7: Save Your Results

```python
# Save results for later analysis
benchmark.save_results_to_json(results, "my_first_benchmark.json")
print("Results saved to my_first_benchmark.json")
```

## Understanding the Output

### Task Types

- **Text-only**: Traditional Q&A tasks
- **Multimodal**: Tasks with images + text

### Metrics

- **Accuracy**: Percentage of correct answers
- **Token Usage**: Cost estimation
- **Success Rate**: Tasks processed without errors

### Example Output

```
Processing task 1/3 (text): Basic Chemistry
Processing task 2/3 (multimodal): Crystal Structure Analysis
Processing task 3/3 (text): Molecular Properties

Benchmark completed!
Results: 3 tasks processed
Successful: 3
Text results: 2
Multimodal results: 1

Summary Metrics:
  Overall accuracy: 0.67
  Text accuracy: 0.50
  Multimodal accuracy: 1.00
```

## Common Issues

### API Key Not Set

```
ValueError: API Key not set.
```

**Solution**: Check your `.env` file has `GOOGLE_API_KEY=your_key`

### Model Doesn't Support Multimodal

```
Multimodal support: ✗
```

**Solution**: Use `gemini-2.5-flash` or `gemini-2.5-pro` for vision tasks

### Rate Limiting

```
Error: Rate limit exceeded
```

**Solution**: Increase delay: `GeminiModel(delay=2)`

## Next Steps

Now that you've run your first benchmark:

1. **[Explore text benchmarks](../guides/text-benchmark.md)** for traditional Q&A
2. **[Try multimodal benchmarks](../guides/multimodal-benchmark.md)** for image tasks
3. **[Check the API reference](../api-reference/task.md)** for advanced usage

## Tips for Success

- **Start small**: Use `max_tasks=5` for testing
- **Monitor costs**: Check token usage in results
- **Validate data**: Ensure your dataset format is correct
- **Save frequently**: Use `save_results_to_json()` for checkpoints
