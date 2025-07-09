# Quick Start

Get up and running with SciRex in 5 minutes.

## Basic Text-Only Benchmark

```python
from scirex import Dataset, GeminiModel, Benchmark

# Initialize components
model = GeminiModel("gemini-2.5-flash")
benchmark = Benchmark(model)

# Load a text-only dataset (replace with your dataset)
dataset = Dataset("your-org/chemistry-dataset")

# Run benchmark
results = benchmark.run_benchmark(dataset, max_tasks=5)

# View results
summary = benchmark.compute_summary_metrics(results)
print(f"Accuracy: {summary['success_rate']:.2%}")
```

## Multimodal Benchmark

```python
from scirex import Dataset, GeminiModel, Benchmark

# Initialize with multimodal support
model = GeminiModel("gemini-2.5-flash")
benchmark = Benchmark(model, test_multimodal=True)

# Load multimodal dataset
dataset = Dataset("jablonkagroup/MaCBench", "material_science")

# Run benchmark - automatically handles multimodal content
results = benchmark.run_benchmark(dataset, max_tasks=5)

# Analyze by content type
multimodal_results = [r for r in results if r.is_multimodal]
text_results = [r for r in results if not r.is_multimodal]

print(f"Multimodal tasks: {len(multimodal_results)}")
print(f"Text-only tasks: {len(text_results)}")
```

## Save Results

```python
# Save detailed results to JSON
benchmark.save_results_to_json(results, "my_benchmark_results.json")
```

## Next Steps

- [Learn about text benchmarks](../guides/text-benchmark.md)
- [Explore multimodal capabilities](../guides/multimodal-benchmark.md)
- [Check the Task API](../api-reference/task.md)
