# scirex

[![Release](https://img.shields.io/github/v/release/n0w0f/scirex)](https://img.shields.io/github/v/release/n0w0f/scirex)
[![codecov](https://codecov.io/gh/n0w0f/scirex/branch/main/graph/badge.svg)](https://codecov.io/gh/n0w0f/scirex)
[![Commit activity](https://img.shields.io/github/commit-activity/m/n0w0f/scirex)](https://img.shields.io/github/commit-activity/m/n0w0f/scirex)
[![License](https://img.shields.io/github/license/n0w0f/scirex)](https://img.shields.io/github/license/n0w0f/scirex)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://n0w0f.github.io/scirex)

# SciRex: Scientific Research Benchmarking Framework

A Python framework for benchmarking large language models on scientific research tasks with support for both text-only and multimodal content.

## Features

- 🔬 **Scientific Focus**: Designed for chemistry, materials science, and other scientific domains
- 🖼️ **Multimodal Support**: Handle text + image tasks seamlessly
- 🚀 **Easy Integration**: Simple API for custom datasets and models

- 📊 **Agent for Reward** : (Incoming)
- 📊 **CoT Faithfulness** : (Incoming)

## Installation

### Using uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project
uv init scirex
cd scirex

# Add scirex as a dependency
uv add git+https://github.com/n0w0f/scirex.git

```

### Using pip

```bash
pip install git+https://github.com/n0w0f/scirex.git
```

## Quick Start

### Setup

Create a `.env` file with your API key:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Text-Only Benchmark

```python
from scirex.task import Dataset
from scirex.model import GeminiModel
from scirex.benchmark import Benchmark
from scirex.prompt import PromptTemplate


model = GeminiModel("gemini-2.5-flash")
prompt_template = PromptTemplate()
benchmark = Benchmark(model, prompt_template)


dataset = Dataset("n0w0f/scirex-text", "particle_energy_2d")

# Run benchmark
results = benchmark.run_benchmark(dataset, max_tasks=10)

# View results
summary = benchmark.compute_summary_metrics(results)
print(f"Overall accuracy: {summary['success_rate']:.2f}")
```

### Multimodal Benchmark

```python
from scirex.task import Dataset
from scirex.model import GeminiModel
from scirex.benchmark import Benchmark
from scirex.prompt import PromptTemplate

model = GeminiModel("gemini-2.5-flash")  # Ensure model supports vision
benchmark = Benchmark(model, test_multimodal=True)

# Load multimodal dataset (images + text)
dataset = Dataset("n0w0f/scirex-image", "particle_energy_2d")

# Run benchmark - automatically detects and handles multimodal content
results = benchmark.run_benchmark(dataset, max_tasks=5)

# Save detailed results
benchmark.save_results_to_json(results, "benchmark_results.json")
```

### Custom Multimodal Task

```python
from scirex.task import Task

# Create a custom multimodal task
task = Task(
    uuid="custom-001",
    name="Molecule Analysis",
    description="Analyze molecular structure in image",
    question="",  # Auto-filled from template
    answer_type="numeric",
    target=6.0,  # Expected answer
    keywords=["chemistry", "molecules"],
    is_multimodal=True,
    input_template="Analyze {type1} {entry1}. How many carbon atoms are visible?",
    qentries_modality={
        "vision": {
            "type1": {"type": "text", "value": "this molecular structure"},
            "entry1": {"type": "image", "value": "data:image/png;base64,..."}
        }
    }
)

# Run single task
result = benchmark.run_single_task(task)
print(f"Predicted: {result.parsed_answer}, Actual: {task.target}")
```

## Supported Dataset Formats

### Text-Only Format

```json
{
  "input": "What is the molecular formula of water?",
  "target_scores": { "H2O": 1.0, "CO2": 0.0, "CH4": 0.0 }
}
```

### Multimodal Format

```json
{
  "input": "{type1} {entry1} shows a molecule. What is its formula?",
  "qentries_modality": {
    "image": {
      "type1": { "type": "text", "value": "The image" },
      "entry1": { "type": "image", "value": "data:image/png;base64,..." }
    }
  },
  "target_scores": { "H2O": 1.0, "CO2": 0.0 }
}
```

## Documentation

For detailed documentation, examples, and API reference, visit: [https://n0w0f.github.io/scirex](https://n0w0f.github.io/scirex)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{scirex2024,
  title={SciRex: Scientific Research Benchmarking Framework},
  author={Nawaf Alampara},
  year={2024},
  url={https://github.com/n0w0f/scirex}
}
```
