# Multimodal Benchmarks

This guide covers everything you need to know about running benchmarks with multimodal content (text + images) using SciRex.

## Overview

Multimodal benchmarks evaluate how well language models can understand and reason about combined text and visual information. This is particularly important in scientific domains where:

- **Chemical structures** need visual interpretation
- **Microscopy images** require analysis alongside textual descriptions
- **Experimental setups** are described through diagrams
- **Data visualizations** need interpretation in context

## Quick Start

```python
from scirex.task import Dataset
from scirex.model import GeminiModel
from scirex.benchmark import Benchmark
from scirex.prompt import PromptTemplate

# Initialize model with vision capabilities
model = GeminiModel("gemini-2.5-flash")  # or gemini-2.5-pro

# Create benchmark with multimodal testing
benchmark = Benchmark(model, test_multimodal=True)

# Load multimodal dataset
dataset = Dataset("jablonkagroup/MaCBench", "material_science")

# Run benchmark - automatically handles multimodal content
results = benchmark.run_benchmark(dataset, max_tasks=50)

# Analyze results
summary = benchmark.compute_summary_metrics(results)
print(f"Multimodal accuracy: {summary.get('multimodal_accuracy', 0):.2%}")
```

## Understanding Multimodal Data Format

### Input Structure

Multimodal tasks use a template-based approach where placeholders are replaced with actual content:

```json
{
  "input": "{type1} {entry1} shows a crystal structure. What is the space group?",
  "qentries_modality": {
    "image": {
      "type1": { "type": "text", "value": "The diffraction pattern" },
      "entry1": {
        "type": "image",
        "value": "data:image/png;base64,iVBORw0KGgo..."
      }
    }
  },
  "selected_entries": ["image"],
  "target_scores": { "P63/mmc": 1.0, "Fm-3m": 0.0, "Pnma": 0.0 }
}
```

### Content Resolution

SciRex automatically resolves templates into model-ready content:

1. **Template**: `"{type1} {entry1} shows a crystal structure"`
2. **Resolved**: `["The diffraction pattern", <Image>, "shows a crystal structure"]`
3. **API Call**: Mixed content array sent to model

## Supported Image Formats

### Base64 Encoding

All images must be base64-encoded. SciRex supports:

```python
# Data URL format (preferred)
"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB..."

# Raw base64 (automatically detected as PNG)
"iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB..."
```

### MIME Types

Automatically detected from data URLs:

- `image/png` - Recommended for diagrams, charts
- `image/jpeg` - Good for photographs, microscopy
- `image/gif` - For simple animations
- `image/webp` - Modern format, smaller files

### Image Processing

```python
import base64
from PIL import Image
import io

def image_to_base64(image_path):
    """Convert image file to base64 string."""
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        b64_string = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/png;base64,{b64_string}"

def resize_image_for_llm(image_path, max_size=(1024, 1024)):
    """Resize image to optimal size for LLM processing."""
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = buffer.getvalue()
        b64_string = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/png;base64,{b64_string}"
```

## Creating Custom Multimodal Tasks

### Single Image Task

```python
from scirex.task import Task

task = Task(
    uuid="crystal-001",
    name="Crystal Structure Analysis",
    description="Identify crystal system from diffraction pattern",
    question="",  # Auto-filled from template
    answer_type="mcq",
    target={"cubic": 1.0, "hexagonal": 0.0, "tetragonal": 0.0},
    keywords=["crystallography", "diffraction"],
    is_multimodal=True,
    input_template="Analyze {image_type} {image_data}. What crystal system does this represent?",
    qentries_modality={
        "xrd": {
            "image_type": {"type": "text", "value": "this X-ray diffraction pattern"},
            "image_data": {"type": "image", "value": "data:image/png;base64,..."}
        }
    },
    selected_entries=["xrd"]
)
```

### Multiple Images Task

```python
task = Task(
    uuid="comparison-001",
    name="Before/After Comparison",
    description="Compare material before and after treatment",
    answer_type="numeric",
    target=15.2,  # Percentage change
    is_multimodal=True,
    input_template="Compare {before_desc} {before_img} with {after_desc} {after_img}. What is the percentage change in grain size?",
    qentries_modality={
        "microscopy": {
            "before_desc": {"type": "text", "value": "the initial microstructure"},
            "before_img": {"type": "image", "value": "data:image/png;base64,..."},
            "after_desc": {"type": "text", "value": "the treated microstructure"},
            "after_img": {"type": "image", "value": "data:image/png;base64,..."}
        }
    }
)
```

## Advanced Configuration

### Model Selection

Choose models based on multimodal capabilities:

```python
# Best for multimodal tasks
model = GeminiModel("gemini-2.5-pro")  # Highest accuracy

# Faster for development
model = GeminiModel("gemini-2.5-flash")  # Good balance

# Check capabilities
benchmark = Benchmark(model, test_multimodal=True)
if benchmark.multimodal_supported:
    print("✓ Model supports vision")
else:
    print("✗ Model is text-only")
```

### Prompt Optimization

Customize prompts for better multimodal performance:

```python
from scirex.prompt import PromptTemplate

custom_templates = {
    "mcq_multimodal": """Analyze the provided scientific image(s) and text.

{content}

Choose the best answer from the options below:
{options}

Think step by step:
1. Examine the visual features
2. Consider the textual context
3. Apply relevant scientific principles
4. Select the most appropriate option

Answer with only the letter(s):"",

    "numeric_multimodal": """Examine the scientific image(s) and accompanying text.

{content}

Provide a numerical answer based on your analysis.

Guidelines:
- Look for quantitative features in the image
- Use scale bars or reference objects when present
- Apply appropriate measurement techniques
- Give your answer as a number only (no units)

Answer:"""
}

prompt_template = PromptTemplate(custom_templates)
benchmark = Benchmark(model, prompt_template)
```

### Performance Optimization

```python
# Batch processing for large datasets
def run_multimodal_benchmark_batched(dataset, batch_size=10):
    results = []

    for i in range(0, len(dataset.tasks), batch_size):
        batch = dataset.tasks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset.tasks)-1)//batch_size + 1}")

        batch_results = []
        for task in batch:
            result = benchmark.run_single_task(task)
            batch_results.append(result)

        results.extend(batch_results)

        # Save intermediate results
        if i > 0 and i % (batch_size * 5) == 0:
            benchmark.save_results_to_json(results, f"checkpoint_{i}.json")

    return results

# Memory management for large images
def optimize_images_for_benchmark(dataset):
    """Resize images to optimal dimensions for processing."""
    for task in dataset.tasks:
        if task.is_multimodal and task.qentries_modality:
            for category, entries in task.qentries_modality.items():
                for key, entry in entries.items():
                    if entry.get('type') == 'image':
                        # Resize to max 1024x1024 to reduce token usage
                        entry['value'] = resize_image_for_llm(entry['value'])
```

## Evaluation Metrics

### Standard Metrics

Multimodal tasks use the same metrics as text-only tasks:

```python
# MCQ tasks
{
    "accuracy": 0.85,    # Exact match
    "precision": 0.87,   # True positives / (TP + FP)
    "recall": 0.83,      # True positives / (TP + FN)
    "f1": 0.85          # Harmonic mean of precision and recall
}

# Numeric tasks
{
    "accuracy": 0.72,    # Within tolerance
    "mae": 1.34         # Mean absolute error
}
```

### Modality-Specific Analysis

```python
results = benchmark.run_benchmark(dataset)

# Separate by modality
text_results = [r for r in results if not r.is_multimodal]
multimodal_results = [r for r in results if r.is_multimodal]

# Compare performance
text_acc = sum(r.metrics['accuracy'] for r in text_results) / len(text_results)
multimodal_acc = sum(r.metrics['accuracy'] for r in multimodal_results) / len(multimodal_results)

print(f"Text-only accuracy: {text_acc:.2%}")
print(f"Multimodal accuracy: {multimodal_acc:.2%}")
print(f"Performance gap: {(text_acc - multimodal_acc):.2%}")

# Analyze by content type
content_breakdown = {}
for result in multimodal_results:
    content_type = result.content_type
    if content_type not in content_breakdown:
        content_breakdown[content_type] = []
    content_breakdown[content_type].append(result.metrics['accuracy'])

for content_type, accuracies in content_breakdown.items():
    avg_acc = sum(accuracies) / len(accuracies)
    print(f"{content_type}: {avg_acc:.2%} ({len(accuracies)} tasks)")
```

## Troubleshooting

### Common Issues

!!! warning "Model Compatibility"

    ```python
    # Check if model supports multimodal input
    if not benchmark.multimodal_supported:
        print("Model doesn't support vision - using text fallback")
        # SciRex automatically falls back to text-only versions
    ```

!!! error "Image Size Limits"

    ```python
    # Images too large cause API errors
    # Solution: Resize images before encoding

    def check_image_size(base64_data):
        """Check if base64 image is within size limits."""
        # Remove data URL prefix
        if 'base64,' in base64_data:
            actual_data = base64_data.split('base64,')[1]
        else:
            actual_data = base64_data

        # Calculate approximate file size
        size_bytes = len(actual_data) * 3 / 4  # Base64 overhead
        size_mb = size_bytes / (1024 * 1024)

        if size_mb > 20:  # Gemini limit
            print(f"Warning: Image is {size_mb:.1f}MB, consider resizing")
            return False
        return True
    ```

!!! tip "Performance Issues"

    ```python
    # Reduce delay for development
    model = GeminiModel("gemini-2.5-flash", delay=0.5)

    # Use smaller test sets
    test_dataset = Dataset("your-dataset")
    results = benchmark.run_benchmark(test_dataset, max_tasks=5)

    # Monitor token usage
    total_tokens = sum(r.total_tokens for r in results)
    print(f"Total tokens used: {total_tokens:,}")
    ```

### Debugging Tools

```python
# Preview multimodal content resolution
from scirex.prompt import PromptTemplate

prompt_template = PromptTemplate()
preview = prompt_template.preview_prompt(task, "mcq")
print("Prompt preview:")
print(preview)

# Examine resolved content
resolved = task.resolve_multimodal_content()
print(f"Content parts: {len(resolved)}")
for i, part in enumerate(resolved):
    if isinstance(part, str):
        print(f"  {i}: Text - {part[:100]}...")
    else:
        print(f"  {i}: Image - {type(part)}")

# Test individual task
result = benchmark.run_single_task(task)
print(f"Success: {result.success}")
print(f"Response: {result.response}")
print(f"Parsed: {result.parsed_answer}")
print(f"Metrics: {result.metrics}")
```

## Best Practices

### 1. Image Quality

- **Resolution**: 512-1024px optimal for most tasks
- **Format**: PNG for diagrams, JPEG for photos
- **Contrast**: Ensure good contrast for text/features
- **Scale**: Include scale bars when relevant

### 2. Template Design

- **Clear Context**: Provide sufficient textual context
- **Specific Questions**: Ask precise, answerable questions
- **Logical Flow**: Order text and images logically
- **Error Handling**: Account for missing or corrupted images

### 3. Evaluation Strategy

- **Balanced Sets**: Mix text-only and multimodal tasks
- **Domain Coverage**: Include diverse scientific domains
- **Difficulty Levels**: Range from basic to expert-level tasks
- **Quality Control**: Manually verify a subset of results

## Next Steps

- **Create Custom Datasets**: Create your own multimodal evaluation sets
- **Advanced Usage**: Power user features and optimizations
- **[Task API Reference](../api-reference/task.md)**: Detailed Task class documentation
