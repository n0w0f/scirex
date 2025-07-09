# Task API Reference

The `Task` class represents a single benchmark task, supporting both text-only and multimodal content.

## Class Definition

```python
@dataclass
class Task:
    """Simple task representation with multimodal support."""

    # Core fields
    uuid: str
    name: str
    description: str
    question: str
    answer_type: str  # "mcq" or "numeric"
    target: float | dict[str, float]
    keywords: list[str]

    # Multimodal fields
    is_multimodal: bool = False
    input_template: Optional[str] = None
    qentries_modality: Optional[Dict[str, Any]] = None
    selected_entries: Optional[List[str]] = None
    resolved_content: Optional[List[Any]] = None
```

## Core Attributes

### `uuid: str`

Unique identifier for the task.

**Example:**

```python
task.uuid  # "63f8f16d-e2f4-0ad1-3b03-d05586d56686"
```

### `name: str`

Human-readable name describing the task.

**Example:**

```python
task.name  # "Molecular Formula Identification"
```

### `description: str`

Detailed description of what the task evaluates.

**Example:**

```python
task.description  # "Identify the molecular formula from a chemical structure diagram"
```

### `question: str`

The actual question text. For multimodal tasks, this is auto-generated from the template.

**Example:**

```python
# Text-only task
task.question  # "What is the molecular formula of benzene?"

# Multimodal task (auto-resolved)
task.question  # "Analyze this molecular structure [IMAGE]. What is its formula?"
```

### `answer_type: str`

Type of expected answer. Must be either `"mcq"` (multiple choice) or `"numeric"`.

**Example:**

```python
task.answer_type  # "mcq" or "numeric"
```

### `target: float | dict[str, float]`

Expected answer(s) for the task.

**For numeric tasks:**

```python
task.target  # 42.5
```

**For MCQ tasks:**

```python
task.target  # {"C6H6": 1.0, "C6H12": 0.0, "C6H14": 0.0}
```

### `keywords: list[str]`

List of keywords describing the task domain/topic.

**Example:**

```python
task.keywords  # ["chemistry", "organic", "molecular_formula"]
```

## Multimodal Attributes

### `is_multimodal: bool`

Flag indicating whether the task contains multimodal content.

**Example:**

```python
task.is_multimodal  # True for image+text tasks, False for text-only
```

### `input_template: Optional[str]`

Template string with placeholders for multimodal content.

**Example:**

```python
task.input_template  # "Analyze {type1} {entry1}. What compound is shown?"
```

### `qentries_modality: Optional[Dict[str, Any]]`

Dictionary containing the actual content to replace template placeholders.

**Structure:**

```python
{
    "category_name": {
        "placeholder1": {"type": "text", "value": "actual text"},
        "placeholder2": {"type": "image", "value": "data:image/png;base64,..."}
    }
}
```

**Example:**

```python
task.qentries_modality = {
    "molecule": {
        "type1": {"type": "text", "value": "this molecular structure"},
        "entry1": {"type": "image", "value": "data:image/png;base64,iVBORw0..."}
    }
}
```

### `selected_entries: Optional[List[str]]`

List of which modality categories are active for this task.

**Example:**

```python
task.selected_entries  # ["molecule", "spectrum"]
```

### `resolved_content: Optional[List[Any]]`

Cached result of template resolution. Automatically populated by `resolve_multimodal_content()`.

**Example:**

```python
task.resolved_content  # ["Analyze this molecular structure", <ImagePart>, ". What compound is shown?"]
```

## Methods

### `resolve_multimodal_content() -> List[Any]`

Resolves template placeholders with actual content, returning a list suitable for API calls.

**Returns:**

- `List[Any]`: Mixed content list with strings and image dictionaries

**Example:**

```python
# Template: "Analyze {type1} {entry1}. What is its formula?"
# Modality: {"type1": {"type": "text", "value": "this structure"},
#           "entry1": {"type": "image", "value": "data:image/png;base64,..."}}

content = task.resolve_multimodal_content()
# Returns: ["Analyze this structure ", {"type": "image", "data": "..."}, ". What is its formula?"]
```

**Caching:**
Results are cached in `resolved_content` for performance.

### `get_text_content() -> str`

Returns a text-only representation of the task content, suitable for parsing and analysis.

**Returns:**

- `str`: Text representation with `[IMAGE]` placeholders

**Example:**

```python
# Multimodal task
text_version = task.get_text_content()
# Returns: "Analyze this structure [IMAGE]. What is its formula?"

# Text-only task
text_version = task.get_text_content()
# Returns: "What is the molecular formula of benzene?"
```

**Use Cases:**

- Response parsing (LLM parsers need text input)
- Debugging and logging
- Text-only fallbacks

## Factory Methods

### Creating Text-Only Tasks

```python
from scirex.task import Task

task = Task(
    uuid="text-001",
    name="Basic Chemistry Q&A",
    description="Test knowledge of basic chemical formulas",
    question="What is the molecular formula of water?",
    answer_type="mcq",
    target={"H2O": 1.0, "CO2": 0.0, "CH4": 0.0},
    keywords=["chemistry", "molecular_formula"],
    is_multimodal=False
)
```

### Creating Multimodal Tasks

```python
task = Task(
    uuid="multimodal-001",
    name="Molecular Structure Analysis",
    description="Identify molecular formula from structure diagram",
    question="",  # Will be auto-filled
    answer_type="mcq",
    target={"C6H6": 1.0, "C6H12": 0.0, "C6H14": 0.0},
    keywords=["chemistry", "organic", "structure"],
    is_multimodal=True,
    input_template="Identify the molecular formula of {desc} {image}.",
    qentries_modality={
        "structure": {
            "desc": {"type": "text", "value": "this organic compound"},
            "image": {"type": "image", "value": "data:image/png;base64,..."}
        }
    },
    selected_entries=["structure"]
)

# Auto-resolve question text
task.question = task.get_text_content()
```

## Integration Examples

### With Dataset Loading

```python
from scirex.task import Dataset

# Tasks are automatically created during dataset loading
dataset = Dataset("chemistry-benchmark")
for task in dataset.tasks:
    print(f"Task: {task.name}")
    print(f"Type: {'Multimodal' if task.is_multimodal else 'Text-only'}")
    print(f"Answer type: {task.answer_type}")
```

### With Benchmark Execution

```python
from scirex import Benchmark, GeminiModel

model = GeminiModel("gemini-2.5-flash")
benchmark = Benchmark(model)

# Single task execution
result = benchmark.run_single_task(task)
print(f"Success: {result.success}")
print(f"Answer: {result.parsed_answer}")

# Batch execution
results = benchmark.run_benchmark(dataset)
```

### With Custom Prompt Templates

```python
from scirex.prompt import PromptTemplate

prompt_template = PromptTemplate()

# Generate prompts from tasks
if task.answer_type == "mcq":
    prompt = prompt_template.format_mcq_prompt(task)
else:
    prompt = prompt_template.format_numeric_prompt(task)

# For multimodal tasks, prompt will be List[Any]
# For text tasks, prompt will be str
```

## Validation and Error Handling

### Required Fields Validation

```python
def validate_task(task: Task) -> bool:
    """Validate that a task has all required fields."""

    # Check core fields
    required_fields = ['uuid', 'name', 'description', 'answer_type', 'target', 'keywords']
    for field in required_fields:
        if not getattr(task, field):
            print(f"Missing required field: {field}")
            return False

    # Validate answer_type
    if task.answer_type not in ['mcq', 'numeric']:
        print(f"Invalid answer_type: {task.answer_type}")
        return False

    # Validate target format
    if task.answer_type == 'mcq' and not isinstance(task.target, dict):
        print("MCQ tasks require dict target with option scores")
        return False

    if task.answer_type == 'numeric' and not isinstance(task.target, (int, float)):
        print("Numeric tasks require numeric target")
        return False

    # Validate multimodal fields
    if task.is_multimodal:
        if not task.input_template:
            print("Multimodal tasks require input_template")
            return False
        if not task.qentries_modality:
            print("Multimodal tasks require qentries_modality")
            return False

    return True

# Usage
if validate_task(task):
    print("✓ Task is valid")
else:
    print("✗ Task validation failed")
```

### Common Error Patterns

```python
try:
    # Resolve multimodal content
    content = task.resolve_multimodal_content()

except AttributeError as e:
    if "qentries_modality" in str(e):
        print("Task missing multimodal fields - check is_multimodal flag")
    else:
        raise

except KeyError as e:
    print(f"Template placeholder not found in modality data: {e}")

except ValueError as e:
    if "base64" in str(e):
        print("Invalid base64 image data")
    else:
        raise
```

## Performance Considerations

### Memory Usage

```python
# Large images can consume significant memory
def estimate_task_memory(task: Task) -> float:
    """Estimate memory usage in MB for a task."""

    if not task.is_multimodal:
        return 0.1  # Minimal for text-only

    total_mb = 0.1  # Base overhead

    for category, entries in task.qentries_modality.items():
        for key, entry in entries.items():
            if entry.get('type') == 'image':
                base64_data = entry.get('value', '')
                # Remove data URL prefix
                if 'base64,' in base64_data:
                    actual_data = base64_data.split('base64,')[1]
                else:
                    actual_data = base64_data

                # Estimate decoded size
                bytes_size = len(actual_data) * 3 / 4
                mb_size = bytes_size / (1024 * 1024)
                total_mb += mb_size

    return total_mb

# Check before processing large batches
for task in dataset.tasks:
    size_mb = estimate_task_memory(task)
    if size_mb > 20:  # Typical API limit
        print(f"Warning: Task {task.uuid} is {size_mb:.1f}MB")
```

### Batch Processing

```python
def process_tasks_by_type(tasks: List[Task]):
    """Process tasks efficiently by grouping by type."""

    text_tasks = [t for t in tasks if not t.is_multimodal]
    multimodal_tasks = [t for t in tasks if t.is_multimodal]

    print(f"Processing {len(text_tasks)} text tasks...")
    # Text tasks are faster, process first

    print(f"Processing {len(multimodal_tasks)} multimodal tasks...")
    # Multimodal tasks use more tokens/time
```

## Related Classes

- **[Dataset](dataset.md)**: Manages collections of tasks
- **[Benchmark](benchmark.md)**: Executes tasks and produces results
- **[PromptTemplate](prompt.md)**: Formats tasks into model prompts
- **[GeminiModel](model.md)**: Handles multimodal content generation
