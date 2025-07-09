# Installation

## Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project
uv init my-benchmark-project
cd my-benchmark-project

# Add scirex as a dependency
uv add git+https://github.com/n0w0f/scirex.git

# Install additional dependencies
uv add python-dotenv google-genai datasets pillow
```

## Using pip

```bash
pip install git+https://github.com/n0w0f/scirex.git
pip install python-dotenv google-genai datasets pillow
```

## Setup

Create a `.env` file with your API key:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Verify Installation

```python
from scirex import Dataset, GeminiModel, Benchmark
print("✅ SciRex installed successfully!")
```
