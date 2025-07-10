import base64
import json
import os
from io import BytesIO
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from huggingface_hub import HfApi, login
from PIL import Image


class SimpleHuggingFaceUploader:
    """Simple uploader for HuggingFace datasets with subset support."""

    def __init__(self, token: str = None):
        """Initialize uploader with HuggingFace token."""
        if token:
            login(token=token)
        self.api = HfApi()

    def base64_to_pil(self, base64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]

        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data))

    def prepare_text_dataset(self, json_file: str) -> Dataset:
        """Prepare text-only dataset for HuggingFace."""
        with open(json_file, "r") as f:
            data = json.load(f)

        examples = data["examples"]

        # Simple feature definition
        rows = []
        for i, example in enumerate(examples):
            example_str = json.dumps(example)

            row = {
                "uuid": f"{data['uuid']}-{i}",
                "canary": data["canary"],
                "name": data["name"],
                "description": data["description"],
                "keywords": data["keywords"],
                "preferred_score": data["preferred_score"],
                "metrics": data["metrics"],
                "examples": [example_str],
                "subfield": data["subfield"],
                "relative_tolerance": None,
            }
            rows.append(row)

        return Dataset.from_list(rows)

    def prepare_multimodal_dataset(self, json_file: str) -> Dataset:
        """Prepare multimodal dataset for HuggingFace."""
        with open(json_file, "r") as f:
            data = json.load(f)

        examples = data["examples"]

        rows = []
        for i, example in enumerate(examples):
            # Extract and convert image
            image_base64 = example["qentries_modality"]["image"]["entry1"]["value"]
            pil_image = self.base64_to_pil(image_base64)

            example_str = json.dumps(example)

            row = {
                "uuid": f"{data['uuid']}-{i}",
                "image": pil_image,
                "canary": data["canary"],
                "name": data["name"],
                "description": data["description"],
                "keywords": data["keywords"],
                "preferred_score": data["preferred_score"],
                "metrics": data["metrics"],
                "examples": [example_str],
                "subfield": data["subfield"],
                "relative_tolerance": None,
            }
            rows.append(row)

        return Dataset.from_list(rows)

    def upload_as_config(self, dataset: Dataset, repo_id: str, config_name: str, private: bool = False):
        """Upload dataset as a configuration."""

        print(f"Uploading to {repo_id} with config '{config_name}'...")

        # Create DatasetDict with train split
        dataset_dict = DatasetDict({"train": dataset})

        # Push to hub
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            config_name=config_name,
            private=private,
            token=True,
        )

        print(f"✅ Successfully uploaded {config_name} to {repo_id}")
        return repo_id

    def create_simple_readme(self, repo_id: str, config_name: str, dataset_type: str, num_examples: int):
        """Create a simple README."""

        readme_content = f"""---
title: {repo_id.split("/")[-1]}
tags:
- dataset
- physics
- energy-computation
{f"- multimodal" if dataset_type == "multimodal" else "- text"}
license: mit
configs:
- {config_name}
---

# {repo_id.split("/")[-1]}

Energy computation dataset for 2D particle systems.

## Configuration: {config_name}

**Type**: {dataset_type.title()}
**Examples**: {num_examples}

**Usage:**
```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}", name="{config_name}")
example = dataset['train'][0]
print(example)
```

## Task
Calculate total energy of 10 particles in a 2D system (10×10 Angstrom box).

Energy = Particle Energy + Pairwise Interactions (within 5 Angstrom cutoff)
"""

        try:
            self.api.upload_file(
                path_or_fileobj=BytesIO(readme_content.encode()),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add README for {config_name}",
            )
            print(f"✅ README updated")
        except Exception as e:
            print(f"⚠️ Could not update README: {e}")

    def verify_upload(self, repo_id: str, config_name: str):
        """Verify that the upload worked."""
        try:
            from datasets import load_dataset

            print(f"Verifying {repo_id} config '{config_name}'...")

            dataset = load_dataset(repo_id, name=config_name)
            train_data = dataset["train"]

            print(f"✅ Verification successful!")
            print(f"   Examples: {len(train_data)}")
            print(f"   Features: {list(train_data.features.keys())}")

            # Show sample
            if len(train_data) > 0:
                sample = train_data[0]
                print(f"   Sample UUID: {sample['uuid']}")
                if "image" in sample:
                    print(f"   Has image: {sample['image'] is not None}")

        except Exception as e:
            print(f"❌ Verification failed: {e}")


def main():
    """Main function to upload both 2D and 3D datasets."""

    # Configuration
    TEXT_REPO = "n0w0f/scirex-text"
    MULTIMODAL_REPO = "n0w0f/scirex-image"
    PRIVATE = False

    # Dataset configurations
    datasets_to_upload = [
        {
            "config_name": "particle_energy_2d",
            "text_file": "energy_computation_text_2d.json",
            "multimodal_file": "energy_computation_multimodal_2d.json",
            "dimension": "2D",
        },
        {
            "config_name": "particle_energy_3d",
            "text_file": "energy_computation_text_3d.json",
            "multimodal_file": "energy_computation_multimodal_3d.json",
            "dimension": "3D",
        },
    ]

    # Check all files exist
    missing_files = []
    for dataset_config in datasets_to_upload:
        if not os.path.exists(dataset_config["text_file"]):
            missing_files.append(dataset_config["text_file"])
        if not os.path.exists(dataset_config["multimodal_file"]):
            missing_files.append(dataset_config["multimodal_file"])

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        print("Run the dataset generator first.")
        return

    print("🚀 UPLOADING 2D + 3D DATASETS")
    print("=" * 50)

    uploader = SimpleHuggingFaceUploader()

    try:
        uploaded_configs = []

        for dataset_config in datasets_to_upload:
            config_name = dataset_config["config_name"]
            text_file = dataset_config["text_file"]
            multimodal_file = dataset_config["multimodal_file"]
            dimension = dataset_config["dimension"]

            print(f"\n{'=' * 20} {dimension} DATASETS {'=' * 20}")

            # Upload text dataset
            print(f"\n📝 Uploading {dimension} TEXT dataset...")
            text_dataset = uploader.prepare_text_dataset(text_file)
            uploader.upload_as_config(dataset=text_dataset, repo_id=TEXT_REPO, config_name=config_name, private=PRIVATE)
            uploader.create_simple_readme(TEXT_REPO, config_name, f"text-{dimension.lower()}", len(text_dataset))
            uploader.verify_upload(TEXT_REPO, config_name)

            # Upload multimodal dataset
            print(f"\n🖼️ Uploading {dimension} MULTIMODAL dataset...")
            multimodal_dataset = uploader.prepare_multimodal_dataset(multimodal_file)
            uploader.upload_as_config(
                dataset=multimodal_dataset, repo_id=MULTIMODAL_REPO, config_name=config_name, private=PRIVATE
            )
            uploader.create_simple_readme(
                MULTIMODAL_REPO, config_name, f"multimodal-{dimension.lower()}", len(multimodal_dataset)
            )
            uploader.verify_upload(MULTIMODAL_REPO, config_name)

            uploaded_configs.append(config_name)

        print(f"\n{'=' * 50}")
        print("🎉 ALL UPLOADS COMPLETE!")
        print(f"{'=' * 50}")

        print(f"\nUploaded configurations: {uploaded_configs}")
        print(f"\nText dataset: https://huggingface.co/datasets/{TEXT_REPO}")
        print(f"Multimodal dataset: https://huggingface.co/datasets/{MULTIMODAL_REPO}")

        print(f"\nUsage examples:")
        for config in uploaded_configs:
            dimension = "2D" if "2d" in config else "3D"
            print(f"\n# {dimension} datasets:")
            print(f"text_{config.split('_')[-1]} = load_dataset('{TEXT_REPO}', name='{config}')")
            print(f"multimodal_{config.split('_')[-1]} = load_dataset('{MULTIMODAL_REPO}', name='{config}')")

        print(f"\nNote: Subsets may take a few minutes to appear in the UI.")

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Verify the repository exists and you have write access")


if __name__ == "__main__":
    main()
