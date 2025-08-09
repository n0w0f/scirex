import base64
import glob
import json
import os
from io import BytesIO
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, Features, Value
from datasets import Image as HFImage
from huggingface_hub import HfApi, HfFolder, login
from PIL import Image
from requests.exceptions import HTTPError


class GeneralHuggingFaceUploader:
    """
    A generalized uploader for HuggingFace datasets with a structured schema.
    It automatically discovers JSON files, uploads them as distinct configurations,
    and dynamically updates the repository README.
    """

    def __init__(self, token: str = None):
        """Initialize uploader with HuggingFace token."""
        if token:
            login(token=token)
        elif HfFolder.get_token():
            print("Login token found in HuggingFace cache.")
        else:
            raise ValueError(
                "No HuggingFace token provided or found in cache. Please log in via `huggingface-cli login`."
            )
        self.api = HfApi()

    def base64_to_pil(self, base64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        if "base64," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data))

    def prepare_dataset(self, json_file: str, multimodal: bool) -> Dataset:
        """
        Prepare a dataset for HuggingFace from a JSON file using the new schema.
        """
        with open(json_file, "r") as f:
            data = json.load(f)

        # Define the features based on the new schema
        feature_dict = {
            "question_template": Value("string"),
            "question_template_input": Value("string"),  # Store the dict as a JSON string
            "answer": Value("string"),
            "compute_function": Value("string"),
        }
        if multimodal:
            feature_dict["image"] = HFImage()

        features = Features(feature_dict)

        rows = []
        for example in data["examples"]:
            row = {
                "question_template": example["question_template"],
                "question_template_input": json.dumps(example["question_template_input"]),
                "answer": example["answer"],
                "compute_function": example["compute_function"],
            }
            if multimodal:
                # Handle the new 'q_entries' structure
                image_base64 = example["q_entries"]["entry1"]["value"]
                row["image"] = self.base64_to_pil(image_base64)

            rows.append(row)

        return Dataset.from_list(rows, features=features)

    def upload_config(self, dataset: Dataset, repo_id: str, config_name: str, private: bool = False):
        """Upload dataset as a configuration (subset)."""
        print(f"Uploading to {repo_id} with config '{config_name}'...")
        dataset_dict = DatasetDict({"train": dataset})
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            config_name=config_name,
            private=private,
            token=True,
        )
        print(f"✅ Successfully uploaded '{config_name}' to {repo_id}")

    def update_repo_readme(self, repo_id: str, configs_info: Dict[str, Dict]):
        """
        Dynamically update the repository's README file.
        It appends documentation for new configs without removing old ones.
        """
        readme_path_in_repo = "README.md"

        # Try to download the existing README
        try:
            readme_content = self.api.hf_hub_download(
                repo_id=repo_id, filename=readme_path_in_repo, repo_type="dataset"
            )
            readme_content = open(readme_content, "r").read()
            print("📄 Found existing README.md. Appending new configurations...")
        except HTTPError:
            print("📄 No README.md found. Creating a new one...")
            readme_content = f"""---
        title: {repo_id.split("/")[-1]}
        tags:
        - scientific-reasoning
        - benchmark
        - STEM
        license: mit
        ---

        # SciREX Benchmark Dataset: {repo_id.split("/")[-1].replace("scirex-", "").title()}

        This repository contains configurations for the SciREX (Scientific Reasoning & EXperimentation) benchmark.
        Each configuration corresponds to a specific reasoning task.

        """

        # Append new config information if not already present
        for config_name, info in configs_info.items():
            if f"config: {config_name}" not in readme_content:
                readme_content += f"""
            ---
            ## Configuration: `{config_name}`

            **Description**: {info["description"]}

            **Number of Examples**: {info["num_examples"]}

            **Keywords**: `{"`, `".join(info["keywords"])}`

            **Usage:**```python
            from datasets import load_dataset

            # Load this specific configuration
            dataset = load_dataset("{repo_id}", name="{config_name}")

            # Access the first example
            example = dataset['train']
            print(example)
            """

            # Upload the updated README
            try:
                self.api.upload_file(
                    path_or_fileobj=BytesIO(readme_content.encode()),
                    path_in_repo=readme_path_in_repo,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="docs: update README with new configurations",
                )
                print(f"✅ README for {repo_id} updated successfully.")
            except Exception as e:
                print(f"⚠️ Could not update README for {repo_id}: {e}")


def main():
    """
    Main function to discover, prepare, and upload all generated datasets.
    """
    # --- Configuration ---
    # Replace with your Hugging Face username
    HF_USERNAME = "n0w0f"
    TEXT_REPO = f"{HF_USERNAME}/scirex-text"
    MULTIMODAL_REPO = f"{HF_USERNAME}/scirex-image"
    PRIVATE = False
    # --- End Configuration ---
    uploader = GeneralHuggingFaceUploader(token=os.getenv("HF_TOKEN"))

    # 1. Discover all generated JSON files in the current directory
    json_files = glob.glob("*.json")
    if not json_files:
        print("❌ No .json dataset files found in the current directory.")
        print("Please run a dataset generator script first.")
        return

    print(f"🔍 Found {len(json_files)} JSON files to process...")

    # 2. Group files by target repository
    text_configs = {}
    multimodal_configs = {}

    for file_path in json_files:
        config_name = os.path.basename(file_path).replace(".json", "")
        with open(file_path, "r") as f:
            data = json.load(f)

        config_info = {
            "description": data.get("description", "N/A"),
            "num_examples": len(data.get("examples", [])),
            "keywords": data.get("keywords", []),
        }

        if "multimodal" in config_name:
            multimodal_configs[config_name] = {"path": file_path, "info": config_info}
        else:
            text_configs[config_name] = {"path": file_path, "info": config_info}

    # 3. Process and upload Text-Only Datasets
    if text_configs:
        print(f"\n{'=' * 20} Processing TEXT-ONLY Datasets {'=' * 20}")
        # for config_name, data in text_configs.items():
        #     print(f"\n--- Preparing '{config_name}' ---")
        #     dataset = uploader.prepare_dataset(data["path"], multimodal=False)
        #     uploader.upload_config(dataset, TEXT_REPO, config_name, private=PRIVATE)
        # # Update README once for all new text configs
        # uploader.update_repo_readme(TEXT_REPO, {k: v["info"] for k, v in text_configs.items()})

    # 4. Process and upload Multimodal Datasets
    if multimodal_configs:
        print(f"\n{'=' * 20} Processing MULTIMODAL Datasets {'=' * 20}")
        for config_name, data in multimodal_configs.items():
            print(f"\n--- Preparing '{config_name}' ---")
            dataset = uploader.prepare_dataset(data["path"], multimodal=True)
            uploader.upload_config(dataset, MULTIMODAL_REPO, config_name, private=PRIVATE)
        # Update README once for all new multimodal configs
        uploader.update_repo_readme(MULTIMODAL_REPO, {k: v["info"] for k, v in multimodal_configs.items()})

    print(f"\n{'=' * 50}")
    print("🎉 ALL TASKS COMPLETE!")
    print(f"{'=' * 50}")
    if text_configs:
        print(f"\nText dataset repository: https://huggingface.co/datasets/{TEXT_REPO}")
    if multimodal_configs:
        print(f"Multimodal dataset repository: https://huggingface.co/datasets/{MULTIMODAL_REPO}")
    print("\nNote: It may take a few minutes for all configurations to appear on the Hugging Face Hub.")


if __name__ == "__main__":
    main()
