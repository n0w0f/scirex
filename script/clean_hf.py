from datasets import load_dataset_builder, load_dataset, DatasetDict

# 1. Define the dataset source and the destination repository ID.
source_dataset_id = "n0w0f/scirex-text"
# You must specify a single repository to push all configs to.
# This will be the name of your new dataset on the Hub.
destination_repo_id = "n0w0f/scirex-text"

configs = [
    "decay_chain_text_len10",
    "decay_chain_text_len25",
    "decay_chain_text_len5",
    "decay_chain_text_len50",
    "diffusion_path_text_grid10",
    "diffusion_path_text_grid25",
    "diffusion_path_text_grid5",
    "diffusion_path_text_grid50",
    "dna_translation_text_len10",
    "dna_translation_text_len25",
    "dna_translation_text_len5",
    "dna_translation_text_len50",
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
    "particle_energy_2d",
    "particle_energy_3d",
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

# 3. Loop through each configuration.
for config_name in configs:
    print(f"\nProcessing config: {config_name}")

    try:
        # Load the specific configuration.
        d = load_dataset(source_dataset_id, config_name)
        train_split = d["train"]

        # Check the number of entries and process accordingly.
        if len(train_split) > 15:
            print(f"  - Original size: {len(train_split)}. Reducing to 15 entries.")
            processed_train_split = train_split.shuffle(seed=42).select(range(15))
        else:
            print(f"  - Size is {len(train_split)}. No reduction needed.")
            processed_train_split = train_split

        # Push the processed 'train' split to the Hub.
        # We specify the `config_name` to push it to the same repository
        # with its original name.
        print(f"  - Pushing {config_name} to {destination_repo_id}...")
        processed_train_split.push_to_hub(destination_repo_id, config_name=config_name)

    except Exception as e:
        print(f"  - An error occurred while processing {config_name}: {e}")
        continue

print("\nAll available configurations have been processed and pushed to the same Hugging Face Hub repository.")
