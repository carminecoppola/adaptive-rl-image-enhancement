"""
Main entry point for sanity checks.

At this stage, this script only verifies that configuration files,
environment variables, and storage paths are correctly configured.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def main() -> None:
    # Load environment variables from .env file
    load_dotenv()
    
    # Define expected environment variables
    expected_vars = [
        "HPC_STORAGE_ROOT",
        "DATA_ROOT",
        "DATASET_ROOT",
        "PROCESSED_DATA_ROOT",
        "CHECKPOINT_ROOT",
        "RESULTS_ROOT",
        "LOGS_ROOT",
        "HF_HOME",
        "TORCH_HOME",
    ]
    
    # Verify and display paths
    print("Configured paths:")
    paths = {}
    for var in expected_vars:
        value = os.environ.get(var)
        if value is None:
            print(f"  ⚠️  {var}: NOT SET")
        else:
            paths[var] = value
            print(f"  ✓ {var}: {value}")
    
    # Create directories if they don't exist
    print("\nCreating directories...")
    dirs_to_create = [
        paths.get("DATASET_ROOT"),
        paths.get("CHECKPOINT_ROOT"),
        paths.get("RESULTS_ROOT"),
        paths.get("LOGS_ROOT"),
        os.path.join(paths.get("LOGS_ROOT", ""), "slurm"),
        os.path.join(paths.get("LOGS_ROOT", ""), "dqn"),
    ]
    
    for dir_path in dirs_to_create:
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {dir_path}")


if __name__ == "__main__":
    main()