"""
Main entry point for sanity checks.

At this stage, this script only verifies that configuration files,
environment variables, and storage paths are correctly configured.
"""

from src.utils.paths import get_paths, ensure_directories


def main() -> None:
    paths = get_paths()
    ensure_directories(paths)

    print("Configured paths:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()