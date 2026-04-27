import argparse
import os
from huggingface_hub import snapshot_download


def download_dataset(dataset_id: str, cache_dir: str, include_patterns: list = None):
    """
    Download specific parts of the OSV-5M dataset files to a local folder.
    """
    if include_patterns:
        print(f"[Download] Starting partial download of '{dataset_id}' matching {include_patterns} to {cache_dir}...")
    else:
        print(f"[Download] Starting full download of '{dataset_id}' to {cache_dir}...")
    
    os.makedirs(cache_dir, exist_ok=True)
    if include_patterns:
        essential = ["osv5m.py", "*.csv", ".gitattributes"]
        include_patterns.extend(essential)

    path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        cache_dir=cache_dir,
        allow_patterns=include_patterns,
        resume_download=True,
    )
    print(f"[Download] Finished! Files are stored in: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-safe partial download of OSV-5M.")
    parser.add_argument("--dataset_id", type=str, default="osv5m/osv5m", help="HuggingFace dataset ID")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory to save the dataset")
    parser.add_argument("--include", type=str, nargs="+", help="Glob patterns of files to include (e.g. 'images/train/0*.zip')")

    args = parser.parse_args()

    download_dataset(args.dataset_id, args.cache_dir, args.include)
