import argparse
import os
from huggingface_hub import snapshot_download


def download_dataset(dataset_id: str, cache_dir: str):
    """
    Download the OSV-5M dataset files to a local folder using a memory-safe
    file transfer. This does not load the dataset into RAM.
    """
    print(f"[Download] Starting memory-safe download of '{dataset_id}' to {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)

    # snapshot_download simply fetches the files from the Hub to your cache.
    # It is much lighter on RAM than load_dataset.
    path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        cache_dir=cache_dir,
        resume_download=True,
    )
    print(f"[Download] Finished! Files are stored in: {path}")
    print("[Download] You can now run your training/notebook with --local_files_only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-safe download of OSV-5M.")
    parser.add_argument("--dataset_id", type=str, default="osv5m/osv5m", help="HuggingFace dataset ID")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory to save the dataset")

    args = parser.parse_args()

    download_dataset(args.dataset_id, args.cache_dir)
