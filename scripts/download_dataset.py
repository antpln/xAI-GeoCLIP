import argparse
import os
from datasets import load_dataset


def download_dataset(dataset_id: str, split: str, cache_dir: str):
    """Download the OSV-5M dataset to a local folder."""
    print(f"[Download] Preparing to download '{dataset_id}' split='{split}' to {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)

    # We use normal loading (not streaming) to ensure all shards are fully downloaded
    # and placed in the local cache.
    hf = load_dataset(dataset_id, split=split, cache_dir=cache_dir, trust_remote_code=True)
    print(f"[Download] Finished! Loaded {len(hf)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the OSV-5M dataset to local storage.")
    parser.add_argument("--dataset_id", type=str, default="osv5m/osv5m", help="HuggingFace dataset ID")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory to save the dataset")

    args = parser.parse_args()

    download_dataset(args.dataset_id, args.split, args.cache_dir)
