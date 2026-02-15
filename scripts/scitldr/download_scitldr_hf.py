import argparse
import os

from huggingface_hub import hf_hub_download, list_repo_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="allenai/scitldr")
    parser.add_argument("--subset", default="Abstract")
    parser.add_argument("--output_dir", default="data/scitldr/raw")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = list_repo_files(args.repo, repo_type="dataset")
    subset_key = args.subset.lower()

    candidates = [
        f for f in files
        if subset_key in f.lower() and f.lower().endswith(".parquet")
    ]
    if not candidates:
        raise SystemExit(f"No parquet files found for subset '{args.subset}'. Files: {files}")

    for filename in candidates:
        path = hf_hub_download(repo_id=args.repo, repo_type="dataset", filename=filename)
        out_path = os.path.join(args.output_dir, os.path.basename(filename))
        if not os.path.exists(out_path):
            os.replace(path, out_path)
        print(f"Saved {filename} -> {out_path}")


if __name__ == "__main__":
    main()
