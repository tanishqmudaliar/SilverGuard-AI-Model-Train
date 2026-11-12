"""
Silver Guard — Upload Trained Model Artifacts to Hugging Face
============================================================
Uploads the new FastText Subword Domain Fusion ONNX model and metadata
to the Hugging Face model repository:
  https://huggingface.co/tanishqmudaliar/SilverGuard

Usage:
  python upload_to_hf.py
  # Reads HUGGINGFACE_API_TOKEN from .env automatically
  # Or via CLI flag:
  python upload_to_hf.py --token <HF_WRITE_TOKEN>
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi

# Load environment variables from root .env if available
ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"

if ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ENV_FILE)
    except ImportError:
        # Fallback basic parser if python-dotenv is not installed
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def upload_model(token: str = None, repo_id: str = "tanishqmudaliar/SilverGuard") -> None:
    token = (
        token
        or os.environ.get("HUGGINGFACE_API_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        print("\n❌ Hugging Face Write Token is required!")
        print("   Set HUGGINGFACE_API_TOKEN in your root .env file:")
        print("     HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxx")
        print("   Or provide it via CLI:")
        print("     python upload_to_hf.py --token hf_xxxxxxxxxxxxxxxxx")
        print("\n   Get your Write access token here: https://huggingface.co/settings/tokens")
        sys.exit(1)

    api = HfApi(token=token)
    print(f"\nConnecting to Hugging Face repository '{repo_id}'...")
    try:
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("fullname") or "User"
        print(f"✅ Authenticated as: {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("   Please verify your token has 'write' permissions.")
        sys.exit(1)

    # Files to upload to Hugging Face
    files_to_upload = [
        ("model/silverguard_fasttext_fusion.onnx", "silverguard_fasttext_fusion.onnx"),
        ("model/silverguard_fasttext_fusion.onnx.data", "silverguard_fasttext_fusion.onnx.data"),
        ("model/model_config.json", "model_config.json"),
        ("model/HF_README.md", "README.md")
    ]

    for local_path, repo_path in files_to_upload:
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"\nUploading {local_path} ({size_mb:.2f} MB) -> {repo_path}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"docs/feat: update {repo_path} (Silver Guard 4-Class FastText Domain Fusion v2.0)"
            )
            print(f"  • {repo_path} uploaded successfully!")
        else:
            print(f"⚠️ Warning: File {local_path} not found, skipping.")

    # Clean legacy model files (MobileBERT artifacts no longer used)
    legacy_files_to_remove = ["silver_guard.onnx", "vocab.txt"]
    try:
        current_repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        for legacy_file in legacy_files_to_remove:
            if legacy_file in current_repo_files:
                print(f"\nRemoving deprecated legacy file: {legacy_file}...")
                api.delete_file(
                    path_in_repo=legacy_file,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"chore: remove legacy {legacy_file} (superseded by FastText Domain Fusion)"
                )
                print(f"  • {legacy_file} removed from repository.")
    except Exception as e:
        print(f"⚠️ Warning during legacy file cleanup: {e}")

    print(f"\n🎉 All model files and Model Card successfully updated at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Silver Guard Model to Hugging Face")
    parser.add_argument("--token", default=None, help="Hugging Face Write Access Token (hf_...)")
    parser.add_argument("--repo-id", default="tanishqmudaliar/SilverGuard", help="Hugging Face repo ID")
    args = parser.parse_args()
    upload_model(token=args.token, repo_id=args.repo_id)
