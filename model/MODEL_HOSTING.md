# Silver Guard Model Distribution & Hosting

## 📌 Architecture & Model Release (v2.0)

Silver Guard uses the **FastText Subword Domain Fusion** architecture (`FastTextDomainFusion`), designed specifically for zero-latency, 100% offline edge execution on low-end Android smartphones across India.

- **Primary Architecture**: Dual-Branch Subword Character N-gram Bag (`nn.EmbeddingBag` with 60,000 hashed buckets, 64-dim) + Tabular Structural Domain MLP (12 features, 32-dim) with Early Fusion classification head.
- **Taxonomy**: 4-Class (`0: Ham`, `1: Promotional`, `2: Commercial Spam UCC`, `3: Scam / Phishing`).
- **RAM Footprint**: < 15 MB RAM runtime (compared to > 120 MB for MobileBERT).
- **Inference Latency**: ~0.8–1.2 ms per message on device CPU.
- **Binary Model Weight**: 56 KB ONNX graph + 15.36 MB embedding table.

> **💡 Git Tracking Note**: Unlike the legacy 105.3 MB MobileBERT model (which exceeded GitHub's 100 MB per-file boundary), the current Silver Guard ONNX model (~15.4 MB total) and GBDT model (~1.5 MB) are **tracked directly in Git** within this repository under the `model/` directory. Cloning this repo provides the trained model out-of-the-box with no separate download required.

---

## 🤖 Hugging Face Model Repository (Mirror)

The complete model artifacts and configuration are also published on the Hugging Face Hub for external integrations, mobile build pipelines, and public sharing:

👉 **[https://huggingface.co/tanishqmudaliar/SilverGuard](https://huggingface.co/tanishqmudaliar/SilverGuard)**

---

## 📦 Model Artifacts (Included in Git & Hugging Face)

| File | Description | Size | Location |
|---|---|:---:|---|
| `silverguard_fasttext_fusion.onnx` | Base ONNX compute graph for dual-branch inference | ~56 KB | `model/` & HF Hub |
| `silverguard_fasttext_fusion.onnx.data` | External tensor data containing subword embedding weights | ~15.36 MB | `model/` & HF Hub |
| `silverguard_gbdt.joblib` | LightGBM multi-class decision tree ensemble model | ~1.50 MB | `model/` & HF Hub |
| `char_vectorizer.joblib` | Subword character n-gram TF-IDF vectorizer | ~112 KB | `model/` & HF Hub |
| `model_config.json` | Label mapping, hyperparameters, and Indic language metadata | < 1 KB | `model/` & HF Hub |
| `gbdt_metadata.json` | GBDT training metadata, class counts, and evaluation metrics | < 1 KB | `model/` & HF Hub |

---

## 🚀 Running Inference Locally

Since the model files are tracked in Git, you can run inference immediately after cloning the repository:

```powershell
# Interactive mode
python silverguard.py

# Non-interactive CLI check
python silverguard.py --header "JK-SMRTBR" --message "Flat 50% discount on fresh groceries today only!"
```

To pull the latest weights from Hugging Face if working outside the Git repository:
```python
from huggingface_hub import hf_hub_download

for filename in ["silverguard_fasttext_fusion.onnx", "silverguard_fasttext_fusion.onnx.data", "model_config.json"]:
    hf_hub_download(repo_id="tanishqmudaliar/SilverGuard", filename=filename, local_dir="model")
```

---

## 📤 Publishing Updates to Hugging Face (`hf upload`)

Updates to the Hugging Face repository are published using the official Hugging Face Hub CLI (`hf`):

```powershell
# 1. One-time authentication (if not already logged in)
hf auth login

# 2. Upload model files and metadata
hf upload tanishqmudaliar/SilverGuard model/ . `
    --include "*.onnx" `
    --include "*.onnx.data" `
    --include "*.joblib" `
    --include "*.json" `
    --exclude "MODEL_HOSTING.md" `
    --commit-message "feat: update Silver Guard model artifacts"

# 3. Update Model Card README
hf upload tanishqmudaliar/SilverGuard model/HF_README.md README.md `
    --commit-message "docs: update Model Card README"
```

