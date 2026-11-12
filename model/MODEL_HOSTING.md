# Silver Guard Model Hosting (Hugging Face)

## 📌 Architecture & Model Release (v2.0)

Silver Guard uses the **FastText Subword Domain Fusion** architecture (`FastTextDomainFusion`), designed specifically for zero-latency, 100% offline edge execution on low-end Android smartphones across India.

- **Primary Architecture**: Dual-Branch Subword Character N-gram Bag (`nn.EmbeddingBag` with 60,000 hashed buckets, 64-dim) + Tabular Structural Domain MLP (12 features, 32-dim) with Early Fusion classification head.
- **Taxonomy**: 4-Class (`0: Ham`, `1: Promotional`, `2: Commercial Spam UCC`, `3: Scam / Phishing`).
- **RAM Footprint**: < 15 MB RAM runtime (compared to > 120 MB for MobileBERT).
- **Inference Latency**: ~0.8 ms per message on device CPU.
- **Binary Model Weight**: 56 KB ONNX graph + 15.3 MB embedding table.

---

## 🤖 Hugging Face Model Repository

The complete model artifacts and configuration are hosted at:

👉 **[https://huggingface.co/tanishqmudaliar/SilverGuard](https://huggingface.co/tanishqmudaliar/SilverGuard)**

---

## 📦 Hosted Files

| File | Description | Size |
|---|---|:---:|
| `silverguard_fasttext_fusion.onnx` | Base ONNX compute graph for dual-branch inference | ~56 KB |
| `silverguard_fasttext_fusion.onnx.data` | External tensor data containing the subword embedding weights | ~15.3 MB |
| `model_config.json` | Label mapping, hyperparameters, and Indic language support metadata | < 1 KB |

---

## 🚀 How to Download & Run Inference

1. Download the hosted files from Hugging Face:
   ```bash
   git lfs install
   git clone https://huggingface.co/tanishqmudaliar/SilverGuard
   ```
   Or download directly via Python:
   ```python
   from huggingface_hub import hf_hub_download
   import shutil

   for filename in ["silverguard_fasttext_fusion.onnx", "silverguard_fasttext_fusion.onnx.data", "model_config.json"]:
       hf_hub_download(repo_id="tanishqmudaliar/SilverGuard", filename=filename, local_dir="model")
   ```

2. Place them inside the `model/` directory:
   ```text
   model/
   ├── silverguard_fasttext_fusion.onnx
   ├── silverguard_fasttext_fusion.onnx.data
   ├── model_config.json
   └── MODEL_HOSTING.md
   ```

3. Run inference directly in Python:
   ```bash
   python silverguard.py --text "Dear customer, your SBI YONO account will be blocked today. Click http://bit.ly/sbi-kyc to update PAN." --sender "+919876543210"
   ```

---

## 📤 How to Upload Updated Models

To upload or update model files on Hugging Face using the command line:

```powershell
# If HUGGINGFACE_API_TOKEN is set in your root .env file, simply run:
python upload_to_hf.py

# Alternatively, pass the token directly as a flag:
python upload_to_hf.py --token "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Or set as an environment variable in PowerShell:
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python upload_to_hf.py
```

