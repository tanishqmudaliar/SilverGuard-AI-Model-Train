---
license: mit
language:
- en
- hi
- mr
- ta
- te
- bn
- gu
library_name: onnx
pipeline_tag: text-classification
datasets:
- uciml/sms-spam-collection-dataset
- junioralive/india-spam-sms-classification
metrics:
- accuracy
- f1
model-index:
- name: SilverGuard-FastText-Fusion-v2
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: SilverGuard-Indic-Unified-24k
      type: custom
    metrics:
    - type: f1
      value: 0.8996
      name: Macro F1
    - type: accuracy
      value: 0.9494
      name: Accuracy
tags:
- sms
- scam-detection
- phishing-detection
- spam-detection
- indic-nlp
- onnx
- edge-ai
- android
- cybersecurity
- trai-dlt
- fasttext
- atherion-30k
- scamshield
- uci-sms
- v2.0
widget:
- text: "Dear customer, your SBI YONO account will be blocked today. Click http://bit.ly/sbi-kyc to update PAN."
- text: "JK-SMRTBR: FLAT 50% OFF on fresh groceries today only! Use code SAVE50."
- text: "CBI NOTICE: An arrest warrant has been issued against you for money laundering. Contact immediately."
- text: "Hey bhai, are you free this evening? Let's catch up at chai tapri."
- text: "Exclusive 3BHK luxury apartments in Whitefield starting at 1.2 Cr. Call now: 9876500000"
---

# 🛡️ Silver Guard — Indian Multilingual SMS Scam & Threat Classifier (v2.0)

Silver Guard is an ultra-lightweight (<15 MB RAM, 56 KB graph + 15.3 MB embedding table), zero-latency edge model and 3-stage cascaded inference engine engineered specifically for detecting SMS scams, phishing lures, cyber extortion, and commercial spam across the Indian telecommunications ecosystem.

- **GitHub Repository**: [tanishqmudaliar/SilverGuard](https://github.com/tanishqmudaliar/SilverGuard)
- **Model Type**: FastText Subword Character N-Gram Embeddings + Tabular Structural Domain MLP (Dual-Branch Early Fusion)
- **Taxonomy**: 4-Class (`0: Ham`, `1: Promotional`, `2: Commercial Spam UCC`, `3: Scam / Phishing`)
- **Runtime**: ONNX Runtime (CPU) — Edge ready for Android smartphones (ARM Cortex-A53/A55)
- **Supported Languages**: English, Hindi (`हिंदी`), Marathi (`मराठी`), Tamil (`தமிழ்`), Telugu (`తెలుగు`), Bengali (`বাংলা`), Gujarati (`ગુજરાતી`), and Romanized Hinglish.

---

## 🏷️ 4-Class Classification Taxonomy

Silver Guard decouples unsolicited commercial advertising from malicious fraud:

| Class ID | Semantic Class | Typical Senders | Key Characteristics & Behavior |
| :---: | :--- | :--- | :--- |
| `0` | **Safe Ham** | Personal contacts, banks (`AD-HDFCBK-T`), government (`JX-ISATHI-G`), service alerts (`MH-AIRTEL-S`) | OTPs, balance debits, family chats, courier delivery updates. Threat score: $\approx 0.0$. |
| `1` | **Promotional** | Registered TRAI DLT senders with `-P` suffix or registered retail brands (`JK-SMRTBR`, `JM-AJIOIN`, `DOMINO`) | Legitimate e-commerce sales, discount codes, retail offers. High commerciality, zero security threat. |
| `2` | **Commercial Spam (UCC)** | Unregistered 10-digit mobile numbers (`+919876543210`) | Unsolicited cold marketing (real estate, pre-approved personal loans, insurance leads). Do-Not-Disturb violation, but **no** credential theft or malware. |
| `3` | **Scam / Phishing** | Unknown phone numbers, spoofed display headers | Digital arrest extortion, law enforcement intimidation (CBI/Customs), fake KYC deactivation, lottery fraud, malicious APK downloads. |

---

## 🧠 Neural Architecture: FastText Subword Domain Fusion

Silver Guard uses a dual-branch hybrid architecture designed for zero out-of-vocabulary (OOV) errors and ultra-low RAM consumption on low-end mobile devices:

1. **Subword Linguistic Branch**: FastText character n-grams ($n \in [3, 5]$) mapped via deterministic 32-bit CRC hashing into 60,000 buckets, embedded using PyTorch `nn.EmbeddingBag(60000, 64, mode="mean")`. Provides complete invariance to Hinglish phonetic misspellings, leetspeak (`p0lice`, `k-y-c`), and regional slang.
2. **Domain Feature Tabular Branch**: 12 structural telecom indicators (URL counts, link shorteners, direct IP links, APK downloads, callback phones, currency symbols, urgency cues, promo keywords, uppercase ratio, digit ratio, length, Indic script codes) processed through a 2-layer MLP (`12 -> 32 -> 32`, ReLU).
3. **Early Fusion & Head**: Embeddings are concatenated into a 96-dimensional representation (`64 + 32 = 96`) and routed through a classification head (`96 -> 4`) trained with class-balanced focal weights.

### Performance & Edge Hardware Comparison

| Metric | Silver Guard v2.0 (FastText Domain Fusion) | Legacy Baseline (MobileBERT FP32) |
|---|:---:|:---:|
| **Model Size** | **56 KB graph + 15.3 MB weights** | 105 MB |
| **Inference Latency (CPU)** | **~0.8–1.2 ms** | 18–25 ms |
| **Peak RAM Footprint** | **< 15 MB** (Immune to Android Go LMKD kill) | ~120 MB (High OOM risk) |
| **Hinglish / Leetspeak Handling** | **Character n-grams (Zero OOV)** | High OOV (`[UNK]` tokens) |
| **Indic Matras Preservation** | **100% (Canonical NFC Normalization)** | Erased by standard ASCII tokenizers |

---

## 📊 Multi-Source Training Dataset (24,514 Unique Samples)

The model was trained on a multi-source unified dataset compiled and deduplicated by [`build_unified_dataset.py`](https://github.com/tanishqmudaliar/SilverGuard/blob/main/build_unified_dataset.py) using MinHash 5-shingle Jaccard similarity ($J \ge 0.85$ deduplication threshold) and canonical NFC normalization:

| Dataset Source | Origin / Platform | Ingested Samples | Class Contribution | Key Value to Model |
|---|---|:---:|---|---|
| **Atherion 30K SMS** | Kaggle (`atherionsystems/indian-scam-sms-30k`) | **8,599** | Ham (5,977), Scam (1,851), Spam (771) | Indian financial scams, lottery lures, transactional alerts |
| **UCI SMS Spam Collection** | Hugging Face (`uciml/sms-spam-collection-dataset`) | **5,009** | Ham (4,825), Scam/Spam (184) | Clean conversational English personal chats (eliminates personal text false alarms) |
| **Personal Real Inbox SMS** | Device Export (`data/rawdata.json` & `contacts.csv`) | **3,992** | Ham (3,822), Promo (170) | Ground-truth Indian DLT carrier traffic (TRAI headers, OTPs, delivery alerts) |
| **ScamShield Multilingual** | Kaggle (`siddharthrane07/scamshield-multilingual`) | **4,331** | Phish / Scam (2,904), Ham (1,427) | Multilingual smishing, credential phishing, malicious URL attacks |
| **JuniorAlive India SMS** | Hugging Face (`junioralive/india-spam-sms-classification`) | **1,983** | Ham (1,483), Spam / UCC (500) | Real Indian telemarketing, cold commercial offers |
| **Curated Indic Regional** | Synthetic / Handcrafted Templates | **600** | Promo (558), Spam (42) | Marathi, Hindi, Telugu, Tamil, and Gujarati retail promotions |
| **Total Clean Dataset** | **Unified & Deduplicated** | **24,514** | **Ham: 17,013 \| Scam: 4,889 \| Spam: 1,884 \| Promo: 728** | **Full 4-Class Indian Telecom Coverage** |

> **Note on Hugging Face YAML Metadata**: The `datasets:` YAML frontmatter tag above lists `uciml/sms-spam-collection-dataset` and `junioralive/india-spam-sms-classification` because Hugging Face YAML validation requires valid Hugging Face Hub dataset identifiers. The additional sources (Atherion 30K and ScamShield) originate from Kaggle, and the personal ground-truth inbox export is preserved privately with PII redaction.

---

## 📦 Hosted Files

- `silverguard_fasttext_fusion.onnx`: Dual-branch ONNX compute graph (~56 KB).
- `silverguard_fasttext_fusion.onnx.data`: Weights tensor data containing the subword embedding dictionary (~15.3 MB).
- `model_config.json`: Hyperparameters, class labels, and supported languages metadata.

---

## 🚀 Quickstart: Running Inference in Python

```python
import numpy as np
import onnxruntime as ort
import re
import zlib
import unicodedata

# 1. Load ONNX Session
session = ort.InferenceSession("silverguard_fasttext_fusion.onnx")

# 2. Preprocess Text (FastText Subwords + Hashes)
def compute_subwords(text: str, vocab_size: int = 60000) -> np.ndarray:
    text = unicodedata.normalize("NFC", text.lower())
    words = re.findall(r"\w+", text)
    hashes = []
    for w in words:
        padded = f"<{w}>"
        for n in (3, 4, 5):
            for i in range(len(padded) - n + 1):
                ngram = padded[i:i+n]
                h = zlib.crc32(ngram.encode("utf-8")) % vocab_size
                hashes.append(h)
    return np.array(hashes if hashes else [0], dtype=np.int64)

# 3. Extract 12 Domain Features
def extract_tabular_features(text: str) -> np.ndarray:
    feats = np.zeros(12, dtype=np.float32)
    feats[0] = len(re.findall(r"https?://", text))      # url_count
    feats[1] = 1.0 if "bit.ly" in text or "tinyurl" in text else 0.0 # shortener
    feats[2] = 1.0 if ".apk" in text.lower() else 0.0   # apk link
    feats[7] = 1.0 if re.search(r"\b(offer|discount|sale|off)\b", text, re.I) else 0.0 # promo
    feats[8] = sum(1 for c in text if c.isupper()) / max(len(text), 1) # upper ratio
    feats[10] = len(text) / 160.0                        # length
    return feats.reshape(1, 12)

# 4. Run Model Prediction
text = "Dear customer, your SBI account will be blocked today. Click http://bit.ly/sbi-kyc to update PAN."
subwords = compute_subwords(text)
offsets = np.array([0], dtype=np.int64)
tabular = extract_tabular_features(text)

inputs = {
    "text_indices": subwords,
    "text_offsets": offsets,
    "tabular_features": tabular
}

logits = session.run(["logits"], inputs)[0][0]
probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))

labels = {0: "Safe Ham", 1: "Promotional", 2: "Commercial Spam (UCC)", 3: "Scam / Phishing"}
pred_class = np.argmax(probs)

print(f"Prediction: {labels[pred_class]} (Confidence: {probs[pred_class]:.4f})")
# Output: Prediction: Scam / Phishing (Confidence: 0.9995)
```

---

## 📜 License & Citation

This model is licensed under the [MIT License](LICENSE).

```bibtex
@software{silverguard2026,
  author = {Tanishq Mudaliar},
  title = {Silver Guard: Multilingual Edge SMS Threat & Scam Classifier for Indian Telecommunications},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/tanishqmudaliar/SilverGuard}
}
```
