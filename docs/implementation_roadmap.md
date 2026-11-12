# Practical Implementation Roadmap & Engineering Blueprint

## 1. Executive Summary

This document translates the empirical findings and research discoveries of the Silver Guard audit into an **actionable, phased engineering roadmap**. The objective is to transform Silver Guard from a flawed, English-centric binary classifier into a **production-ready, high-accuracy, multilingual, edge-optimized mobile SMS security system**.

The roadmap is structured into five distinct phases:
*   **Phase 0: Immediate Hotfixes & Stabilization** (Zero-risk code patches to stop bleeding).
*   **Phase 1: Dataset Harmonization & The SilverGuard-Indic-40K Benchmark** (Multi-source data pipeline).
*   **Phase 2: Dual-Engine Model Training & Calibration** (FastText + GBDT Tier 1 & IndicBERT v2 Tier 2).
*   **Phase 3: ONNX INT8 Optimization & Tokenizer Packaging** (Edge compression to $<35\text{ MB}$, FastText to $<5\text{ MB}$).
*   **Phase 4: Cascaded Inference Engine & Mobile Integration** (Production rollout to Flutter/Android).

```text
  Phase 0              Phase 1              Phase 2              Phase 3              Phase 4
Immediate           Indic-40K Dataset    Dual-Engine Train    Edge Compression     Production Rollout
Hotfixes            Curation & Rules     & Calibration        & Quantization       & Mobile App
┌──────────────┐    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│• Fix Matras  │    │• Atherion 30K│     │• FastText+GBD│     │• INT8 PTQ    │     │• 3-Stage     │
│• Fix DLT Regx│───>│• Bolewara Scm│───>│• IndicBERT v2│────>│• mmap Models │────>│  Cascade    │
│• Label Rules │    │• PII Masking │     │• Temp Scaling│     │• C++ SPM Tok │     │• Flutter Int │
└──────────────┘    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
    Week 1              Weeks 2–3            Weeks 4–5            Weeks 6–7            Weeks 8–9
```

---

## 2. Phase 0: Immediate Hotfixes & Pipeline Stabilization (Week 1)

These changes require zero architectural retraining and directly patch the most severe bugs in the current scripts.

### Task 0.1: Fix Indic Vowel Stripping in `silverguard.py`
*   **Current Defect**: `_strip_accents()` decomposes text via NFD and removes all characters in Unicode category `Mn` (Mark, nonspacing), erasing all Indic vowel signs (*matras*) and viramas.
*   **Required Fix**: Restrict accent stripping exclusively to Latin combining diacritics, or use Unicode NFC normalization without category filtering on South Asian Brahmic scripts:
```python
def normalize_multilingual_text(text: str) -> str:
    """Canonical NFC normalization that preserves Indic matras and conjunct viramas."""
    import unicodedata
    # NFC preserves precomposed characters and Brahmic vowel marks
    return unicodedata.normalize("NFC", text).strip()
```

### Task 0.2: Patch TRAI DLT Header Regex in `silverguard.py` and `label_sms.py`
*   **Current Defect**: Regex assumes headers must end in `-T`, `-S`, `-P`, or `-G`, failing on real-world retail headers like `JK-SMRTBR` or `AM-AIRTEL`.
*   **Required Fix**: Update regex to recognize full 6-character alphanumeric Principal Entity headers with optional hyphenated category suffixes:
```python
# Matches: JK-SMRTBR, AD-HDFCBK, JK-620040-P, VM-DOMINO, etc.
DLT_REGEX_COMPREHENSIVE = re.compile(r'^[A-Z]{2}-([A-Z0-9]{4,8})(-[TSPG])?$')
```

### Task 0.3: Update User Labeling Instructions in `README.md` and `label_sms.py`
*   **Current Defect**: Instructions tell annotators: `"label": 1 — Scam (fraud, phishing, spam)`.
*   **Required Fix**: Explicitly separate commercial spam from scams:
    *   `"label": 0` $\rightarrow$ Ham (Personal conversations, bank transactional OTPs, service alerts).
    *   `"label": 1` $\rightarrow$ Promotional Marketing (Discounts, offers, retail ads).
    *   `"label": 2` $\rightarrow$ Commercial Spam (Unsolicited 10-digit telemarketing).
    *   `"label": 3` $\rightarrow$ Malicious Scam / Phishing (Fraud, extortion, digital arrest, fake KYC links).

---

## 3. Phase 1: Dataset Harmonization & The SilverGuard-Indic-40K Corpus (Weeks 2–3)

### Task 1.1: Ingest Public Academic & Industry Datasets
1.  **Download Indian Scam SMS 30k Dataset** (Atherion Systems, Kaggle CC0, `dataset_30k_pure.jsonl`).
2.  **Download Hinglish Scam Text Dataset** (Bolewara, Hugging Face, `bolewara/hinglish-scam-text-dataset`).
3.  **Download Mendeley SMS Phishing Dataset** (Mishra & Soni, 2020, CC BY 4.0, 5,971 messages).
4.  **Relabel UCI SMS Spam Collection**: Map UCI `ham` $\rightarrow 0$ and UCI `spam` $\rightarrow 1$ (Promotional Commercial Ad), eliminating the false assignment of UCI ads to the scam class.

### Task 1.2: PII Redaction & Normalization Pipeline
*   Execute automated regex masking:
    *   `re.sub(r'(\+?91|0)?[6-9]\d{9}', '[PHONE_NUM]', text)`
    *   `re.sub(r'\b\d{9,18}\b|\bX{4,}\d{4}\b', '[ACCT_NUM]', text)`
    *   `re.sub(r'\b\d{4,6}\b(?=.*otp)', '[OTP_CODE]', text)`
*   Tokenize URLs into canonical risk representations (`[URL_SHORTENER]`, `[URL_IP]`, `[URL_APK]`, `[URL_WHITELISTED]`).

### Task 1.3: Data Deduplication & Grouped Splitting
*   Apply MinHash LSH fuzzy deduplication ($J > 0.85$) to eliminate duplicate template sentences.
*   Assign group IDs based on source campaigns and prompt templates.
*   Execute `GroupKFold` splitting (70% Train, 15% Validation, 15% Test) to guarantee that synthetic template variations never leak across splits.

---

## 4. Phase 2: Dual-Engine Model Training & Calibration (Weeks 4–5)

### Task 2.1: Train Tier-1 Primary Engine (Subword FastText + GBDT Metadata Fusion)
1.  **Train Supervised FastText Model**:
    *   Configure character $n$-grams (lengths 3 to 6) with dimension 100, learning rate 0.25, 25 epochs.
    *   Quantize vectors using `fasttext quantize -qnorm -cutoff 50000` to compress model to **~4.2 MB**.
2.  **Train LightGBM Metadata Fusion Classifier**:
    *   Extract 12 structural domain features (DLT header validity, URL count, IP format, APK presence, callback phone numbers, uppercase ratio, currency symbols).
    *   Ensemble FastText output probabilities with GBDT features using Soft-Voting.
    *   Validate execution: **1.2 ms inference latency, < 22 MB RAM, LMKD-immune**.

### Task 2.2: Fine-Tune Tier-2 Deep Reasoner (IndicBERT v2 ALBERT)
1.  Initialize checkpoint `ai4bharat/IndicBERTv2-MLM-only` (13.5M parameters).
2.  Attach a 4-class linear classification head with dropout 0.1.
3.  Train using **Focal Loss** ($\gamma = 2.0$) with `max_length = 64`:
    *   Batch size: 32; Optimizer: AdamW ($3 \times 10^{-5}$); Warmup: 10% steps; Epochs: 4.
4.  Fit a post-hoc **TemperatureScaler** on validation logits to ensure calibrated confidence scores ($\text{ECE} < 0.03$).

---

## 5. Phase 3: ONNX INT8 Optimization & Tokenizer Packaging (Weeks 6–7)

### Task 3.1: Export PyTorch IndicBERT v2 to Dynamic ONNX
*   Wrap model in an inference wrapper outputting calibrated 4-class probabilities.
*   Export with dynamic axes (`batch_size`, `sequence_length`) targeting opset 17.

### Task 3.2: Dynamic INT8 Quantization via ONNX Runtime
*   Execute `quantize_dynamic` targeting `QInt8`.
*   Verify output file size: **$\le 32\text{ MB}$** on disk.
*   Validate ARM Cortex-A53 CPU inference latency: **$\le 35\text{ ms}$** on mid-range devices.

### Task 3.3: Native Tokenizer Packaging
*   Bundle SentencePiece `.spm` vocabulary model with native C++ bindings for mobile loading without Python dependencies.

---

## 6. Phase 4: Production Inference & Mobile Client Integration (Weeks 8–9)

### Task 4.1: Refactor `silverguard.py` into a 3-Stage Cascaded Engine
Implement the 3-Stage Cascaded Architecture in the CLI:
*   **Stage 1 (< 0.2 ms)**: Deterministic contact matching and DLT regex gateway (resolves 75%–82% of messages instantly).
*   **Stage 2 (1.2 ms – 2.0 ms)**: FastText + LightGBM subword classifier (< 22 MB RAM, resolving 95% of remaining messages).
*   **Stage 3 (35 ms – 180 ms)**: Quantized multilingual IndicBERT v2 invoked exclusively on ambiguous messages ($0.35 \le S_{\text{scam}} \le 0.80$).

### Task 4.2: Flutter / Android Application Integration
Update the mobile client in `backup/silver_guard_flutter.zip`:
*   Integrate FastText C++ library via Dart FFI and `onnxruntime_flutter` for the secondary tier.
*   Register an Android `BroadcastReceiver` listening to `android.provider.Telephony.SMS_RECEIVED`.
*   Surface actionable UI cards:
    *   *"✅ Verified Bank Transaction from HDFC Bank"*
    *   *"📢 Retail Promotion from SMART Bazaar (Moved to Offers)"*
    *   *"🚨 HIGH RISK SCAM: Unverified 10-digit number claiming to be CBI. Demands urgent callback under threat of digital arrest."*

### Task 4.3: Automated Over-the-Air (OTA) Model Update Channel
*   Configure the client to query the Hugging Face Model Hub (`tanishqmudaliar/SilverGuard`) periodically over Wi-Fi.
*   Download quantized `.onnx` and FastText delta checkpoints dynamically.

---

## 7. Priority Matrix & Impact Assessment

| Priority Level | Initiative | Direct Impact on Project | Implementation Effort |
| :--- | :--- | :--- | :--- |
| **P0 (Immediate)** | Fix Indic vowel stripping in `silverguard.py` | Restores Hindi & Marathi readability | Low (< 2 hours) |
| **P0 (Immediate)** | Patch DLT header regex in `silverguard.py` | Eliminates false alarms on 6-char retail headers | Low (< 2 hours) |
| **P0 (Immediate)** | Split binary labels into 4-class taxonomy | Resolves root cause of spam-to-scam conflation | Medium (1–2 days) |
| **P1 (High)** | Ingest Atherion 30K, Bolewara & Mendeley data | Constructs SilverGuard-Indic-40K benchmark | Medium (2–3 days) |
| **P1 (High)** | Train Tier-1 FastText + GBDT Engine | Delivers sub-2ms, LMKD-safe edge classifier | High (3–4 days) |
| **P1 (High)** | GroupKFold template splitting | Eliminates data leakage and false 99% metrics | Medium (1 day) |
| **P2 (Medium)** | Fine-Tune & INT8 Quantize IndicBERT v2 | State-of-the-art secondary deep reasoner (~31 MB)| High (4–5 days) |
| **P2 (Medium)** | Implement 3-Stage Cascaded Pipeline | Cuts mobile battery drain by 85% | High (4–5 days) |
| **P3 (Future)** | Flutter onnxruntime background service | Fully automated on-device Android SMS defense | High (1–2 weeks) |
