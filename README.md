# 🛡️ Silver Guard — Indian Multilingual SMS Scam & Threat Classifier

An open-source, edge-optimized cascaded AI inference engine and training pipeline for classifying Indian SMS messages into **Safe Ham**, **Promotional Offers**, **Commercial Spam (UCC)**, and **Malicious Scams / Phishing**.

Designed specifically for Indian telecommunications (TRAI DLT sender headers, regional Indic languages, Hinglish, UPI fraud, digital arrest extortion, and fake KYC lures) with zero cloud dependency.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/Runtime-ONNX-005CED?logo=onnx&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-FastText%20Domain%20Fusion-FF6F00?logo=pytorch&logoColor=white)
![Model Size](https://img.shields.io/badge/Model%20Size-15.4%20MB-success)
![Taxonomy](https://img.shields.io/badge/Taxonomy-4--Class%20v2.0-brightgreen)
![Indic NLP](https://img.shields.io/badge/Multilingual-Hindi%20%7C%20Marathi%20%7C%20Tamil%20%7C%20Telugu-blueviolet)
![Tests](https://img.shields.io/badge/Tests-28%20Passed-success)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📑 Table of Contents

- [Overview & Problem Solved](#-overview--problem-solved)
- [Quick Diagnostic Matrix: Assumptions vs Reality](#-quick-diagnostic-matrix-assumptions-vs-reality)
- [4-Class Classification Taxonomy (v2.0)](#-4-class-classification-taxonomy-v20)
- [3-Stage Cascaded Architecture](#-3-stage-cascaded-architecture)
  - [Stage 1: Deterministic Telecom Gateway (< 0.2 ms)](#stage-1-deterministic-telecom-gateway--02-ms)
  - [Stage 2: Multilingual Neural Inference (~1.2 ms)](#stage-2-multilingual-neural-inference-12-ms)
  - [Stage 3: Evidentiary Risk & Commercial Synthesis (< 1.0 ms)](#stage-3-evidentiary-risk--commercial-synthesis--10-ms)
- [Neural Architectures: FastText Domain Fusion vs MobileBERT](#-neural-architectures-fasttext-domain-fusion-vs-mobilebert)
- [Multilingual Indic Script Support](#-multilingual-indic-script-support)
- [Repository Structure](#-repository-structure)
- [Quickstart: Running Inference](#-quickstart-running-inference)
  - [Interactive REPL Mode](#interactive-repl-mode)
  - [Single SMS Evaluation (CLI & JSON)](#single-sms-evaluation-cli--json)
  - [Contact Book Whitelisting](#contact-book-whitelisting)
- [Multi-Source Training Dataset (24,514 Unique Samples)](#-multi-source-training-dataset-24514-unique-samples)
- [Training Data Pipeline](#-training-data-pipeline)
  - [Step 1 — Auto-Label Messages (`label_sms.py`)](#step-1--auto-label-messages-label_smspy)
  - [Step 2 — Merge Labels (`merge_labels.py`)](#step-2--merge-labels-merge_labelspy)
  - [Step 3 — Dataset Sanitization & Deduplication (`dataset_pipeline.py`)](#step-3--dataset-sanitization--deduplication-dataset_pipelinepy)
  - [Step 4 — Build Unified Dataset (`build_unified_dataset.py`)](#step-4--build-unified-dataset-build_unified_datasetpy)
  - [Step 5 — Retrain Neural Model & Export ONNX (`train_model.py`)](#step-5--retrain-neural-model--export-onnx-train_modelpy)
- [Empirical Benchmark Results (5,153 Real Inbox SMS)](#-empirical-benchmark-results-5153-real-inbox-sms)
- [Automated Testing Suite](#-automated-testing-suite)
- [Master Documentation Directory (`docs/`)](#-master-documentation-directory-docs)
- [Protocol for Autonomous AI Agents & Developers](#-protocol-for-autonomous-ai-agents--developers)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview & Problem Solved

Traditional SMS scam classifiers rely on binary classification (`Ham` vs `Scam`). In the Indian telecommunications ecosystem, this naive binary setup causes severe real-world failure modes:

1. **Promotional Misclassification False Positives**: Legitimate retail sales, discount vouchers, and brand promotional SMS (from registered DLT entities like `JK-SMRTBR`, `JM-AJIOIN`, `CP-MAXFSN`) contain words like _"50% OFF"_, _"flat discount"_, _"coupon code"_, and _"free delivery"_. Binary models mistake these marketing words for financial fraud, alarming the user with false positive scam warnings.
2. **Multilingual & Indic Script Blindness**: Previous ASCII/Latin-centric tokenizers stripped Indic dependent vowel signs (matras: `ा`, `ि`, `ी`), viramas (`्`), and nuktas during accent-stripping passes, collapsing Devanagari (Hindi, Marathi), Dravidian (Tamil, Telugu), and Bengali words into unreadable character sequences or `[UNK]` tokens.
3. **Edge Resource Constraints on Budget Smartphones**: Low-end Android smartphones commonly used across India (2GB–3GB RAM, Android Go, ARM Cortex-A53/A55) have strict background memory limits ($< 50\text{ MB}$ RSS). Heavy transformer models cause background process termination by the Android Low Memory Killer Daemon (`lmkd`).

Silver Guard solves these challenges using a **3-Stage Cascaded Engine** combined with an **Indic-safe canonical normalization pipeline**, a **FastText Subword + 12 Domain Features Neural Fusion architecture (50 KB ONNX binary)**, and a **4-Class Telecom Taxonomy**.

---

## 🔍 Quick Diagnostic Matrix: Assumptions vs Reality

The original implementation operated on several critical assumptions that broke down when deployed against real-world Indian SMS traffic:

```text
+------------------------------------+---------------------------------------+----------------------------------------+
| Dimension                          | Original Project Assumption           | Verified Empirical Reality             |
+------------------------------------+---------------------------------------+----------------------------------------+
| 1. Classification Strategy         | Binary classification:                | Collapses commercial advertisements    |
|                                    | 0 = Ham, 1 = Scam                     | into scams; 100% of false positives   |
|                                    |                                       | in Iteration 2 are legitimate promos.  |
+------------------------------------+---------------------------------------+----------------------------------------+
| 2. Language Support                | English MobileBERT uncased covers     | Erases Indic vowels (matras); up to   |
|                                    | all necessary messaging vocabulary.   | 66.7% out-of-vocabulary rate in Telugu;|
|                                    |                                       | Hinglish shredded into noisy subwords. |
+------------------------------------+---------------------------------------+----------------------------------------+
| 3. Telecom DLT Headers             | Legitimate headers strictly end with  | Live retail headers (e.g. JK-SMRTBR,  |
|                                    | -T, -S, -P, or -G.                    | AM-AIRTEL) omit suffixes, failing     |
|                                    |                                       | regex and triggering scam alarms.      |
+------------------------------------+---------------------------------------+----------------------------------------+
| 4. Dataset Quality & Balance       | Public Kaggle spam datasets and       | training_data.json had 0 scams (100%   |
|                                    | synthetic templates reflect scams.    | ham); synthetic templates leaked into  |
|                                    |                                       | test splits with deterministic headers.|
+------------------------------------+---------------------------------------+----------------------------------------+
| 5. Evaluation & Accuracy           | Test accuracy is ~99% with            | Random train/test split leaked repeated|
|                                    | real-world FPR < 1%.                  | templates; real personal SMS showed a  |
|                                    |                                       | 96.8% False Positive Rate in Iteration 1.|
+------------------------------------+---------------------------------------+----------------------------------------+
| 6. Edge Deployment (Low-End Phones)| 105 MB FP32 ONNX model is             | Exceeds GitHub limits, crashes 2GB RAM |
|                                    | ready for mobile on-device inference. | Android Go devices via LMKD kill, and  |
|                                    |                                       | causes multi-second cold starts.       |
+------------------------------------+---------------------------------------+----------------------------------------+
```

---

## 🏷️ 4-Class Classification Taxonomy (v2.0)

Silver Guard decouples commercial intent from malicious intent across four distinct classes:

| Class ID | Semantic Class            | Typical Senders                                                                                               | Key Characteristics & Behavior                                                                                                                            |
| :------: | :------------------------ | :------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   `0`    | **Safe Ham**              | Personal contacts, banks (`AD-HDFCBK-T`), government (`JX-ISATHI-G`), service alerts (`MH-AIRTEL-S`)          | OTPs, balance debits, family chats, courier delivery updates. Threat score: $\approx 0.0$.                                                                |
|   `1`    | **Promotional**           | Registered TRAI DLT senders with `-P` suffix or registered retail brands (`JK-SMRTBR`, `JM-AJIOIN`, `DOMINO`) | Legitimate e-commerce sales, discount codes, retail offers. High commerciality, zero security threat.                                                     |
|   `2`    | **Commercial Spam (UCC)** | Unregistered 10-digit mobile numbers (`+919876543210`)                                                        | Unsolicited cold marketing (real estate, pre-approved personal loans, insurance leads). Do-Not-Disturb violation, but **no** credential theft or malware. |
|   `3`    | **Scam / Phishing**       | Unknown phone numbers, spoofed display headers                                                                | Digital arrest extortion, law enforcement intimidation (CBI/Customs), fake KYC deactivation, lottery fraud, malicious APK downloads.                      |

---

## ⚡ 3-Stage Cascaded Architecture

```
                        Incoming SMS: (Header, Body)
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Deterministic Telecom Gateway (< 0.2 ms, 0 MB ML RAM)         │
├────────────────────────────────────────────────────────────────────────┤
│ • Matches Contacts Whitelist?                 ──► SAFE (HAM)           │
│ • Verified Bank/Govt/Service DLT (-T/-S/-G)   ──► SAFE (HAM)           │
│ • Registered Retail Promo Header (-P, Brand)  ──► PROMOTIONAL (OFFER)  │
│ • Telecom Operator Shortcodes (1909, 56767)   ──► SAFE (HAM)           │
│ >> Resolves 97.6% of messages instantly without neural weights! <<     │
└───────────────────────────────────┬────────────────────────────────────┘
                       (Non-DLT / Unverified Sender)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Multilingual Neural Inference (~1.2 ms, < 15 MB RAM)          │
├────────────────────────────────────────────────────────────────────────┤
│ • Primary Engine: FastText Subwords (3–5 n-grams) + 12 Domain Features │
│ • Secondary Fallback: MobileBERT (24M parameters, ONNX)                │
│ • NFC Canonical Normalization (Preserves Brahmic Matras & Emojis)      │
│ • 4-Class Softmax Logits: [P(Ham), P(Promo), P(Spam_UCC), P(Scam)]     │
└───────────────────────────────────┬────────────────────────────────────┘
                     (Base Threat & Commercial Probs)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Evidentiary Risk & Commercial Synthesis (< 1.0 ms)            │
├────────────────────────────────────────────────────────────────────────┤
│ • Link Shorteners (bit.ly), Direct IPs, APK Downloads Extraction       │
│ • Intimidation & Extortion Regex Triggers (CBI, Police, Digital Arrest)│
│ • Commercial vs Threat Decoupling (DLT Promo vs Unsolicited UCC Spam)  │
│ • Dual Output: Final Verdict, Threat Score (0–1), Commercial Score(0–1)│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Neural Architecture: FastText Domain Fusion

Silver Guard deploys the **FastText Subword Domain Fusion** architecture (`FastTextDomainFusion`), replacing the legacy 105 MB MobileBERT baseline:

| Architectural Dimension            | Production Model: FastText Domain Fusion (`model/silverguard_fasttext_fusion.onnx`)         | Legacy Baseline: MobileBERT (Deprecated)                                 |
| :--------------------------------- | :------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------- |
| **Model Type**                     | Dual-Branch Hybrid: FastText Subword Embeddings + Tabular MLP                               | Monolithic 24-Layer Transformer (`google/mobilebert-uncased`)            |
| **Model Disk Size**                | **56 KB graph + 15.3 MB weights table**                                                     | 105.0 MB (FP32) / 26.2 MB (INT8)                                         |
| **Inference Latency (CPU)**        | **~0.8–1.2 ms / SMS**                                                                       | 18–25 ms / SMS                                                           |
| **Peak RAM Footprint**             | **< 15 MB** (Immune to Android Go LMKD process killing)                                     | ~120 MB (Frequently killed on 2GB RAM phones)                            |
| **Text Representation**            | FastText character subword n-grams (lengths 3–5) via 32-bit CRC hashing into 60,000 buckets | WordPiece tokenization (30,522 vocab tokens)                             |
| **Phonetic & Hinglish Resilience** | High: Character n-grams capture misspellings (`discnt`, `k-y-c`, `p0lice`) natively         | Low: Out-of-vocabulary terms collapse into multiple noisy `[UNK]` tokens |
| **Domain Feature Injection**       | Direct: 12 handcrafted telecom features fused into hidden dense layer                       | Indirect: String concatenation `[CLS] HEADER [SEP] MESSAGE [SEP]`        |
| **Output**                         | 4-Class Softmax Vector (`[Ham, Promo, Spam, Scam]`)                                         | Binary Logit / Threat Probability scalar                                 |

### What is "FastText Domain Fusion"?

"FastText Domain Fusion" is a **dual-branch multimodal neural architecture**:

1. **Branch 1 (Subword Linguistic Branch)**: Uses FastText-style character n-grams (Bojanowski et al., 2017) embedded via PyTorch `nn.EmbeddingBag(vocab_size=60000, embed_dim=64, mode="mean")`. Character n-grams (lengths 3 to 5) are hashed deterministically with 32-bit CRC. This provides language-agnostic phonetic invariance across Hindi, Hinglish, Marathi, and English.
2. **Branch 2 (Domain Feature Tabular Branch)**: 12 handcrafted telecom and structural indicators (URL counts, shorteners, direct IPs, APK downloads, callback phones, currency counts, urgency triggers, promotional keywords, uppercase ratios, digit ratios, length, Indic script codes) pass through a 2-layer MLP (`12 -> 32 -> 32`, ReLU).
3. **Early Fusion & Classification Head**: The 64-dim text embedding and 32-dim tabular embedding are concatenated (`torch.cat([text_emb, tab_emb], dim=1) -> 96 dim`) and fed into a classification head with dropout (`96 -> 4`) trained with class-balanced cross-entropy loss.

---

## 🌐 Multilingual Indic Script Support

Indian SMS messages frequently contain regional languages and code-switching:

- **Devanagari**: Hindi (`हिंदी`), Marathi (`मराठी`)
- **Dravidian**: Tamil (`தமிழ்`), Telugu (`తెలుగు`), Kannada (`ಕನ್ನಡ`), Malayalam (`മലയാളം`)
- **Eastern Indic**: Bengali (`বাংলা`), Gujarati (`ગુજરાતી`)
- **Code-Switched Hinglish**: Hindi written in Latin script (_"Aapka account block ho gaya hai"_)

Silver Guard's [`dataset_pipeline.py`](file:///c:/Users/tanis/Documents/VSC%20Projects/Python%20Projects/Silver%20Guard%20AI%20Model%20Training/dataset_pipeline.py) enforces **Indic-Safe Canonical NFC Normalization**:

1. Strictly preserves Brahmic vowel signs (matras: `ा`, `ि`, `ी`, `ु`, `ू`), viramas (`्`), and nuktas.
2. Strips only Western Latin combining accents (`Mn`), preventing character corruption.
3. Whitespace-isolates emojis and Indic punctuation (danda `।` and double danda `॥`) to prevent token collapse.

---

## 📁 Repository Structure

```
.
├── silverguard.py                  # Production cascaded inference engine (CLI, JSON & REPL)
├── train_model.py                  # PyTorch FastText Domain Fusion training & ONNX export script
├── train_gbdt.py                   # LightGBM GBDT multi-class training script
├── build_unified_dataset.py        # 5-source dataset aggregator, cleaner & balancer
├── dataset_pipeline.py             # PII sanitization, URL tokenization & shingle deduplication
├── label_sms.py                    # 4-class auto-labeling script with DLT parser
├── merge_labels.py                 # Review merger for manual audit samples
├── main.ipynb                      # Google Colab / Jupyter training notebook
│
├── model/                          # Trained model artifacts (Tracked directly in Git & Hugging Face)
│   ├── silverguard_fasttext_fusion.onnx       # Primary FastText ONNX compute graph (56 KB)
│   ├── silverguard_fasttext_fusion.onnx.data  # Subword embedding weights table (15.36 MB)
│   ├── silverguard_gbdt.joblib                # LightGBM GBDT multi-class model (1.50 MB)
│   ├── char_vectorizer.joblib                 # Subword character n-gram TF-IDF vectorizer (112 KB)
│   ├── gbdt_metadata.json                     # GBDT training metadata & evaluation metrics
│   ├── model_config.json                      # ONNX model metadata, taxonomy, and metrics
│   ├── HF_README.md                           # Hugging Face Model Card README
│   └── MODEL_HOSTING.md                       # Model distribution & Hugging Face CLI guide
│
├── data/
│   ├── silverguard_indic_unified.json    # Gold-standard 24,514 unique 4-class dataset
│   ├── rawdata.json                      # 5,226 raw personal SMS messages
│   ├── contacts.csv                      # Whitelisted address book contacts
│   └── external/                         # Ingested datasets (Atherion 30K, ScamShield, JuniorAlive, UCI)
│
├── tests/                                # Comprehensive test suite (32 tests passing)
│   ├── test_dlt_parser.py
│   ├── test_indic_normalization.py
│   ├── test_dataset_pipeline.py
│   ├── test_labeling_pipeline.py
│   └── test_classification_engine.py
│
└── docs/                                 # Specialized technical and academic research
    ├── repository_audit.md               # Forensic code, model, and dataset audit
    ├── classification_taxonomy.md        # 4-Class taxonomy specification
    ├── multilingual_nlp_research.md      # Indic NLP, tokenizers, and script normalization
    ├── datasets_research.md              # Dataset survey and curation methodology
    ├── model_architecture_research.md    # Edge hardware profiling and model trade-offs
    ├── evaluation_strategy.md            # Cost matrix and template leakage prevention
    ├── research_references.md            # Peer-reviewed academic bibliography
    ├── model_comparison_benchmark.md     # 5,153 message benchmark & forensic triage
    └── implementation_roadmap.md         # Phased engineering blueprint
```

---

## 🚀 Quickstart: Running Inference

### Interactive REPL Mode

```powershell
python silverguard.py --contacts data/contacts.csv
```

Example session:

```text
Loading Silver Guard (model/silverguard_fasttext_fusion.onnx)...
Loaded 44 contacts for whitelisting.
✅ Ready.

  Input format :  HEADER [SEP] message body
  Example DLT  :  JK-SMRTBR [SEP] 50% OFF on all groceries!
  Example Scam :  +919876543210 [SEP] Your Aadhaar is blocked by CBI. Call now!

>>> JK-SMRTBR [SEP] SMART Bazaar Full Paisa Vasool Sale! Get 50% OFF on groceries today.
┌──────────────────────────────────────────────────────────────────┐
│  Verdict         : 📢 PROMOTIONAL (MARKETING / OFFER)            │
│  Threat Score    : 0.0150  [░░░░░░░░░░░░░░░░░░░░]                │
│  Commerciality   : 0.9800  [████████████████████]                │
├──────────────────────────────────────────────────────────────────┤
│  Sender Identity : 📢  SMRTBR retail marketing 📢                 │
│  Decision Stage  : Stage 1 (DLT Promo Gateway)                   │
├──────────────────────────────────────────────────────────────────┤
│  Explanation     : Legitimate retail marketing offer from regist │
│                    ered entity 'SMRTBR'. Not a security scam.    │
└──────────────────────────────────────────────────────────────────┘
```

### Single SMS Evaluation (CLI & JSON)

```powershell
# Personal chat from unknown mobile number (Evaluated in Stage 2 as Safe Ham)
python silverguard.py --header "+919876543210" --message "Can we meet at Starbucks tomorrow at 5 PM for coffee?" --json

# Registered DLT Retail Promo
python silverguard.py --header "JK-SMRTBR" --message "Flat 50% discount on groceries today only!"

# Unsolicited telemarketing (Classified as Commercial Spam UCC, NOT Scam)
python silverguard.py --header "+919876543210" --message "Exclusive 2BHK Luxury Flats in Noida starting 45 Lakhs. Zero down payment."

# High-risk Cyber Extortion
python silverguard.py --header "+919876543210" --message "CBI NOTICE: Arrest warrant issued. Your Aadhaar is blocked. Call now." --json
```

---

## 📊 Multi-Source Training Dataset (24,514 Unique Samples)

Silver Guard trains on [`data/silverguard_indic_unified.json`](file:///c:/Users/tanis/Documents/VSC%20Projects/Python%20Projects/Silver%20Guard%20AI%20Model%20Training/data/silverguard_indic_unified.json), compiled from 5 datasets:

| Dataset Source                                 | Original Size | Extracted Subset | Ingested Classes                   | Purpose in Silver Guard                                      |
| :--------------------------------------------- | :-----------: | :--------------: | :--------------------------------- | :----------------------------------------------------------- |
| **Atherion Indian Scam SMS 30K** (Kaggle)      |    30,000     |      19,500      | Ham (`0`), Spam (`2`), Scam (`3`)  | Real Indian smishing, rental scams, fake KYC alerts          |
| **ScamShield Multilingual Smishing** (Kaggle)  |    85,000+    |      4,500       | Ham (`0`), Scam (`3`)              | Social engineering lures, urgent bank phishing               |
| **JuniorAlive India SMS Spam** (Kaggle)        |     2,267     |      2,267       | Ham (`0`), Spam (`2`), Scam (`3`)  | Real Indian SMS spam, telecom service notifications          |
| **UCI SMS Spam Collection** (Kaggle)           |     5,572     |      5,572       | Ham (`0`), Spam (`2`), Scam (`3`)  | Natural interpersonal English conversational chats           |
| **Personal Indian SMS Inbox** (`rawdata.json`) |     5,226     |      5,154       | Ham (`0`), Promo (`1`), Spam (`2`) | Active Indian inbox ground truth (DLT headers, personal SMS) |

After PII sanitization, URL threat classification, and inverted-index shingle deduplication, the dataset contains **24,514 unique samples**:

- **Class 0 (Safe Ham)**: 17,013 (69.4%)
- **Class 1 (Promotional)**: 728 (3.0%)
- **Class 2 (Commercial Spam UCC)**: 1,884 (7.7%)
- **Class 3 (Scam / Phishing)**: 4,889 (19.9%)

---

## 🛠️ Training Data Pipeline

```powershell
# 1. Compile, sanitize, and deduplicate all 5 datasets
python build_unified_dataset.py

# 2. Retrain FastText Domain Fusion neural model and export to ONNX
python train_model.py

# 3. Run full automated test suite
python -m unittest discover -s tests -v
```

---

## 📈 Empirical Benchmark Results (5,153 Real Inbox SMS)

Evaluated across **5,153 real-world personal SMS messages** in [`data/second_model_output.csv`](file:///c:/Users/tanis/Documents/VSC%20Projects/Python%20Projects/Silver%20Guard%20AI%20Model%20Training/data/second_model_output.csv):

```text
+------------------------------------+--------------------+--------------------+-------------------------+
| Metric / Characteristic            | Iteration 1 (Base) | Iteration 2 (Tuned)| Silver Guard (FastText) |
+------------------------------------+--------------------+--------------------+-------------------------+
| Total Messages Evaluated           | 5,153              | 5,153              | 5,153                   |
| Classified as SAFE (Ham)           | 164  ( 3.2%)       | 5,128 (99.5%)      | 4,732 (91.8%)           |
| Classified as PROMOTIONAL (Offer)  | N/A  (No Class)    | N/A   (No Class)   | 419   ( 8.1%)           |
| Classified as SCAM (Malicious)     | 4,989 (96.8%)      | 25    ( 0.49%)     | 0     ( 0.0%)           |
| Classified as SPAM (UCC) / Caution | N/A  (No Class)    | N/A   (No Class)   | 2     ( 0.04%)          |
| False Positive Rate on Inbox       | 96.8% (Catastrophic| 100% of all alerts | 0.0%  (Zero False       |
|                                    | false alarms)      | were false alarms  | Alarms in real inbox)   |
| Iteration 2 FPs Eliminated         | 0 / 25             | 0 / 25             | 25 / 25 (100% Resolved) |
| Total Benchmark Execution Time     | ~115.0 seconds     | ~115.0 seconds     | 2.008 seconds           |
| Average Latency per SMS            | 22.3 ms            | 22.3 ms            | 0.390 ms (57.3x faster!)|
| Model Binary Size (ONNX)           | 105.0 MB           | 105.0 MB           | 15.4 MB (56KB + 15.36MB)|
| Peak Inference RAM                 | ~120 MB            | ~120 MB            | < 15 MB (LMKD Immune)   |
| Messages Handled in Stage 1 (<0.2ms)| 0 (0.0%)          | 0 (0.0%)           | 5,031 (97.6%)           |
| Indic Matras (Brahmic Vowels)      | Stripped (NFD Mn)  | Stripped (NFD Mn)  | 100% Preserved (NFC)    |
| Emoji Tokenization                 | [UNK] Collapse     | [UNK] Collapse     | Isolated Clean Tokens   |
+------------------------------------+--------------------+--------------------+-------------------------+
```

### 🎯 Held-Out Test Evaluation & 4x4 Confusion Matrices (4,903 Unseen Messages)

Both models were evaluated on a stratified 20% test split (**4,903 messages**) held out from the 24,514-sample unified dataset (`silverguard_indic_unified.json`):

#### Model 1: Production On-Device FastText Domain Fusion (`silverguard_fasttext_fusion.onnx`)
* **Target Environment**: Android / iOS / Flutter edge deployment (15.4 MB total, < 22 MB RAM, ~1.2 ms latency)
* **Overall Accuracy**: **94.94%** | **Macro F1**: **89.94%** | **Weighted F1**: **94.96%**

```text
              precision    recall  f1-score   support

         Ham     0.9793    0.9736    0.9764      3403
       Promo     0.8533    0.8828    0.8678       145
    Spam_UCC     0.8399    0.8488    0.8443       377
        Scam     0.9039    0.9141    0.9090       978

    accuracy                         0.9494      4903
   macro avg     0.8941    0.9048    0.8994      4903
weighted avg     0.9498    0.9494    0.9496      4903
```

**4x4 Confusion Matrix (ONNX Edge Model):**
```text
               Pred_Ham  Pred_Promo  Pred_Spam  Pred_Scam
Actual_Ham   :      3313          17         21         52
Actual_Promo :        11         128          2          4
Actual_Spam  :        16           2        320         39
Actual_Scam  :        43           3         38        894
```

---

#### Model 2: Server / Microservice LightGBM GBDT (`silverguard_gbdt.joblib`)
* **Target Environment**: Python backend / server microservice (1.50 MB model, 0.11 MB vectorizer, < 18 MB RAM)
* **Overall Accuracy**: **97.84%** | **Macro F1**: **94.99%** | **Weighted F1**: **97.83%**

```text
              precision    recall  f1-score   support

         Ham     0.9977    0.9982    0.9979      3403
       Promo     0.9929    0.9655    0.9790       145
    Spam_UCC     0.8874    0.8568    0.8718       377
        Scam     0.9436    0.9581    0.9508       978

    accuracy                         0.9784      4903
   macro avg     0.9554    0.9446    0.9499      4903
weighted avg     0.9782    0.9784    0.9783      4903
```

**4x4 Confusion Matrix (GBDT Server Model):**
```text
               Pred_Ham  Pred_Promo  Pred_Spam  Pred_Scam
Actual_Ham   :      3397           1          2          3
Actual_Promo :         5         140          0          0
Actual_Spam  :         1           0        323         53
Actual_Scam  :         2           0         39        937
```
*(Out of 3,403 actual Ham messages in the test set, **3,397 were correctly classified** by GBDT with only 3 false positives into Scam).*

---

## 🧪 Automated Testing Suite

Silver Guard includes 32 unit and integration tests executing in $< 1.0\text{ s}$:

```powershell
python -m unittest discover -s tests -v
```

```text
test_dlt_commercial_brands (test_dlt_parser.TestDLTParser) ... ok
test_dlt_shortcodes (test_dlt_parser.TestDLTParser) ... ok
test_dlt_suffixed_headers (test_dlt_parser.TestDLTParser) ... ok
test_indic_devanagari_matra_preservation (test_indic_normalization.TestIndicNormalization) ... ok
test_indic_emoji_token_isolation (test_indic_normalization.TestIndicNormalization) ... ok
test_pii_redaction (test_dataset_pipeline.TestDatasetPipeline) ... ok
test_url_tokenization (test_dataset_pipeline.TestDatasetPipeline) ... ok
test_fasttext_contact_whitelisting (test_classification_engine.TestFastTextClassificationEngine) ... ok
test_fasttext_dlt_promotional (test_classification_engine.TestFastTextClassificationEngine) ... ok
test_fasttext_high_risk_scam_extortion (test_classification_engine.TestFastTextClassificationEngine) ... ok
test_fasttext_malicious_apk (test_classification_engine.TestFastTextClassificationEngine) ... ok
test_fasttext_unsolicited_commercial_spam (test_classification_engine.TestFastTextClassificationEngine) ... ok
----------------------------------------------------------------------
Ran 32 tests in 0.833s

OK
```

---

## 📚 Master Documentation Directory (`docs/`)

Specialized academic and architectural research documents in `docs/`:

| Document                                                                     | Primary Scope                                    | Key Takeaway                                                        |
| :--------------------------------------------------------------------------- | :----------------------------------------------- | :------------------------------------------------------------------ |
| [`docs/repository_audit.md`](docs/repository_audit.md)                       | Forensic audit of old code, models, and datasets | Root cause isolation of the 96.8% FPR and 25 false positives        |
| [`docs/classification_taxonomy.md`](docs/classification_taxonomy.md)         | 4-Class semantic taxonomy specification          | Formal boundaries separating Promo from UCC Spam and Scam           |
| [`docs/multilingual_nlp_research.md`](docs/multilingual_nlp_research.md)     | Indic NLP, Brahmic scripts, and Unicode NFC      | Mathematical and linguistic justification for NFC normalization     |
| [`docs/datasets_research.md`](docs/datasets_research.md)                     | Survey of external datasets & curation strategy  | Specifications for Atherion 30K, ScamShield, and SilverGuard corpus |
| [`docs/model_architecture_research.md`](docs/model_architecture_research.md) | Mobile hardware profiling & edge benchmarks      | Detailed CPU/RAM/Battery profiling on ARM Cortex-A53                |
| [`docs/evaluation_strategy.md`](docs/evaluation_strategy.md)                 | Asymmetric cost matrix & template leakage        | Evaluation protocols preventing synthetic data leakage              |
| [`docs/research_references.md`](docs/research_references.md)                 | Peer-reviewed academic bibliography              | Citations for FastText, MuRIL, IndicBERT, and TRAI TCCCPR           |
| [`docs/model_comparison_benchmark.md`](docs/model_comparison_benchmark.md)   | Empirical 5,153 message benchmark report         | Full item-by-item triage of the 25 legacy false positives           |
| [`docs/implementation_roadmap.md`](docs/implementation_roadmap.md)           | Actionable 5-phase engineering roadmap           | Milestone schedule for enterprise mobile deployment                 |

---

## 🛡️ Protocol for Autonomous AI Agents & Developers

When working on this repository, all engineers and autonomous agents must strictly adhere to these 6 engineering directives:

1. **Do Not Modify Code Speculatively**: Never adjust model weights, hyperparameters, or preprocessing regexes without reviewing [`docs/repository_audit.md`](docs/repository_audit.md) and [`docs/classification_taxonomy.md`](docs/classification_taxonomy.md).
2. **Preserve Indic Diacritics**: Never apply `unicodedata.normalize("NFD")` followed by `Mn` category filtering on South Asian text. All text normalization must use NFC format as documented in [`docs/multilingual_nlp_research.md`](docs/multilingual_nlp_research.md).
3. **Adhere to the 4-Class Taxonomy**: Cease all binary labeling. All newly collected, synthesized, or transformed datasets must map into the standard classes defined in [`docs/classification_taxonomy.md`](docs/classification_taxonomy.md):
   - `Class 0`: Ham (Personal, Bank Transactional, Carrier Service)
   - `Class 1`: Promotional (DLT-registered commercial retail ads)
   - `Class 2`: Commercial Spam (Unsolicited 10-digit bulk advertising)
   - `Class 3`: Malicious Scam & Phishing (Financial fraud, digital arrest, fake KYC, malicious APKs)
4. **Enforce Template-Aware Splitting**: When training or evaluating on synthetic datasets, always use group-based or shingle-deduplicated splits to guarantee zero data leakage between train and test splits, following [`docs/evaluation_strategy.md`](docs/evaluation_strategy.md).
5. **Respect Mobile Hardware Limits**: On-device primary models must not exceed **25 MB RAM** or **10 MB disk footprint** to guarantee execution on 2GB RAM Android Go devices without being terminated by `lmkd`.
6. **Follow the Phased Roadmap**: Implement features according to the prioritized schedule established in [`docs/implementation_roadmap.md`](docs/implementation_roadmap.md).

---

## 🤝 Contributing

We welcome contributions to Silver Guard!

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/better-indic-heuristics`.
3. Ensure all tests pass: `python -m unittest discover -s tests`.
4. Submit a Pull Request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

_Made with ❤️ by Tanishq Mudaliar_  
_Protecting Indian Citizens from Smishing, Digital Arrest Cyber Extortion, and Telemarketing Spam._
