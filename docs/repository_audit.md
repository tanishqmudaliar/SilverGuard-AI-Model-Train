# Silver Guard: Comprehensive Repository & Pipeline Audit

## 1. Executive Summary of the Audit

This document provides a forensic, empirical audit of the **Silver Guard AI Model Training** repository as of November 2026. The original goal of this repository was to build an automated machine learning system capable of classifying Indian SMS messages into **genuine (ham)** or **malicious (scam)**, with a particular focus on Indian telecom features such as Telecom Regulatory Authority of India (TRAI) Distributed Ledger Technology (DLT) headers, UPI fraud, and cyber extortion.

Empirical evaluation of the current implementation reveals two fatal failure modes:
1. **Pervasive False Positives on Ordinary Commercial Spam & Promotions**: The model classifies legitimate marketing, sales offers, discount coupons, and transactional alerts from reputable brands (such as SMART Bazaar, AJIO, Max Fashion, and Airtel) as high-risk scams (often with threat scores $> 0.95$). In historical runs (`data/first_model_output.csv`), the initial model classified **4,989 out of 5,153 legitimate messages as scams (a 96.8% False Positive Rate)**. While the second iteration (`data/second_model_output.csv`) reduced the false positive count to 25 messages, **100% of those 25 flagged messages are legitimate promotional or transactional communications**.
2. **Total Breakdown on Multilingual and Code-Mixed SMS**: Although 8.73% of raw messages in the personal corpus (`data/rawdata.json`) contain Devanagari script (Hindi and Marathi), and significant portions of Latin-script messages represent romanized Hindi (Hinglish), the system relies on an English-only `google/mobilebert-uncased` checkpoint. The custom tokenizer strips all non-spacing mark Unicode characters (Unicode category `Mn`), which strips out all Indic vowel signs (*matras*) and viramas, rendering Indic text unreadable. Crucial regional languages (such as Telugu and Tamil) suffer from out-of-vocabulary (`[UNK]`) rates between 15% and 67%, causing the model to either hallucinate threat scores or default to random predictions.

The sections below document the precise technical mechanisms, code implementations, dataset pipelines, and architectural assumptions responsible for these failures.

---

## 2. Repository Layout & Component Architecture

The repository contains the following primary files and assets:

```text
Silver Guard AI Model Training/
├── label_sms.py          # Auto-labeling utility using regex heuristics & contact lookup
├── merge_labels.py       # JSON merger combining auto-labeled ham and reviewed data
├── silverguard.py        # Standalone local ONNX inference CLI with custom WordPiece tokenizer
├── main.ipynb            # Google Colab Jupyter Notebook (19 cells) for data gen & training
├── README.md             # Project overview, Colab workflow instructions, and performance claims
├── model/
│   ├── silver_guard.onnx # Exported MobileBERT ONNX model artifact (~105 MB)
│   ├── vocab.txt         # 30,522-token vocabulary from google/mobilebert-uncased
│   ├── model_config.json # Runtime metadata specification
│   └── MODEL_HOSTING.md  # Hugging Face model hub hosting documentation
├── data/
│   ├── rawdata.json      # 5,226 exported personal SMS messages
│   ├── contacts.csv      # Exported phone address book (528 contact rows)
│   ├── training_data.json# Merged personal training dataset (5,154 messages, all label=0)
│   ├── first_model_output.csv   # Inference log of Iteration 1 (96.8% False Positive Rate)
│   └── second_model_output.csv  # Inference log of Iteration 2 (25 false positives)
└── backup/
    ├── silver_guard_backup.zip  # Environment and training state snapshot
    └── silver_guard_flutter.zip # Android/Flutter mobile client integration application
```

---

## 3. Dataset Audit & Data Flow Analysis

### 3.1 Datasets Utilized in Training Pipeline

The training procedure in `main.ipynb` combines four disparate data sources:

| Dataset Source | Stated Size | Effective Language | Class Balance in Source | Labeling Paradigm |
| :--- | :--- | :--- | :--- | :--- |
| **UCI SMS Spam Collection** | 5,574 msgs | English (UK/Singapore) | Ham: ~4,825, Spam: ~747 (~13.4% spam) | Binary: `ham` vs `spam` (commercial promos) |
| **Junioralive India SMS** (Kaggle) | ~1,000–3,000 msgs | English (Indian context) | Varies by split | Binary: `ham` vs `spam` |
| **Synthetic Generator** (`main.ipynb` Cell 6) | 5,190 msgs | English (Indian keywords) | Scam: ~3,150, Ham: ~2,040 | Binary: `0` (Safe) vs `1` (Scam) |
| **Personal Messages** (`training_data.json`) | 5,154 msgs | English, Hinglish, Devanagari | Ham: 5,154 (100%), Scam: 0 (0%) | Binary: `0` (Safe) |

### 3.2 Forensic Tracing of `main.ipynb` Data Processing

A line-by-line inspection of `main.ipynb` reveals the foundational flaws in how data is ingested, labeled, and combined:

#### A. Conflation of Commercial Spam with Malicious Fraud
In Cell 4 of `main.ipynb`:
```python
# UCI Dataset ingestion
df_uci = pd.DataFrame({
    "text":   raw[text_col].astype(str),
    "label":  raw[label_col].apply(lambda x: 1 if str(x).lower().strip() == "spam" else 0),
    "source": "uci"
})
```
In the UCI SMS Spam Collection, the `spam` class consists predominantly of commercial advertising: ringtones, sweepstakes vouchers, club memberships, shopping discounts, and chat lines. By mapping `spam -> 1`, the notebook equates an unsolicited retail advertisement with cyber extortion or credential harvesting.

#### B. Artificial Header Injection & Spurious Correlation Induction
In Cell 8 of `main.ipynb`:
```python
def add_header_to_kaggle_row(row):
    """Add a synthetic DLT header to Kaggle dataset rows."""
    text = str(row["text"])
    if "[SEP]" in text:
        return text
    if row["label"] == 1:
        header = get_scam_header()
    else:
        # Assign legitimate DLT bank/UPI/telecom header
        header = random.choice(ALL_LEGIT_HEADERS)
    return format_with_header(header, text)
```
This function introduces a catastrophic spurious correlation:
1. Every single message from UCI that was labeled `spam` (which includes legitimate commercial promotions) is forcefully prefixed with a **scam header** (a 10-digit mobile number or spoofed alphanumeric string like `+919876543210` or `VM-URGENT`).
2. Every single message labeled `ham` is prefixed with a **legitimate DLT header** (`JD-SBINOT`, `AD-HDFCBK`, etc.).
3. Consequently, the transformer model learns two conflicting and corrupted rules:
   - **Rule A (Structural shortcut)**: If the sender ID is not formatted like a known legitimate DLT header, mark it as scam.
   - **Rule B (Lexical shortcut)**: Any message body containing promotional tokens—such as `"OFF"`, `"SALE"`, `"FREE"`, `"DISCOUNT"`, `"WIN"`, `"CASH"`, `"APPLY NOW"`—is intrinsically associated with `label = 1` (Scam).

#### C. Extreme Class Imbalance & Data Leakage in Personal Training Data
In `data/training_data.json`:
- Total messages: **5,154**
- Ham count (`label: 0`): **5,154 (100.0%)**
- Scam count (`label: 1`): **0 (0.0%)**

The user labeled zero personal messages as scams. When personal data is injected in Cell 8, it introduces 5,154 purely negative examples without a single positive counterpart from the real distribution.

Furthermore, synthetic scam generation in Cell 6 generates 30 to 40 noisy variants from identical string templates:
```python
# Template replication
NUM_SCAM_PER_TEMPLATE = 30   # 105 templates x 30 = ~3,150
NUM_HAM_PER_TEMPLATE = 40    # 51 templates x 40 = ~2,040
```
Because the train/validation/test split in Cell 8 uses a naive random `train_test_split(test_size=0.2)` on the flattened rows rather than a `GroupKFold` split over template IDs, **virtually identical template sentences appear in both the training set and the test set**. This yields an artificially inflated test accuracy of 99%, creating the illusion of generalization while the model fails completely on unseen out-of-distribution real-world SMS.

---

## 4. Root Causes of Promotional/Spam Misclassification

Empirical evidence from `data/second_model_output.csv` documents 25 persistent false positives. Analysis of these 25 failures exposes the four interlocking root causes:

### Root Cause 1: Lack of a Semantic "Promotional" Class
The system forces a binary categorization:
$$\hat{y} \in \{0 = \text{Safe (Ham)}, 1 = \text{Malicious (Scam)}\}$$
A legitimate marketing message from a supermarket (e.g. `JK-SMRTBR`: *"Offers You can't Miss @SMART Bazaar Onion Rs 22/Kg Alphonso Mango Rs 199/Kg BOGO Kwality Walls Ice cream"*) is inherently **unsolicited advertising (promotional spam)**, but it contains zero malicious intent, phishing links, or credential theft. Because the model has no category for "Promotional" or "Spam", and its training set paired promotional vocabulary with `label = 1`, the model outputs a scam threat score of `0.9088`.

### Root Cause 2: Flawed DLT Header Parsing and Validation Regex
In `silverguard.py`:
```python
# Legitimate DLT format check
if re.match(r"^[A-Z]{2}-[A-Z0-9]{6,8}$", h):
    suffix_map = {
        "G": ("government entity ✅",       "🏛️"),
        "T": ("transactional / bank ✅",    "🏦"),
        "S": ("service / OTP ✅",           "⚙️"),
        "P": ("promotional ✅",              "📢"),
    }
    return suffix_map.get(h[-1].upper(), ("registered DLT ✅", "✅"))
```
In real Indian telecom operations under TRAI TCCCPR:
- Headers are assigned by telecom service providers (Access Providers) using a 2-character circle prefix followed by a hyphen and a **6-character Principal Entity Header** (e.g., `JK-SMRTBR`, `JM-AJIOIN`, `CP-MAXFSN`, `AM-AIRTEL`).
- The single-letter category suffixes (`-P`, `-T`, `-S`, `-G`) are appended dynamically or recorded in DLT metadata, but in millions of live delivered SMS messages on user handsets, the SMS display header is strictly `XX-YYYYYY` (9 characters total: 2 letters, hyphen, 6 alphanumeric).
- When `silverguard.py` evaluates `h[-1]` on `JK-SMRTBR`, `h[-1]` is `'R'`. It fails the suffix map lookup!
- Furthermore, in `label_sms.py`, `DLT_PATTERN = re.compile(r'^[A-Z]{2}-[A-Z0-9]{5,7}-[STGP]$')` strictly requires a hyphen before the suffix, which fails on legitimate standard headers like `JK-SMRTBR` or `AM-AIRTEL`.

### Root Cause 3: Semantic Confusion in User Labeling Guidelines
The project's `README.md` explicitly instructs the human annotator:
> `"label": 0` — **Ham** (safe, legitimate)  
> `"label": 1` — **Scam** (fraud, phishing, spam)  
> `unknown with links/offers: Probably scam (1) — especially crypto/lottery/loans`

This instruction directly commands human annotators to label commercial spam and promotional offers as "scam", corrupting the ground truth of the dataset.

### Root Cause 4: Lexical Attention Dominance over Header Tokens
In `main.ipynb` Cell 8, inputs are formatted as:
`HEADER [SEP] message body`
When passed into `MobileBertForSequenceClassification`, the cross-attention layers attend to both the header and the body. However:
- The body contains 30 to 100 tokens rich in high-arousal commercial keywords (`"SALE"`, `"50% OFF"`, `"BOGO"`, `"Holi"`, `"Last 2 Days"`).
- The header is only 3 to 5 WordPiece tokens.
- When an e-commerce advertisement arrives, the massive concentration of promotional tokens dominates the self-attention weights, overwhelming the header representation and driving the classification logit into the scam class.

---

## 5. Audit of Multilingual & Script Capabilities

### 5.1 Script Distribution in Real-World Indian SMS

An automated scan of the 5,226 messages in `data/rawdata.json` yields the following distribution:

| Script / Linguistic Modality | Message Count | Percentage | Real-World Context |
| :--- | :--- | :--- | :--- |
| **Latin (English / Hinglish / Romanized)** | 4,767 | 91.22% | English, Hinglish (*"Aapka account block..."*), Romanized Marathi |
| **Mixed: Devanagari + Latin** | 315 | 6.03% | Hindi/Marathi with English URLs, brand names, or phone numbers |
| **Pure Devanagari** | 141 | 2.70% | Formal government notices (e.g. Sanchar Saathi), carrier promos |
| **Other / Regional Scripts** | 3 | 0.06% | Regional telecom notifications |
| **Total Non-Latin Messages** | **456** | **8.73%** | Completely mishandled by current pipeline |

### 5.2 Failure of the Custom WordPiece Tokenizer

`silverguard.py` implements a standalone `WordpieceTokenizer` to avoid a runtime dependency on Hugging Face `transformers`. An audit of its normalization and tokenization code reveals severe defects:

#### A. Destructive Stripping of Indic Combining Characters (Matras)
Lines 71–74 of `silverguard.py`:
```python
@staticmethod
def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
```
- In Unicode, **all Indic vowel signs (*matras*) and viramas (*halant*) belong to category `Mn` (Mark, nonspacing)**. Examples include Devanagari `ा` (`\u093E`), `ि` (`\u093F`), `ी` (`\u0940`), `ु` (`\u0941`), `ू` (`\u0942`), `े` (`\u0947`), `ै` (`\u0948`), `ो` (`\u094B`), `ौ` (`\u094C`), `ं` (`\u0902`), and `्` (`\u094D`).
- When `_strip_accents` executes on Devanagari text, it decomposes the characters via NFD and **deletes every single vowel sign and conjunct virama**.
- **Empirical verification**:
  - Input: `पसंदीदा गाना को बनाएँ फ्री कॉलर ट्यून`
  - Output after `_strip_accents`: `पसदीदा गाना को बनाए फरी कॉलर टयन`
  - Words like `फ्री` (Free) become `फरी`, `ट्यून` (Tune) becomes `टयन`, and `करें` becomes `कर`.
  - In languages like Marathi, grammatical verb markers are completely erased.

#### B. Severe Out-of-Vocabulary (`[UNK]`) Rates across Indic Languages
Testing `model/vocab.txt` (`google/mobilebert-uncased`) across representative Indian languages reveals:

```text
Telugu:  Total non-pad tokens:  6 | UNK count: 4 | UNK Rate: 66.7%
Tamil:   Total non-pad tokens: 19 | UNK count: 3 | UNK Rate: 15.8% (remainder split into single chars)
Marathi: Total non-pad tokens: 15 | UNK count: 1 | Matras stripped, fragmented into letters
Hinglish:Total non-pad tokens: 20 | UNK count: 0 | Split into unnatural subwords ('gaya' -> 'gay', '##a')
```

- In **Telugu**, 4 out of 5 words in an ordinary banking alert (*"మీ బ్యాంక్ ఖాతా నిలిपिవేయబడింది"*) map directly to `[UNK]`. The model receives: `[CLS] [UNK] [UNK] [UNK] [UNK] [SEP]`. It is mathematically blind to the text.
- In **Tamil**, text is shredded into single graphemic characters: `['[CLS]', '[UNK]', '[UNK]', '[UNK]', 'ம', '##ு', '##ட', '##க', '##க', '##ப', '##ப', '##ட', '##ட', '##ு', '##ள', '##ள', '##த', '##ு', '[SEP]']`.
- In **Hinglish**, while the Latin characters are known to WordPiece, the subword boundaries are trained on standard English prose. Common Hindi words like `gaya` decompose into `gay` and `##a`, creating noisy representations and false activations in downstream classification layers.

---

## 6. Audit of Model Architecture & ONNX Pipeline

### 6.1 Model Architecture Specifications

- **Base Model**: `google/mobilebert-uncased`
- **Parameters**: 24.5 Million parameters
- **Sequence Length**: `max_length = 128`
- **Current Export Size**: `silver_guard.onnx` is **105,331,457 bytes (~100.4 MB)** (Unquantized FP32)
- **Quantization**: None. The model is stored as 32-bit floating point, exceeding GitHub's 100 MB file boundary and requiring external hosting on Hugging Face Hub.

### 6.2 Inference Logic Flaws in `silverguard.py`

In `silverguard.py`, the thresholding table is hardcoded as:
```python
THRESHOLDS = [
    (0.80, "🚨 HIGH RISK SCAM",  "Strong scam indicators detected."),
    (0.55, "⚠️  LIKELY SCAM",    "Probably a scam — exercise caution."),
    (0.40, "🟡 BORDERLINE",      "Ambiguous — treat with caution."),
    (0.00, "✅ SAFE (HAM)",      "Message looks legitimate."),
]
```
1. There is no calibrated probability calculation; the raw softmax logit from a model trained on heavily distorted synthetic ratios is used directly.
2. Any message scoring $\ge 0.55$ is displayed with an alarmist siren emoji (`🚨`) and labelled as "HIGH RISK SCAM" or "LIKELY SCAM", directly inducing user panic when ordinary retail marketing arrives.

---

## 7. Summary of Findings & Verification Status

| Audit Finding | Current Status | Impact Severity | Root Cause Location |
| :--- | :--- | :--- | :--- |
| **Spam-to-Scam Misclassification** | **Verified** | **Critical** | Conflation of classes in `main.ipynb`, `label_sms.py`, and `README.md` |
| **Synthetic DLT Header Leakage** | **Verified** | **Critical** | `add_header_to_kaggle_row()` in `main.ipynb` Cell 8 |
| **Indic Vowel Stripping (Matra Deletion)** | **Verified** | **Critical** | `_strip_accents()` in `silverguard.py` Line 73 |
| **Monolingual Vocab Blindness ([UNK] spikes)** | **Verified** | **Critical** | `google/mobilebert-uncased` `vocab.txt` |
| **Synthetic Template Data Leakage** | **Verified** | **Major** | Unstratified random `train_test_split` on template data in `main.ipynb` |
| **Zero Positive Scams in Personal Data** | **Verified** | **Major** | `data/training_data.json` contains 5,154 ham, 0 scam |
| **Incomplete DLT Regex in Header Analyzer** | **Verified** | **Major** | Regex assumes strict `-T/-S/-P/-G` suffix in `silverguard.py` |
| **Unquantized ONNX Artifact Size (105 MB)** | **Verified** | **Moderate** | Missing INT8 quantization step for mobile deployment |

The audit concludes that incremental fine-tuning of the current pipeline cannot resolve these issues. The system requires a fundamental restructuring of its classification taxonomy, multilingual representation engine, dataset curation pipeline, and edge-inference architecture.
