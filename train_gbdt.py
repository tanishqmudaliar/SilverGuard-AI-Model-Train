"""
Silver Guard — Production GBDT Feature Fusion 4-Class Training Pipeline
========================================================================
Implements the research-recommended on-device architecture:
  • FastText-style Subword Character N-Grams (lengths 3 to 5)
  • 12 Handcrafted Telecom Domain Features (DLT, URLs, shorteners, APKs, urgency, scripts)
  • LightGBM Multi-Class GBDT with class-balanced cross-entropy loss

Outputs:
  • model/silverguard_gbdt.joblib (Trained LightGBM model, < 5 MB disk footprint)
  • model/char_vectorizer.joblib (Subword character n-gram vectorizer)
  • model/feature_names.json (Feature metadata)
  • Evaluation report with 4x4 Confusion Matrix, Macro-F1, Precision, and Recall
"""

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from label_sms import parse_sender_info, normalize_phone, DLT_PATTERN, SUSPICIOUS_SHORTENERS, EXTORTION_KEYWORDS, KYC_PHISHING_KEYWORDS, FRAUD_LURE_KEYWORDS, PROMO_KEYWORDS

# ─────────────────────────────────────────────────────────────────────────────
# 1. Handcrafted Telecom Domain Feature Extractor (12 Features)
# ─────────────────────────────────────────────────────────────────────────────

def extract_domain_features(header: str, text: str) -> np.ndarray:
    """
    Extracts 12 engineered telecom domain features for tabular fusion:
      1. is_dlt_registered (0 or 1)
      2. dlt_suffix_type (0=None, 1=T, 2=S, 3=P, 4=G)
      3. is_phone_sender (0 or 1)
      4. url_count (integer)
      5. has_shortener (0 or 1)
      6. has_ip_url (0 or 1)
      7. has_apk (0 or 1)
      8. has_callback_phone (0 or 1)
      9. currency_count (integer)
      10. urgency_count (integer)
      11. uppercase_ratio (float in [0, 1])
      12. script_type (0=Latin, 1=Devanagari, 2=Dravidian, 3=Mixed)
    """
    header_str = str(header).strip().upper()
    text_str = str(text)
    lower = text_str.lower()

    # 1. DLT header detection
    m = DLT_PATTERN.match(header_str)
    is_dlt = 1.0 if m else 0.0

    # 2. Suffix type
    suffix_code = 0.0
    if m and m.group(2):
        sfx = m.group(2).replace("-", "").upper()
        if sfx == "T": suffix_code = 1.0
        elif sfx == "S": suffix_code = 2.0
        elif sfx == "P": suffix_code = 3.0
        elif sfx == "G": suffix_code = 4.0

    # 3. Phone sender
    is_phone = 1.0 if normalize_phone(header_str) else 0.0

    # 4. URLs
    urls = re.findall(r"(?:https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)", text_str)
    url_count = float(len(urls))

    # 5, 6, 7. URL risk types
    has_shortener = 1.0 if any(s in lower for s in SUSPICIOUS_SHORTENERS) or "[url_shortener]" in lower else 0.0
    has_ip = 1.0 if re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text_str) or "[url_ip]" in lower else 0.0
    has_apk = 1.0 if ".apk" in lower or "[url_apk]" in lower else 0.0

    # 8. Callback phone in body
    phones = re.findall(r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b", text_str)
    has_callback = 1.0 if (len(phones) > 0 and not is_phone) or "[phone_num]" in lower else 0.0

    # 9. Currency count
    currency_count = float(len(re.findall(r"(?:₹|rs\.?|inr|lakh|crore)", lower)))

    # 10. Urgency / threat triggers
    urgency_count = 0.0
    for pat_list in (EXTORTION_KEYWORDS, KYC_PHISHING_KEYWORDS, FRAUD_LURE_KEYWORDS):
        for pat in pat_list:
            if re.search(pat, lower):
                urgency_count += 1.0

    # 11. Uppercase character ratio
    letters = [c for c in text_str if c.isalpha()]
    upper_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0

    # 12. Script type
    has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text_str)
    has_dravidian = any(0x0B80 <= ord(c) <= 0x0D7F for c in text_str)
    if has_devanagari and has_dravidian: script_code = 3.0
    elif has_devanagari: script_code = 1.0
    elif has_dravidian: script_code = 2.0
    else: script_code = 0.0

    return np.array([
        is_dlt, suffix_code, is_phone, url_count, has_shortener,
        has_ip, has_apk, has_callback, currency_count, urgency_count,
        upper_ratio, script_code
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Main Training & Evaluation Routine
# ─────────────────────────────────────────────────────────────────────────────

def train_silverguard_gbdt(
    dataset_path: str = "data/silverguard_indic_unified.json",
    model_output_dir: str = "model"
) -> Dict[str, Any]:
    print("=" * 70)
    print("SILVER GUARD — GBDT + FASTTEXT SUBWORD FEATURE FUSION TRAINING")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run build_unified_dataset.py first.")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    print(f"Loaded {len(messages)} unique records from {dataset_path}")

    # Extract texts, headers, labels
    texts = [m.get("text", "") for m in messages]
    headers = [m.get("address", "") for m in messages]
    labels = np.array([int(m.get("label", 0)) for m in messages], dtype=np.int32)

    counts = Counter(labels)
    print("\nTraining Class Distribution:")
    class_names = ["Ham (0)", "Promotional (1)", "Commercial Spam (2)", "Scam (3)"]
    for i, name in enumerate(class_names):
        print(f"  • {name:<22} : {counts[i]:5d} ({counts[i]/len(labels)*100:5.1f}%)")

    # 1. Feature Extraction: FastText-style Character Subwords (n-grams 3 to 5)
    print("\nFitting Subword Character N-Gram Vectorizer (FastText representations)...")
    start_v = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=12000,
        sublinear_tf=True
    )
    X_text = vectorizer.fit_transform(texts)
    print(f"  • Extracted {X_text.shape[1]} subword character n-gram features in {time.time()-start_v:.2f} s")

    # 2. Extract Tabular Telecom Features
    print("Extracting 12 Telecom Domain Features...")
    start_f = time.time()
    tabular_features = np.zeros((len(messages), 12), dtype=np.float32)
    for i, (hdr, txt) in enumerate(zip(headers, texts)):
        tabular_features[i] = extract_domain_features(hdr, txt)
    X_tabular = csr_matrix(tabular_features)
    print(f"  • Handcrafted domain feature matrix constructed in {time.time()-start_f:.2f} s")

    # 3. Feature Fusion: Stack Character Subwords + Domain Features
    X_all = hstack([X_text, X_tabular]).tocsr()
    print(f"Fused Feature Space Dimension: {X_all.shape[1]} features")

    # 4. Stratified Train / Test Split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, labels, test_size=0.20, random_state=42, stratify=labels
    )
    print(f"\nTrain Split: {X_train.shape[0]} samples | Test Split: {X_test.shape[0]} samples")

    # 5. Train LightGBM Multi-Class Model
    print("\nTraining LightGBM Multi-Class Classifier...")
    start_t = time.time()
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        class_weight="balanced",
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_train, y_train)
    train_time = time.time() - start_t
    print(f"✅ Training completed in {train_time:.2f} seconds.")

    # 6. Comprehensive Evaluation on Held-Out Test Set
    print("\n" + "=" * 70)
    print("TEST EVALUATION PERFORMANCE ON HELD-OUT DATA (4-CLASS TAXONOMY)")
    print("=" * 70)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(
        y_test, y_pred, target_names=["Ham", "Promo", "Spam_UCC", "Scam"], digits=4
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("4x4 Confusion Matrix (Rows = Actual, Columns = Predicted):")
    print("               Pred_Ham  Pred_Promo  Pred_Spam  Pred_Scam")
    for i, name in enumerate(["Actual_Ham  ", "Actual_Promo", "Actual_Spam ", "Actual_Scam "]):
        print(f"{name} : {cm[i][0]:9d}  {cm[i][1]:10d}  {cm[i][2]:9d}  {cm[i][3]:9d}")

    # 7. Model Persistence
    os.makedirs(model_output_dir, exist_ok=True)
    gbdt_path = os.path.join(model_output_dir, "silverguard_gbdt.joblib")
    vec_path = os.path.join(model_output_dir, "char_vectorizer.joblib")
    meta_path = os.path.join(model_output_dir, "gbdt_metadata.json")

    joblib.dump(clf, gbdt_path, compress=3)
    joblib.dump(vectorizer, vec_path, compress=3)

    model_size_mb = os.path.getsize(gbdt_path) / (1024 * 1024)
    vec_size_mb = os.path.getsize(vec_path) / (1024 * 1024)
    print(f"\nModel Artifacts Saved:")
    print(f"  • LightGBM Model   : {gbdt_path} ({model_size_mb:.2f} MB)")
    print(f"  • Char Vectorizer  : {vec_path} ({vec_size_mb:.2f} MB)")

    metadata = {
        "model_type": "LightGBM + FastText Subword Char N-Grams",
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "confusion_matrix": cm.tolist(),
        "model_size_mb": round(model_size_mb, 2),
        "vectorizer_size_mb": round(vec_size_mb, 2),
        "features": {
            "n_text_features": X_text.shape[1],
            "n_domain_features": 12,
            "total_features": X_all.shape[1]
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata

if __name__ == "__main__":
    train_silverguard_gbdt()
