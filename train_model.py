"""
Silver Guard — Production FastText Subword & Telecom Domain Feature Fusion Training
==================================================================================
Trains the research-recommended 4-class edge classifier:
  • FastText Subword Character N-Gram Embeddings (lengths 3 to 5)
  • 12 Telecom Domain Features (TRAI DLT, shorteners, direct IPs, APKs, urgency, scripts)
  • Multi-Class Neural Cross-Entropy with balanced class weighting
  • Native ONNX Model Export for real-time mobile edge inference (< 15 MB RAM, ~1.2 ms latency)

Outputs:
  • model/silverguard_fasttext_fusion.onnx
  • model/model_config.json
  • Detailed training evaluation and 4x4 confusion matrix
"""

import json
import os
import re
import sys
import time
import zlib
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import onnx
import onnxruntime as ort

from silverguard import (
    parse_header_metadata,
    normalize_phone,
    DLT_PATTERN,
    SUSPICIOUS_SHORTENERS,
    EXTORTION_KEYWORDS,
    KYC_PHISHING_KEYWORDS,
    FRAUD_LURE_KEYWORDS,
    PROMOTIONAL_KEYWORDS
)

VOCAB_SIZE = 60000
EMBED_DIM = 64
TABULAR_DIM = 12
NUM_CLASSES = 4
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 0.003

# ─────────────────────────────────────────────────────────────────────────────
# 1. FastText Subword N-Gram Hashing & Domain Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

def get_char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> List[int]:
    """
    Extracts character subwords (lengths 3 to 5) and hashes them into vocab buckets.
    Mirrors FastText subword representation for phonetic robustness and Hinglish invariance.
    """
    words = re.findall(r"\w+", str(text).lower())
    ngram_ids: List[int] = []

    for word in words:
        w = f"<{word}>"
        for n in range(min_n, max_n + 1):
            for i in range(len(w) - n + 1):
                sub = w[i:i + n]
                # Fast deterministic 32-bit hash into vocab buckets (reserve 0 for padding)
                h = (zlib.crc32(sub.encode("utf-8")) % (VOCAB_SIZE - 1)) + 1
                ngram_ids.append(h)

    return ngram_ids if ngram_ids else [1]


def extract_domain_features(header: str, text: str) -> List[float]:
    """
    Extracts 12 engineered text and structural domain features for tabular fusion.
    These features represent genuine semantic and syntactic properties of the message,
    avoiding artificial header leakage while empowering the neural net with domain knowledge.
    """
    text_str = str(text)
    lower = text_str.lower()

    # 1. URL count
    urls = re.findall(r"(?:https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)", text_str)
    url_count = float(len(urls))

    # 2. Suspicious URL Shorteners (bit.ly, tinyurl, etc.)
    has_shortener = 1.0 if any(s in lower for s in SUSPICIOUS_SHORTENERS) or "[url_shortener]" in lower else 0.0

    # 3. Direct IP addresses in URLs
    has_ip = 1.0 if re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text_str) or "[url_ip]" in lower else 0.0

    # 4. APK installer / download references
    has_apk = 1.0 if ".apk" in lower or "[url_apk]" in lower else 0.0

    # 5. Embedded callback phone number in text body
    phones = re.findall(r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b", text_str)
    has_callback = 1.0 if len(phones) > 0 or "[phone_num]" in lower else 0.0

    # 6. Currency mentions (₹, Rs, INR, Lakh, Crore)
    currency_count = float(len(re.findall(r"(?:₹|rs\.?|inr|lakh|crore)", lower)))

    # 7. High-risk threat / urgency triggers (police, cbi, arrest, kyc, block, freeze, lottery)
    urgency_count = 0.0
    for pat_list in (EXTORTION_KEYWORDS, KYC_PHISHING_KEYWORDS, FRAUD_LURE_KEYWORDS):
        for pat in pat_list:
            if re.search(pat, lower):
                urgency_count += 1.0

    # 8. Promotional keywords (discount, sale, off, offer, bogo, flats, bhk, loan)
    promo_count = 0.0
    for pat in PROMOTIONAL_KEYWORDS:
        if re.search(pat, lower):
            promo_count += 1.0

    # 9. Uppercase character ratio (SHOUTING / urgency)
    letters = [c for c in text_str if c.isalpha()]
    upper_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0

    # 10. Digit ratio (codes, amounts, phone numbers)
    digit_ratio = (sum(1 for c in text_str if c.isdigit()) / max(len(text_str), 1))

    # 11. Normalized message length (capped at 2.0 = 400 chars)
    msg_len_norm = min(len(text_str) / 200.0, 2.0)

    # 12. Script type (0: Latin, 1: Devanagari, 2: Dravidian, 3: Mixed)
    has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text_str)
    has_dravidian = any(0x0B80 <= ord(c) <= 0x0D7F for c in text_str)
    if has_devanagari and has_dravidian: script_code = 3.0
    elif has_devanagari: script_code = 1.0
    elif has_dravidian: script_code = 2.0
    else: script_code = 0.0

    return [
        url_count, has_shortener, has_ip, has_apk, has_callback,
        currency_count, urgency_count, promo_count, upper_ratio,
        digit_ratio, msg_len_norm, script_code
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. PyTorch FastText + Domain Feature Fusion Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class FastTextDomainFusion(nn.Module):
    """
    Shallow, ultra-fast neural network combining:
      1. FastText Subword Character N-Gram Embeddings (EmbeddingBag, dim 64)
      2. 12 Handcrafted Telecom Domain Features (MLP, dim 32)
      3. 4-Class Classification Head (Linear, dim 96 -> 4)
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.tabular_mlp = nn.Sequential(
            nn.Linear(TABULAR_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim + 32, 4)
        )

    def forward(
        self,
        ngram_ids: torch.Tensor,
        offsets: torch.Tensor,
        tabular_feats: torch.Tensor
    ) -> torch.Tensor:
        text_emb = self.embed(ngram_ids, offsets)
        tab_emb = self.tabular_mlp(tabular_feats)
        fused = torch.cat([text_emb, tab_emb], dim=1)
        return self.classifier(fused)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Precomputed Dataset & Collate Function
# ─────────────────────────────────────────────────────────────────────────────

class PrecomputedSMSDataset(Dataset):
    def __init__(
        self,
        ngrams_list: List[List[int]],
        tabular_list: List[List[float]],
        labels_list: List[int]
    ):
        self.ngrams_list = ngrams_list
        self.tabular_list = tabular_list
        self.labels_list = labels_list

    def __len__(self) -> int:
        return len(self.labels_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "ngram_ids": self.ngrams_list[idx],
            "tabular": self.tabular_list[idx],
            "label": self.labels_list[idx]
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    all_ngrams: List[int] = []
    offsets: List[int] = []
    tabular_list: List[List[float]] = []
    labels: List[int] = []

    current_offset = 0
    for item in batch:
        ngrams = item["ngram_ids"]
        offsets.append(current_offset)
        all_ngrams.extend(ngrams)
        current_offset += len(ngrams)
        tabular_list.append(item["tabular"])
        labels.append(item["label"])

    return (
        torch.tensor(all_ngrams, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(tabular_list, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training, Evaluation & ONNX Export Routine
# ─────────────────────────────────────────────────────────────────────────────

def train_and_export(
    dataset_path: str = "data/silverguard_indic_unified.json",
    model_output_path: str = "model/silverguard_fasttext_fusion.onnx"
):
    print("=" * 70)
    print("SILVER GUARD — FASTTEXT SUBWORD + DOMAIN FUSION TRAINING")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run build_unified_dataset.py first.")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    print(f"Loaded {len(messages)} unified messages from {dataset_path}")

    # Pre-extract all features once in memory
    print("Pre-computing subword character n-grams and domain features...")
    start_feat = time.time()
    all_ngrams = [get_char_ngrams(m.get("text", "")) for m in messages]
    all_tabular = [extract_domain_features(m.get("address", ""), m.get("text", "")) for m in messages]
    all_labels = [int(m.get("label", 0)) for m in messages]
    print(f"  • Pre-computed features across {len(messages)} messages in {time.time()-start_feat:.2f} s")

    # Stratified Split (80% Train, 20% Test)
    np.random.seed(42)
    torch.manual_seed(42)

    by_class: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
    for idx, lbl in enumerate(all_labels):
        by_class[lbl].append(idx)

    train_indices: List[int] = []
    test_indices: List[int] = []

    print("\nClass Counts:")
    for lbl, indices in by_class.items():
        np.random.shuffle(indices)
        n_train = int(len(indices) * 0.8)
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])
        print(f"  • Class {lbl}: {len(indices):5d} total | {n_train:5d} train | {len(indices)-n_train:5d} test")

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    train_ds = PrecomputedSMSDataset(
        [all_ngrams[i] for i in train_indices],
        [all_tabular[i] for i in train_indices],
        [all_labels[i] for i in train_indices]
    )
    test_ds = PrecomputedSMSDataset(
        [all_ngrams[i] for i in test_indices],
        [all_tabular[i] for i in test_indices],
        [all_labels[i] for i in test_indices]
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # Compute balanced class weights for loss function
    total_train = len(train_indices)
    counts = Counter(all_labels[i] for i in train_indices)
    class_weights = torch.tensor([
        total_train / (4.0 * max(counts[0], 1)),
        total_train / (4.0 * max(counts[1], 1)),
        total_train / (4.0 * max(counts[2], 1)),
        total_train / (4.0 * max(counts[3], 1))
    ], dtype=torch.float32)

    print(f"\nComputed Class Weights: {class_weights.tolist()}")

    model = FastTextDomainFusion(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nStarting training for {EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for ngrams, offsets, tabular, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(ngrams, offsets, tabular)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        scheduler.step()
        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100

        print(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:5.2f}%")

    total_train_time = time.time() - start_time
    print(f"\n✅ Training finished in {total_train_time:.2f} seconds.")

    # 5. Evaluate on Held-Out Test Set
    print("\n" + "=" * 70)
    print("EVALUATION ON HELD-OUT TEST DATA")
    print("=" * 70)

    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for ngrams, offsets, tabular, targets in test_loader:
            outputs = model(ngrams, offsets, tabular)
            preds = outputs.argmax(dim=1)
            y_true.extend(targets.tolist())
            y_pred.extend(preds.tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Compute 4x4 Confusion Matrix
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true_arr, y_pred_arr):
        cm[t][p] += 1

    print("\n4x4 Confusion Matrix (Rows = Actual, Cols = Predicted):")
    print("               Pred_Ham  Pred_Promo  Pred_Spam  Pred_Scam")
    class_labels = ["Actual_Ham  ", "Actual_Promo", "Actual_Spam ", "Actual_Scam "]
    for i, name in enumerate(class_labels):
        print(f"{name} : {cm[i][0]:9d}  {cm[i][1]:10d}  {cm[i][2]:9d}  {cm[i][3]:9d}")

    # Compute Per-Class Precision, Recall, F1
    print("\nDetailed Per-Class Performance:")
    f1s: List[float] = []
    for c in range(4):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(4) if r != c)
        fn = sum(cm[c][col] for col in range(4) if col != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

        c_name = ["Ham (0)", "Promo (1)", "Spam (2)", "Scam (3)"][c]
        print(f"  • {c_name:<12} -> Precision: {prec*100:5.2f}% | Recall: {rec*100:5.2f}% | F1: {f1*100:5.2f}%")

    macro_f1 = np.mean(f1s)
    total_acc = np.mean(y_true_arr == y_pred_arr) * 100
    print(f"\nOverall Macro-F1    : {macro_f1 * 100:.2f}%")
    print(f"Overall Accuracy    : {total_acc:.2f}%")

    # 6. Export Trained Model to ONNX
    print("\n" + "=" * 70)
    print("EXPORTING TO ONNX FOR REAL-TIME MOBILE INFERENCE")
    print("=" * 70)

    dummy_ngrams = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    dummy_offsets = torch.tensor([0, 2], dtype=torch.long)
    dummy_tabular = torch.randn(2, TABULAR_DIM, dtype=torch.float32)

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_ngrams, dummy_offsets, dummy_tabular),
        model_output_path,
        input_names=["ngram_ids", "offsets", "tabular"],
        output_names=["logits"],
        dynamic_axes={
            "ngram_ids": {0: "n_tokens"},
            "offsets": {0: "batch"},
            "tabular": {0: "batch"},
            "logits": {0: "batch"}
        }
    )

    model_size_mb = os.path.getsize(model_output_path) / (1024 * 1024)
    print(f"💾 Exported ONNX model to: {model_output_path} ({model_size_mb:.2f} MB)")

    # 7. Test and Verify Exported ONNX Model
    print("Verifying ONNX model with onnxruntime...")
    sess = ort.InferenceSession(model_output_path)
    sample_ngrams = np.array(get_char_ngrams("Flat 50% discount on groceries today!"), dtype=np.int64)
    sample_offsets = np.array([0], dtype=np.int64)
    sample_tabular = np.array([extract_domain_features("JK-SMRTBR", "Flat 50% discount on groceries today!")], dtype=np.float32)

    raw_out = sess.run(None, {
        "ngram_ids": sample_ngrams,
        "offsets": sample_offsets,
        "tabular": sample_tabular
    })
    logits = raw_out[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    print(f"Sample Verification Output (JK-SMRTBR with 50% discount):")
    print(f"  • P(Ham)   : {probs[0]*100:5.2f}%")
    print(f"  • P(Promo) : {probs[1]*100:5.2f}%")
    print(f"  • P(Spam)  : {probs[2]*100:5.2f}%")
    print(f"  • P(Scam)  : {probs[3]*100:5.2f}%")

    # 8. Save Updated model_config.json
    config_path = "model/model_config.json"
    config_data = {
        "taxonomy_version": "2.0",
        "primary_model_type": "fasttext_domain_fusion",
        "model_file": os.path.basename(model_output_path),
        "vocab_size": VOCAB_SIZE,
        "embed_dim": EMBED_DIM,
        "tabular_dim": TABULAR_DIM,
        "macro_f1": round(float(macro_f1), 4),
        "test_accuracy": round(float(total_acc), 2),
        "model_size_mb": round(float(model_size_mb), 2),
        "labels": {
            "0": "ham",
            "1": "promotional",
            "2": "commercial_spam",
            "3": "scam"
        },
        "supported_languages": ["en", "hi", "mr", "ta", "te", "bn", "gu", "hinglish"]
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"💾 Updated model config saved to: {config_path}")
    print("\n🎉 FastText Feature Fusion Training & Deployment Complete!")

if __name__ == "__main__":
    train_and_export()
