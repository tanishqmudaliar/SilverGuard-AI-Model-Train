"""
Silver Guard — Multilingual Dataset Preparation & Preprocessing Pipeline
========================================================================
Production-grade preprocessing, PII redaction, URL risk classification,
template deduplication, and multi-source alignment for Indian SMS classification.

Features:
  1. Indic-Safe Canonical Normalization (Devanagari, Bengali, Tamil, Telugu, etc.)
  2. PII Sanitization (Phone numbers, Bank Accounts, OTP tokens)
  3. Evidentiary URL Risk Tokenization ([URL_SHORTENER], [URL_IP], [URL_APK], [URL_WHITELISTED], [URL])
  4. Template Near-Duplicate Filtering (Shingle Jaccard clustering)
  5. Multi-Source Dataset Ingestion & 4-Class Schema Alignment
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1. Indic-Safe Normalization
# ─────────────────────────────────────────────────────────────────────────────

def is_indic_mark(ch: str) -> bool:
    """Checks if character is a dependent vowel sign or virama in South Asian scripts."""
    cp = ord(ch)
    return 0x0900 <= cp <= 0x0D7F


def normalize_indic_text(text: str) -> str:
    """
    Normalizes multilingual text canonically (NFC) while:
      1. Strictly preserving Indic vowel signs (matras), viramas, and nuktas.
      2. Separating emojis and symbols from adjacent words to prevent [UNK] collapses.
      3. Separating Indic punctuation (danda / double danda).
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    out: List[str] = []
    for ch in text:
        cp = ord(ch)
        cat = unicodedata.category(ch)

        # Separate emojis and symbols from alphanumeric characters
        if cat in ("So", "Sk") or (0x1F300 <= cp <= 0x1FAFF) or (0x2600 <= cp <= 0x27BF):
            out.append(f" {ch} ")
        # Separate Indic Danda (।) and Double Danda (॥)
        elif cp in (0x0964, 0x0965):
            out.append(f" {ch} ")
        # Strictly preserve Indic non-spacing marks (matras, viramas)
        elif is_indic_mark(ch):
            out.append(ch)
        # Strip Western Latin combining accents only
        elif cat == "Mn":
            continue
        # Remove control characters and replacement chars
        elif cat == "Cc" or cp == 0 or cp == 0xFFFD:
            continue
        # Normalize whitespace
        elif cat == "Zs" or ch == "\t":
            out.append(" ")
        else:
            out.append(ch)

    cleaned = "".join(out)
    return re.sub(r"\s+", " ", cleaned).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. PII Sanitization & Redaction Engine
# ─────────────────────────────────────────────────────────────────────────────

# Mobile phone regex: 10-digit Indian numbers with optional country code or leading 0
PHONE_REGEX = re.compile(
    r"(?:\+?91[\s-]?)?(?:0)?[6-9]\d{4}[\s-]?\d{5}\b",
    re.IGNORECASE
)

# Bank account patterns: e.g. A/c *******4120, A/c No. 123456789012, Account ending 9876
ACCT_REGEX = re.compile(
    r"\b(?:a/c|acct|account|acc)\s*(?:no\.?|num|number)?\s*[:\-\s]*([xX*]*\d{3,18}|\d{9,18})\b",
    re.IGNORECASE
)

# OTP regex: 4 to 8 digit codes following OTP/code/verification keywords
OTP_REGEX = re.compile(
    r"\b(?:otp|code|one\s+time\s+password|verification\s+code|secret\s+code)\s*(?:is|:|-)?\s*([0-9]{4,8})\b",
    re.IGNORECASE
)


def redact_pii(text: str) -> str:
    """
    Redacts personally identifiable information (PII) to protect privacy during model training.
    Replaces:
      • OTP numbers with [OTP_CODE]
      • Bank account numbers with [ACCT_NUM]
      • Indian mobile numbers with [PHONE_NUM]
    """
    if not text:
        return ""

    # 1. Redact OTPs
    def _mask_otp(match: re.Match) -> str:
        full = match.group(0)
        code = match.group(1)
        return full.replace(code, "[OTP_CODE]")

    text = OTP_REGEX.sub(_mask_otp, text)

    # 2. Redact Account numbers
    def _mask_acct(match: re.Match) -> str:
        full = match.group(0)
        acct_val = match.group(1)
        return full.replace(acct_val, "[ACCT_NUM]")

    text = ACCT_REGEX.sub(_mask_acct, text)

    # 3. Redact Mobile Phone numbers (ensure we don't redact [OTP_CODE] or [ACCT_NUM])
    text = PHONE_REGEX.sub("[PHONE_NUM]", text)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evidentiary URL Risk Classification & Tokenization
# ─────────────────────────────────────────────────────────────────────────────

SUSPICIOUS_SHORTENERS = {
    "bit.ly", "tinyurl.com", "is.gd", "rb.gy", "cutt.ly", "shorturl.at",
    "t.co", "ow.ly", "buff.ly", "goo.gl", "bit.do"
}

WHITELISTED_ROOT_DOMAINS = {
    "gov.in", "nic.in", "hdfcbank.com", "icicibank.com", "sbi.co.in",
    "axisbank.com", "kotak.com", "pnbindia.in", "bankofbaroda.in",
    "jio.com", "airtel.in", "myvi.in", "irctc.co.in", "uidai.gov.in",
    "incometax.gov.in", "epfindia.gov.in", "sancharsaathi.gov.in"
}


def classify_and_tokenize_urls(text: str) -> str:
    """
    Identifies URLs and replaces them with semantic threat tokens:
      • [URL_SHORTENER]: Obfuscated link shorteners commonly used in smishing.
      • [URL_IP]: Direct IP URLs bypassing DNS domain registration.
      • [URL_APK]: Direct Android APK installation packages.
      • [URL_WHITELISTED]: Trusted banks and Indian government portals.
      • [URL]: Generic web links.
    """
    if not text:
        return ""

    url_pattern = re.compile(
        r"(https?://[^\s/$.?#].[^\s]*|www\.[^\s/$.?#].[^\s]*|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)",
        re.IGNORECASE
    )

    def _replace_url(match: re.Match) -> str:
        u = match.group(0)
        u_lower = u.lower()

        # Reject common false positives (e.g. abbreviations ending with periods)
        if not ("." in u and len(u) > 4 and not u.endswith(".")):
            return u

        # APK download
        if ".apk" in u_lower or "download" in u_lower and ("apk" in u_lower or "app" in u_lower):
            return "[URL_APK]"

        # Direct IP address
        if re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", u):
            return "[URL_IP]"

        # Link shorteners
        if any(shortener in u_lower for shortener in SUSPICIOUS_SHORTENERS):
            return "[URL_SHORTENER]"

        # Whitelisted verified domains
        if any(wld in u_lower for wld in WHITELISTED_ROOT_DOMAINS):
            return "[URL_WHITELISTED]"

        return "[URL]"

    return url_pattern.sub(_replace_url, text)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Template Near-Duplicate Deduplication (Shingling & Jaccard)
# ─────────────────────────────────────────────────────────────────────────────

def get_word_shingles(text: str, k: int = 2) -> Set[str]:
    """Extracts word k-shingles from normalized text."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    """Computes Jaccard similarity index between two shingle sets."""
    if not s1 or not s2:
        return 0.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union > 0 else 0.0


def deduplicate_dataset(
    messages: List[Dict[str, Any]],
    similarity_threshold: float = 0.85
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Deduplicates messages based on normalized shingle similarity to prevent
    template leakage across training/validation splits.
    Optimized with an inverted index for sub-second scaling over large datasets.
    """
    unique_messages: List[Dict[str, Any]] = []
    shingle_index: Dict[str, List[int]] = {}  # shingle -> list of item indices
    item_shingles: List[Set[str]] = []
    item_labels: List[int] = []
    duplicate_count = 0

    for msg in messages:
        text = str(msg.get("text", ""))
        label = int(msg.get("label", 0))

        shingles = get_word_shingles(text, k=2)
        if not shingles:
            unique_messages.append(msg)
            continue

        # Find candidates sharing at least one shingle and same label
        candidate_counts: Counter = Counter()
        for sh in shingles:
            if sh in shingle_index:
                for idx in shingle_index[sh]:
                    if item_labels[idx] == label:
                        candidate_counts[idx] += 1

        is_dup = False
        min_overlap = int(len(shingles) * similarity_threshold * 0.7)
        for cand_idx, overlap in candidate_counts.items():
            if overlap >= min_overlap:
                sim = jaccard_similarity(shingles, item_shingles[cand_idx])
                if sim >= similarity_threshold:
                    is_dup = True
                    break

        if is_dup:
            duplicate_count += 1
        else:
            new_idx = len(item_shingles)
            unique_messages.append(msg)
            item_shingles.append(shingles)
            item_labels.append(label)
            for sh in shingles:
                if sh not in shingle_index:
                    shingle_index[sh] = []
                shingle_index[sh].append(new_idx)

    return unique_messages, duplicate_count


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-Source Ingestion & Schema Alignment
# ─────────────────────────────────────────────────────────────────────────────

def align_external_dataset(source: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Ingests external datasets and maps them to the 4-Class Taxonomy:
      0: Ham
      1: Promotional
      2: Commercial Spam
      3: Scam / Phishing
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    aligned: List[Dict[str, Any]] = []

    if source.lower() == "atherion":
        # Atherion 30K dataset format (CSV: text, label)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                txt = row.get("text", "") or row.get("message", "")
                raw_lbl = str(row.get("label", "")).strip().lower()

                # Map Atherion labels
                if raw_lbl in ("ham", "safe", "0"):
                    lbl = 0
                elif raw_lbl in ("promo", "promotional", "advertisement"):
                    lbl = 1
                elif raw_lbl in ("spam", "commercial_spam", "ucc"):
                    lbl = 2
                elif raw_lbl in ("phishing", "scam", "fraud", "malicious", "1"):
                    lbl = 3
                else:
                    continue

                aligned.append({
                    "text": txt,
                    "address": row.get("sender", row.get("header", "")),
                    "label": lbl,
                    "source": "atherion_30k"
                })

    elif source.lower() == "bolewara":
        # Bolewara Hinglish dataset format (JSON or CSV)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("messages", [])
            for row in items:
                txt = row.get("text", "")
                lbl_val = str(row.get("label", "")).strip().lower()
                lbl = 3 if lbl_val in ("spam", "scam", "1") else 0
                aligned.append({
                    "text": txt,
                    "address": row.get("address", ""),
                    "label": lbl,
                    "source": "bolewara_hinglish"
                })

    elif source.lower() == "personal":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("messages", []) if isinstance(data, dict) else data
        for row in items:
            lbl = row.get("label")
            if lbl is not None:
                aligned.append({
                    "text": row.get("text", ""),
                    "address": row.get("address", ""),
                    "label": int(lbl),
                    "source": "personal_export"
                })

    else:
        raise ValueError(f"Unsupported dataset source: '{source}'. Choose 'personal', 'atherion', or 'bolewara'.")

    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Silver Guard — Multilingual Dataset Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline action to run")

    # Command: redact
    p_redact = subparsers.add_parser("redact", help="Redact PII from a single string or file")
    p_redact.add_argument("--text", help="Direct text string to redact")
    p_redact.add_argument("--input", help="Input JSON file to redact")
    p_redact.add_argument("--output", help="Output JSON file path")

    # Command: normalize
    p_norm = subparsers.add_parser("normalize", help="Run Indic-safe normalization and URL tokenization")
    p_norm.add_argument("--text", help="Direct text string to normalize")
    p_norm.add_argument("--input", help="Input JSON file to normalize")
    p_norm.add_argument("--output", help="Output JSON file path")

    # Command: dedup
    p_dedup = subparsers.add_parser("dedup", help="Deduplicate messages via Jaccard template clustering")
    p_dedup.add_argument("--input", required=True, help="Input JSON dataset file")
    p_dedup.add_argument("--output", required=True, help="Output deduplicated JSON file")
    p_dedup.add_argument("--threshold", type=float, default=0.85, help="Jaccard similarity cutoff (default: 0.85)")

    # Command: align
    p_align = subparsers.add_parser("align", help="Align an external dataset into Silver Guard 4-Class format")
    p_align.add_argument("--source", required=True, choices=["personal", "atherion", "bolewara"], help="Dataset source type")
    p_align.add_argument("--input", required=True, help="Input raw dataset path")
    p_align.add_argument("--output", required=True, help="Output aligned JSON file")

    # Command: stats
    p_stats = subparsers.add_parser("stats", help="Compute class counts and token distribution")
    p_stats.add_argument("--input", required=True, help="Path to Silver Guard JSON dataset")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "redact":
        if args.text:
            cleaned = redact_pii(args.text)
            print(f"\n[Original]: {args.text}")
            print(f"[Redacted]: {cleaned}")
        elif args.input and args.output:
            with open(args.input, "r", encoding="utf-8") as f:
                data = json.load(f)
            msgs = data.get("messages", []) if isinstance(data, dict) else data
            for m in msgs:
                m["text"] = redact_pii(m.get("text", ""))
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ Redacted PII across {len(msgs)} messages. Saved to {args.output}")

    elif args.command == "normalize":
        if args.text:
            norm = normalize_indic_text(args.text)
            tok = classify_and_tokenize_urls(norm)
            print(f"\n[Original]  : {args.text}")
            print(f"[Normalized]: {tok}")
        elif args.input and args.output:
            with open(args.input, "r", encoding="utf-8") as f:
                data = json.load(f)
            msgs = data.get("messages", []) if isinstance(data, dict) else data
            for m in msgs:
                m["text"] = classify_and_tokenize_urls(normalize_indic_text(m.get("text", "")))
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ Normalized {len(msgs)} messages. Saved to {args.output}")

    elif args.command == "dedup":
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        msgs = data.get("messages", []) if isinstance(data, dict) else data
        unique, removed = deduplicate_dataset(msgs, similarity_threshold=args.threshold)
        print(f"\n{'=' * 60}")
        print("DEDUPLICATION RESULTS")
        print(f"{'=' * 60}")
        print(f"Original Count : {len(msgs)}")
        print(f"Duplicates     : {removed} (removed)")
        print(f"Unique Count   : {len(unique)}")
        out_payload = {
            "total": len(unique),
            "duplicates_removed": removed,
            "messages": unique
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved clean dataset to: {args.output}")

    elif args.command == "align":
        aligned = align_external_dataset(args.source, args.input)
        out_payload = {
            "source": args.source,
            "total": len(aligned),
            "messages": aligned
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
        print(f"✅ Aligned {len(aligned)} messages from '{args.source}'. Saved to: {args.output}")

    elif args.command == "stats":
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        msgs = data.get("messages", []) if isinstance(data, dict) else data
        counts = Counter(m.get("label") for m in msgs)
        print(f"\n{'=' * 60}")
        print(f"DATASET METRICS: {args.input}")
        print(f"{'=' * 60}")
        print(f"Total Messages  : {len(msgs)}")
        print(f"  • Class 0 (Ham)        : {counts.get(0, 0)}")
        print(f"  • Class 1 (Promo)      : {counts.get(1, 0)}")
        print(f"  • Class 2 (Spam)       : {counts.get(2, 0)}")
        print(f"  • Class 3 (Scam)       : {counts.get(3, 0)}")
        print(f"  • Unlabeled (None)     : {counts.get(None, 0)}")


if __name__ == "__main__":
    main()
