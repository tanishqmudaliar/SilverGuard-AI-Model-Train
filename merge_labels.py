"""
Silver Guard — Merge Auto-labeled & Reviewed Data (4-Class Taxonomy)
--------------------------------------------------------------------
Merges auto-labeled messages (output.json) and manually reviewed messages (review.json)
into a unified training dataset (training_data.json).

Taxonomy (v2.0):
  0: Ham (Personal, Bank Notifications, OTP, Services)
  1: Promotional (Registered Brand Offers & Sales)
  2: Commercial Spam (Unsolicited 10-digit mobile telemarketing / UCC)
  3: Scam / Phishing (Cyber extortion, digital arrest, fake KYC, lottery, APK malware)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Silver Guard — Merge Auto-labeled & Reviewed Data into 4-Class Training Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="output.json", help="Path to auto-labeled JSON (default: output.json)")
    parser.add_argument("--review", default="review.json", help="Path to reviewed JSON (default: review.json)")
    parser.add_argument("--final", default="training_data.json", help="Output path for final merged JSON (default: training_data.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Verify input files
    if not Path(args.output).exists():
        print(f"❌ Error: Auto-labeled file '{args.output}' not found. Run label_sms.py first.")
        sys.exit(1)

    with open(args.output, "r", encoding="utf-8") as f:
        auto_data = json.load(f)
    auto_msgs: List[Dict[str, Any]] = auto_data.get("messages", [])

    review_msgs: List[Dict[str, Any]] = []
    if Path(args.review).exists():
        with open(args.review, "r", encoding="utf-8") as f:
            review_data = json.load(f)
        review_msgs = review_data.get("messages", [])
    else:
        print(f"ℹ️ Note: Review file '{args.review}' not found. Merging only auto-labeled messages.")

    # Triage review messages
    unlabeled = [m for m in review_msgs if m.get("label") is None]
    if unlabeled:
        print(f"⚠️ Warning: {len(unlabeled)} messages in '{args.review}' still have label=null.")
        print("   These will be SKIPPED. To include them, edit review.json and set labels first.")
        print(f"   First skipped: [{unlabeled[0].get('address')}] {str(unlabeled[0].get('text', ''))[:50]}...")

    labeled_review = [m for m in review_msgs if m.get("label") is not None]

    all_msgs: List[Dict[str, Any]] = []

    # 1. Ingest auto-labeled messages
    for m in auto_msgs:
        lbl = int(m["label"])
        all_msgs.append({
            "text": str(m["text"]),
            "address": str(m.get("address", "")),
            "label": lbl
        })

    # 2. Ingest reviewed messages with strict validation
    skipped_invalid = 0
    for m in labeled_review:
        try:
            lbl = int(m["label"])
            if lbl not in (0, 1, 2, 3):
                skipped_invalid += 1
                continue
            all_msgs.append({
                "text": str(m["text"]),
                "address": str(m.get("address", "")),
                "label": lbl
            })
        except (ValueError, TypeError):
            skipped_invalid += 1

    if skipped_invalid > 0:
        print(f"⚠️ Warning: Skipped {skipped_invalid} messages with invalid labels (labels must be 0, 1, 2, or 3).")

    # Compute per-class counts
    ham = sum(1 for m in all_msgs if m["label"] == 0)
    promo = sum(1 for m in all_msgs if m["label"] == 1)
    spam = sum(1 for m in all_msgs if m["label"] == 2)
    scam = sum(1 for m in all_msgs if m["label"] == 3)

    print(f"\n{'=' * 66}")
    print("FINAL 4-CLASS TRAINING DATA SUMMARY")
    print(f"{'=' * 66}")
    print(f"Total Combined Messages       : {len(all_msgs)}")
    print(f"  • Class 0 (Safe Ham)        : {ham}")
    print(f"  • Class 1 (Promotional)     : {promo}")
    print(f"  • Class 2 (Commercial Spam) : {spam}")
    print(f"  • Class 3 (Scam / Phishing) : {scam}")
    print(f"Skipped Unlabeled Messages    : {len(unlabeled)}")
    print(f"{'=' * 66}")

    final_payload = {
        "taxonomy_version": "2.0",
        "classes": {
            "0": "ham",
            "1": "promotional",
            "2": "commercial_spam",
            "3": "scam"
        },
        "total": len(all_msgs),
        "ham_count": ham,
        "promo_count": promo,
        "spam_count": spam,
        "scam_count": scam,
        "messages": all_msgs
    }

    with open(args.final, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)

    print(f"💾 Successfully saved final dataset to: {args.final}")
    print(f"\nUpload '{args.final}' to Google Colab or use with dataset_pipeline.py for training.")


if __name__ == "__main__":
    main()
