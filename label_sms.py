"""
Silver Guard — SMS Auto-Labeler for 4-Class Training Data
---------------------------------------------------------
Inputs:
  • Raw SMS export JSON (from Phone Link or SMS Backup & Restore)
  • Contacts CSV or VCF export (Google Contacts or phone vCard)

Outputs:
  • output.json: Messages auto-labeled into Ham (0) and Promotional (1)
  • review.json: Unknown phone numbers needing manual review (with smart heuristic hints)

Taxonomy (v2.0):
  0: Ham (Personal, Bank Transactional, Service, OTP, Govt)
  1: Promotional (Registered DLT Brand Marketing, Retail Offers)
  2: Commercial Spam (Unsolicited 10-digit mobile cold marketing / UCC)
  3: Scam / Phishing (Cyber extortion, digital arrest, fake KYC, lottery, APK malware)
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# TRAI DLT pattern: XX-YYYYYYZ or XX-YYYYYY
DLT_PATTERN = re.compile(r"^[A-Z]{2}-([A-Z0-9]{4,8})(-[STGP])?$", re.IGNORECASE)

# Well-known Indian commercial retail & e-commerce brands
RETAIL_PROMO_BRANDS = {
    "SMRTBR", "AJIOIN", "MAXFSN", "DOMINO", "TRENDS", "SWIGGY", "ZOMATO",
    "MYNTRA", "NYKAA", "FLPKRT", "RUTIME", "REGINF", "BIGBSK", "ZEPTO",
    "BLINKT", "KFCIND", "PIZZAH", "LENSKT", "PVRINO", "DECATH", "AMAZON",
    "TATACL", "PHARME", "APOLLO", "MEESHO", "SHOPCL", "PANTAL", "WESTSD",
    "RTRNDS"
}

# Well-known Indian banks, fintechs, government bodies, and service carriers
BANK_GOVT_SERVICES = {
    "SBINOT", "HDFCBK", "ICICIB", "AXISBK", "PNBSMS", "BOBSMS", "CANBKT",
    "UNIONT", "KOTAKT", "INDUSB", "YESBKT", "IDBIBK", "BKOINT", "CENTBK",
    "UIDAIG", "IRCTCG", "INCOTG", "EPFORG", "ISATHI", "AIRTEL", "JIOPAY",
    "BSNLSMS", "VIINFO", "PAYTM", "PHNPEP", "GPAYIN", "BHIMUP", "CREDIN"
}

# High-risk scam and threat keywords
SCAM_INDICATORS = [
    r"\bdigital\s+arrest\b",
    r"\barrest\s+warrant\b",
    r"\bcbi\b",
    r"\bmumbai\s+police\b",
    r"\bdelhi\s+police\b",
    r"\bcyber\s+crime\b",
    r"\bkyc\s+(update|pending|expired|blocked)\b",
    r"\bpan\s+(card\s+)?(link|update|blocked)\b",
    r"\belectricity\s+(power\s+)?(disconnect|cut)\b",
    r"\blucky\s+draw\b",
    r"\bpart[- ]time\s+job\b",
    r"\btelegram\s+task\b",
    r"\.apk\b"
]

# Promotional / Commercial keywords
PROMO_KEYWORDS = [
    r"\b\d+%\s+off\b",
    r"\bflat\s+\d+%\b",
    r"\bdiscount\b",
    r"\bbogo\b",
    r"\bbuy\s+1\s+get\s+1\b",
    r"\bcoupon\b",
    r"\bpromo\s*code\b",
    r"\bflash\s+sale\b",
    r"\bclearance\s+sale\b",
    r"\bexclusive\s+offer\b",
    r"\b(flats?|apartments?|plots?|bhk|villas?|real\s+estate|site\s+visit)\b",
    r"\b(pre[- ]?approved|personal\s+loan|home\s+loan|instant\s+loan)\b"
]


def normalize_phone(phone: Any) -> Optional[str]:
    """Normalize phone number to 10 digits."""
    if not phone:
        return None
    digits = re.sub(r"\D", "", str(phone))
    if len(digits) > 10 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) > 10 and digits.startswith("0"):
        digits = digits[1:]
    return digits[-10:] if len(digits) >= 10 else None


def load_contacts(contacts_path: Optional[str]) -> Dict[str, str]:
    """Loads phone numbers from Google Contacts CSV or phone VCF."""
    contacts: Dict[str, str] = {}
    if not contacts_path:
        return contacts

    path = Path(contacts_path)
    if not path.exists():
        return contacts

    try:
        if path.suffix.lower() == ".csv":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (
                        row.get("Name", "")
                        or row.get("First Name", "")
                        or row.get("Display Name", "")
                    )
                    for key in row:
                        if "phone" in key.lower() and row[key]:
                            norm = normalize_phone(row[key])
                            if norm:
                                contacts[norm] = name.strip() or "Known Contact"
        elif path.suffix.lower() == ".vcf":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                current_name = "Known Contact"
                for line in f:
                    line = line.strip()
                    if line.startswith("FN:"):
                        current_name = line[3:].strip()
                    elif line.startswith("TEL"):
                        phone = line.split(":")[-1]
                        norm = normalize_phone(phone)
                        if norm:
                            contacts[norm] = current_name
    except Exception as e:
        print(f"⚠️ Warning loading contacts ({contacts_path}): {e}")

    return contacts


def parse_sender_info(address: str) -> Dict[str, Any]:
    """Analyzes sender ID into telecom classification."""
    addr = str(address).strip().upper()
    if not addr:
        return {"type": "unknown", "is_dlt": False, "category": "unknown", "entity": ""}

    norm_phone = normalize_phone(addr)
    if norm_phone:
        return {"type": "phone", "is_dlt": False, "category": "phone_number", "entity": norm_phone}

    if re.match(r"^\d{3,6}$", addr):
        return {"type": "shortcode", "is_dlt": True, "category": "service", "entity": addr}

    m = DLT_PATTERN.match(addr)
    if m:
        entity = m.group(1).upper()
        suffix = m.group(2).upper().replace("-", "") if m.group(2) else ""

        if suffix == "P" or entity in RETAIL_PROMO_BRANDS:
            return {"type": "dlt", "is_dlt": True, "category": "promotional", "entity": entity}
        elif suffix in ("T", "S", "G") or entity in BANK_GOVT_SERVICES or any(
            b in entity for b in ("SBI", "HDFC", "ICICI", "AXIS", "JIO", "AIRTEL", "GOVT")
        ):
            return {"type": "dlt", "is_dlt": True, "category": "transactional", "entity": entity}
        else:
            return {"type": "dlt", "is_dlt": True, "category": "registered_dlt", "entity": entity}

    return {"type": "unknown", "is_dlt": False, "category": "unknown", "entity": addr}


def get_heuristic_review_hint(text: str, phone_freq: int) -> Tuple[Optional[int], str]:
    """Generates a suggested label and human-readable hint for manual review."""
    lower = text.lower()

    if any(re.search(pat, lower) for pat in SCAM_INDICATORS):
        return 3, "threat_indicators -> likely 3 (scam)"

    if any(re.search(pat, lower) for pat in PROMO_KEYWORDS):
        return 2, "commercial_offer -> likely 2 (spam) or 1 (promo)"

    if phone_freq >= 3:
        return 0, "frequent_sender -> likely 0 (ham)"

    return None, "unrecognized -> review manually (0=ham, 1=promo, 2=spam, 3=scam)"


def find_default_path(candidates: List[str]) -> Optional[str]:
    """Returns the first existing file from a list of candidate paths."""
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Silver Guard — SMS Auto-Labeler for 4-Class Training Data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default=None, help="Path to raw SMS JSON export (default: auto-detect)")
    parser.add_argument("--contacts", default=None, help="Path to contacts.csv or contacts.vcf (default: auto-detect)")
    parser.add_argument("--output", default="output.json", help="Path for auto-labeled JSON output (default: output.json)")
    parser.add_argument("--review", default="review.json", help="Path for manual review JSON output (default: review.json)")
    args = parser.parse_args()

    # Auto-detect input file
    input_path = args.input or find_default_path(["data.json", "data/data.json", "data/rawdata.json"])
    if not input_path or not Path(input_path).exists():
        print("❌ Error: No input SMS file found. Place 'data.json' in current directory or specify --input.")
        return

    # Auto-detect contacts file
    contacts_path = args.contacts or find_default_path(["contacts.csv", "data/contacts.csv", "contacts.vcf", "data/contacts.vcf"])

    contacts = load_contacts(contacts_path) if contacts_path else {}
    if contacts:
        print(f"📖 Loaded {len(contacts)} contacts from {contacts_path}")
    else:
        print("ℹ️ No contacts file loaded (pass --contacts to auto-whitelist known numbers)")

    print(f"📥 Loading SMS data from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "messages" in raw:
        messages = raw["messages"]
    elif isinstance(raw, list):
        messages = raw
    else:
        print("❌ Error: Unrecognized JSON structure (expected list or object with 'messages' key)")
        return

    print(f"Loaded {len(messages)} total messages.")
    received = [m for m in messages if m.get("kind") != "sent"]
    print(f"Filtered to {len(received)} received messages (excluded sent).")

    # Track phone frequency
    phone_counts = Counter(
        normalize_phone(m.get("address", ""))
        for m in received
        if normalize_phone(m.get("address", ""))
    )

    auto_labeled: List[Dict[str, Any]] = []
    needs_review: List[Dict[str, Any]] = []

    for msg in received:
        addr = str(msg.get("address", "")).strip()
        text = str(msg.get("text", "")).strip()
        norm_phone = normalize_phone(addr)

        sender_info = parse_sender_info(addr)

        # 1. Contact Whitelist -> Class 0 (Ham)
        if norm_phone and norm_phone in contacts:
            auto_labeled.append({
                "text": text,
                "address": addr,
                "label": 0,
                "category_name": "ham",
                "auto_reason": f"contact:{contacts[norm_phone]}"
            })
        # 2. Registered DLT Promotional -> Class 1 (Promotional)
        elif sender_info["is_dlt"] and sender_info["category"] == "promotional":
            auto_labeled.append({
                "text": text,
                "address": addr,
                "label": 1,
                "category_name": "promotional",
                "auto_reason": f"dlt_promo:{sender_info['entity']}"
            })
        # 3. Registered DLT Bank / Govt / Service / Shortcode -> Class 0 (Ham)
        elif sender_info["is_dlt"] and sender_info["category"] in ("transactional", "service", "registered_dlt"):
            auto_labeled.append({
                "text": text,
                "address": addr,
                "label": 0,
                "category_name": "ham",
                "auto_reason": f"dlt_service:{sender_info['entity']}"
            })
        # 4. Unknown Phone Number or Unregistered Sender -> Needs Review
        else:
            freq = phone_counts.get(norm_phone, 1) if norm_phone else 0
            suggested_label, hint = get_heuristic_review_hint(text, freq)
            needs_review.append({
                "text": text,
                "address": addr,
                "label": None,
                "suggested_label": suggested_label,
                "contact_frequency": freq,
                "hint": hint
            })

    # Sort review items: threats first, then suggested spam, then unknown, then frequent contacts
    needs_review.sort(key=lambda x: (
        0 if x["suggested_label"] == 3 else
        1 if x["suggested_label"] == 2 else
        3 if x["suggested_label"] == 0 else 2,
        -x.get("contact_frequency", 0)
    ))

    # Calculate statistics
    auto_ham = sum(1 for m in auto_labeled if m["label"] == 0)
    auto_promo = sum(1 for m in auto_labeled if m["label"] == 1)
    rev_threats = sum(1 for m in needs_review if m["suggested_label"] == 3)
    rev_spam = sum(1 for m in needs_review if m["suggested_label"] == 2)
    rev_ham = sum(1 for m in needs_review if m["suggested_label"] == 0)

    print(f"\n{'=' * 66}")
    print("AUTO-LABELING SUMMARY (4-CLASS TAXONOMY)")
    print(f"{'=' * 66}")
    print(f"Auto-labeled Total           : {len(auto_labeled)}")
    print(f"  • Safe Ham (0)             : {auto_ham}")
    print(f"  • Registered Promo (1)     : {auto_promo}")
    print(f"Needs Manual Review          : {len(needs_review)}")
    print(f"  • Suggested Scam (3)       : {rev_threats}")
    print(f"  • Suggested Spam (2)       : {rev_spam}")
    print(f"  • Suggested Ham (0)        : {rev_ham}")
    print(f"  • Other / Unclassified     : {len(needs_review) - (rev_threats + rev_spam + rev_ham)}")

    # Save output.json
    output_data = {
        "taxonomy_version": "2.0",
        "classes": {
            "0": "ham",
            "1": "promotional",
            "2": "commercial_spam",
            "3": "scam"
        },
        "auto_labeled_count": len(auto_labeled),
        "needs_review_count": len(needs_review),
        "messages": auto_labeled
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved auto-labeled dataset to : {args.output}")

    # Save review.json
    review_data = {
        "instructions": (
            "Set 'label' for each message according to the 4-Class Taxonomy:\n"
            "  0 = Ham (personal, family, friends, bank notifications, OTP)\n"
            "  1 = Promotional (legitimate brand sales, discount coupons)\n"
            "  2 = Commercial Spam (unsolicited 10-digit telemarketing, loans, real estate)\n"
            "  3 = Scam / Phishing (cyber extortion, digital arrest, fake KYC, lottery, APKs)\n"
            "Tip: You may accept 'suggested_label' by copying it to 'label'."
        ),
        "total": len(needs_review),
        "messages": needs_review
    }
    with open(args.review, "w", encoding="utf-8") as f:
        json.dump(review_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved review list to          : {args.review}")

    print(f"\n{'=' * 66}")
    print("NEXT STEPS")
    print(f"{'=' * 66}")
    print(f"1. Open '{args.review}' and set 'label' to 0, 1, 2, or 3.")
    print("2. Run 'python merge_labels.py' to produce 'training_data.json'.")


if __name__ == "__main__":
    main()
