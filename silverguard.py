"""
Silver Guard — Indian Multilingual SMS Scam & Threat Classifier
================================================================
Production-ready cascaded inference engine for Indian telecommunications.

Features:
  • 3-Stage Cascaded Architecture (Deterministic Gateway -> ML Inference -> Threat Heuristics)
  • Full 4-Class Taxonomy Support (Ham, Promotional, Commercial Spam, Scam/Phishing)
  • Indic-Safe Normalization (preserves Hindi/Marathi/Tamil/Telugu matras, viramas & emojis)
  • Comprehensive TRAI DLT Header Engine (supports 6-char alphanumeric headers, suffixes & shortcodes)
  • Contact Book Whitelisting (instant zero-compute HAM verification)
  • Structural Risk Scorer (URL shorteners, APK downloads, callback phones, extortion triggers)
  • Dual Score Output (Threat Score & Commercial Score)

Usage:
  Interactive CLI:
    python silverguard.py
    python silverguard.py --contacts data/contacts.csv

  Single Message Evaluation:
    python silverguard.py --header "JK-SMRTBR" --message "50% OFF on all groceries today only!"
    python silverguard.py --header "+919876543210" --message "CBI NOTICE: Arrest warrant issued. Call now." --json
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import zlib

import numpy as np
import onnxruntime as ort

# Fix emoji printing on Windows terminals (cp1252 etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Indic-Safe Text Normalization & Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def is_indic_mark(ch: str) -> bool:
    """Checks if character is a dependent vowel sign or virama in South Asian scripts."""
    cp = ord(ch)
    # Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
    return 0x0900 <= cp <= 0x0D7F


def clean_and_normalize_multilingual(text: str) -> str:
    """
    Normalizes multilingual text canonically (NFC) while:
      1. Strictly preserving Indic vowel signs (matras), viramas, and nuktas.
      2. Separating emojis and symbols from adjacent words to prevent [UNK] collapses.
      3. Separating Indic punctuation (danda / double danda).
    """
    if not text:
        return ""

    # NFC composition retains precomposed Indic characters
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
# 2. Comprehensive TRAI DLT Header & Sender Analyzer
# ─────────────────────────────────────────────────────────────────────────────

DLT_PATTERN = re.compile(r"^[A-Z]{2}-([A-Z0-9]{4,8})(-[STGP])?$", re.IGNORECASE)

# Well-known Indian commercial retail & e-commerce brands
RETAIL_PROMO_BRANDS = {
    "SMRTBR", "AJIOIN", "MAXFSN", "DOMINO", "TRENDS", "SWIGGY", "ZOMATO",
    "MYNTRA", "NYKAA", "FLPKRT", "RUTIME", "REGINF", "BIGBSK", "ZEPTO",
    "BLINKT", "KFCIND", "PIZZAH", "LENSKT", "PVRINO", "DECATH", "AMAZON",
    "TATACL", "PHARME", "APOLLO", "MEESHO", "SHOPCL", "PANTAL", "WESTSD",
    "RTRNDS", "SAMSNG", "LNKART", "METLAB", "THEBWC", "HKMIND"
}

# Well-known Indian banks, fintechs, government bodies, and service carriers
BANK_GOVT_SERVICES = {
    "SBINOT", "HDFCBK", "ICICIB", "AXISBK", "PNBSMS", "BOBSMS", "CANBKT",
    "UNIONT", "KOTAKT", "INDUSB", "YESBKT", "IDBIBK", "BKOINT", "CENTBK",
    "UIDAIG", "IRCTCG", "INCOTG", "EPFORG", "ISATHI", "AIRTEL", "JIOPAY",
    "BSNLSMS", "VIINFO", "PAYTM", "PHNPEP", "GPAYIN", "BHIMUP", "CREDIN",
    "KOTAKB", "YESBNK", "VPAYTM"
}


def normalize_phone(phone: Any) -> Optional[str]:
    """Extracts standard 10-digit Indian phone number."""
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
    if not contacts_path or not os.path.exists(contacts_path):
        return contacts

    path = Path(contacts_path)
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


def parse_header_metadata(header: str) -> Dict[str, Any]:
    """
    Parses sender ID into structured telecom metadata according to TRAI TCCCPR standards.
    """
    h = str(header).strip().upper()
    if not h:
        return {
            "category": "personal",
            "is_dlt": False,
            "description": "personal / no sender ID",
            "emoji": "👤",
            "entity": "",
            "is_trusted": False
        }

    # 10-digit Indian mobile number
    norm_phone = normalize_phone(h)
    if norm_phone:
        return {
            "category": "phone_number",
            "is_dlt": False,
            "description": f"unverified mobile (+91 {norm_phone[:5]} {norm_phone[5:]}) ⚠️",
            "emoji": "📵",
            "entity": norm_phone,
            "is_trusted": False
        }

    # Telecom shortcodes (e.g. 1909, 56767, 1900)
    if re.match(r"^\d{3,6}$", h):
        return {
            "category": "shortcode",
            "is_dlt": True,
            "description": f"telecom shortcode ({h}) ✅",
            "emoji": "🔢",
            "entity": h,
            "is_trusted": True
        }

    # TRAI DLT pattern: XX-YYYYYYZ or XX-YYYYYY
    m = DLT_PATTERN.match(h)
    if m:
        entity = m.group(1).upper()
        suffix = m.group(2).upper().replace("-", "") if m.group(2) else ""

        if suffix == "T":
            return {
                "category": "transactional",
                "is_dlt": True,
                "description": f"transactional / bank ({entity}) ✅",
                "emoji": "🏦",
                "entity": entity,
                "is_trusted": True
            }
        elif suffix == "S":
            return {
                "category": "service",
                "is_dlt": True,
                "description": f"service / OTP alert ({entity}) ✅",
                "emoji": "⚙️",
                "entity": entity,
                "is_trusted": True
            }
        elif suffix == "G":
            return {
                "category": "government",
                "is_dlt": True,
                "description": f"government entity ({entity}) ✅",
                "emoji": "🏛️",
                "entity": entity,
                "is_trusted": True
            }
        elif suffix == "P":
            return {
                "category": "promotional",
                "is_dlt": True,
                "description": f"promotional offer ({entity}) 📢",
                "emoji": "📢",
                "entity": entity,
                "is_trusted": True
            }

        # Check known retail brands
        if entity in RETAIL_PROMO_BRANDS or any(
            b in entity for b in ("SMRTBR", "AJIO", "MAXFS", "DOMINO", "TREND", "SWIGGY", "ZOMATO", "MYNTRA", "NYKAA", "FLPKRT", "LNKART", "LENSK", "SAMSNG", "METLAB", "THEBWC", "RUTIME")
        ):
            return {
                "category": "promotional",
                "is_dlt": True,
                "description": f"{entity} retail marketing 📢",
                "emoji": "📢",
                "entity": entity,
                "is_trusted": True
            }

        # Check known banks/services
        if entity in BANK_GOVT_SERVICES or any(
            b in entity for b in ("SBI", "HDFC", "ICICI", "AXIS", "JIO", "AIRTEL", "GOVT", "KOTAK", "YESB", "PAYTM", "PNB", "BOB")
        ):
            return {
                "category": "transactional",
                "is_dlt": True,
                "description": f"{entity} verified service/bank ✅",
                "emoji": "🏦",
                "entity": entity,
                "is_trusted": True
            }

        return {
            "category": "registered_dlt",
            "is_dlt": True,
            "description": f"{entity} registered DLT ✅",
            "emoji": "✅",
            "entity": entity,
            "is_trusted": True
        }

    return {
        "category": "suspicious",
        "is_dlt": False,
        "description": "unregistered / suspicious sender ⚠️",
        "emoji": "⚠️",
        "entity": h,
        "is_trusted": False
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Structural Threat & Link Heuristic Engine
# ─────────────────────────────────────────────────────────────────────────────

# Known link shorteners heavily abused by smishers
SUSPICIOUS_SHORTENERS = {
    "bit.ly", "tinyurl.com", "is.gd", "rb.gy", "cutt.ly", "shorturl.at",
    "t.co", "ow.ly", "buff.ly", "goo.gl", "bit.do"
}

# High-urgency intimidation triggers typical of cyber extortion & digital arrest
EXTORTION_KEYWORDS = [
    r"\bdigital\s+arrest\b",
    r"\barrest\s+warrant\b",
    r"\bmumbai\s+police\b",
    r"\bdelhi\s+police\b",
    r"\bcyber\s+crime\b",
    r"\bcbi\b",
    r"\bcustoms\s+department\b",
    r"\benforcement\s+directorate\b",
    r"\bmoney\s+laundering\b",
    r"\bpolice\s+notice\b",
    r"\blaw\s+enforcement\b",
    r"\bcourt\s+notice\b"
]

# Account freeze / KYC deactivation triggers
KYC_PHISHING_KEYWORDS = [
    r"\baccount\s+(blocked|frozen|deactivated|suspended)\b",
    r"\bkyc\s+(update|pending|expired|verification)\b",
    r"\bpan\s+(card\s+)?(link|update|blocked)\b",
    r"\baadhaar\s+(link|update|suspend)\b",
    r"\belectricity\s+(power\s+)?(disconnect|cut)\b",
    r"\bbill\s+(due\s+)?tonight\b",
    r"\bsbi\s+netbanking\b",
    r"\bsim\s+(block|deactivate)\b"
]

# Advance-fee lottery / task fraud triggers
FRAUD_LURE_KEYWORDS = [
    r"\blucky\s+draw\b",
    r"\bwon\s+(rs\.?|inr|₹)\b",
    r"\bkbc\b",
    r"\bpart[- ]time\s+job\b",
    r"\bdaily\s+(income|earning)\b",
    r"\brate\s+hotels\b",
    r"\byoutube\s+like\b",
    r"\btelegram\s+task\b"
]

# Retail & commercial promotional keywords (including real estate & financial leads)
PROMOTIONAL_KEYWORDS = [
    r"\b\d+%\s+off\b",
    r"\bflat\s+\d+%\b",
    r"\bdiscount\b",
    r"\bbogo\b",
    r"\bbuy\s+1\s+get\s+1\b",
    r"\bcoupon\b",
    r"\bpromo\s*code\b",
    r"\bflash\s+sale\b",
    r"\bend\s+of\s+season\b",
    r"\bclearance\s+sale\b",
    r"\bfree\s+delivery\b",
    r"\bcashback\b",
    r"\bexclusive\s+offer\b",
    r"\bsave\s+upto\b",
    r"\b(flats?|apartments?|plots?|bhk|villas?|real\s+estate|site\s+visit|stamp\s+duty)\b",
    r"\b(pre[- ]?approved|personal\s+loan|home\s+loan|credit\s+card|car\s+loan|instant\s+loan|zero\s+down\s+payment)\b",
    r"\b(insurance|premium|policy|mutual\s+fund|demat|stock\s+tips)\b",
    r"\b(admissions?\s+open|coaching|training\s+course)\b"
]


def extract_structural_features(header: str, text: str) -> Dict[str, Any]:
    """
    Extracts evidentiary threat and promotional features from the message.
    """
    features: Dict[str, Any] = {
        "urls": [],
        "has_shortener": False,
        "has_ip_url": False,
        "has_apk_link": False,
        "callback_phone": None,
        "is_extortion": False,
        "is_kyc_phishing": False,
        "is_fraud_lure": False,
        "is_promotional_content": False,
        "indicators": []
    }

    lower_text = text.lower()

    # URL extraction
    url_pattern = re.compile(
        r"(https?://[^\s/$.?#].[^\s]*|www\.[^\s/$.?#].[^\s]*|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)",
        re.IGNORECASE
    )
    raw_urls = url_pattern.findall(text)
    # Filter out common false matches like names, currency, or acronyms
    filtered_urls = [u for u in raw_urls if "." in u and not u.endswith(".") and len(u) > 4]
    features["urls"] = filtered_urls

    for url in filtered_urls:
        url_lower = url.lower()
        if any(shortener in url_lower for shortener in SUSPICIOUS_SHORTENERS):
            features["has_shortener"] = True
            features["indicators"].append("Shortened URL (obfuscated destination)")
        if re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", url):
            features["has_ip_url"] = True
            features["indicators"].append("Direct IP URL (bypasses domain registration)")
        if ".apk" in url_lower or "download" in url_lower:
            features["has_apk_link"] = True
            features["indicators"].append("Direct Android APK download link")

    # Embedded callback phone numbers in the message body
    phone_match = re.search(r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b", text)
    if phone_match and phone_match.group(0).replace(" ", "").replace("-", "") != header.replace(" ", ""):
        features["callback_phone"] = phone_match.group(0)
        features["indicators"].append(f"Embedded callback phone: {features['callback_phone']}")

    # Content intent regex matches
    if any(re.search(pat, lower_text) for pat in EXTORTION_KEYWORDS):
        features["is_extortion"] = True
        features["indicators"].append("Intimidation / Digital Arrest / Law enforcement claim")

    if any(re.search(pat, lower_text) for pat in KYC_PHISHING_KEYWORDS):
        features["is_kyc_phishing"] = True
        features["indicators"].append("Account freeze / KYC suspension threat")

    if any(re.search(pat, lower_text) for pat in FRAUD_LURE_KEYWORDS):
        features["is_fraud_lure"] = True
        features["indicators"].append("Lottery / Part-time task fraud lure")

    if any(re.search(pat, lower_text) for pat in PROMOTIONAL_KEYWORDS):
        features["is_promotional_content"] = True

    return features


FASTTEXT_VOCAB_SIZE = 60000


def get_char_ngrams(text: str, min_n: int = 3, max_n: int = 5, vocab_size: int = FASTTEXT_VOCAB_SIZE) -> List[int]:
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
                h = (zlib.crc32(sub.encode("utf-8")) % (vocab_size - 1)) + 1
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
# 4. Cascaded 3-Stage Classification Engine
# ─────────────────────────────────────────────────────────────────────────────

def classify_sms(
    session: ort.InferenceSession,
    header: str = "",
    message: str = "",
    contacts: Optional[Dict[str, str]] = None,
    tokenizer: Any = None,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Executes the 3-Stage Cascaded Architecture:
      Stage 1: Zero-compute deterministic gateway (contacts & verified DLT).
      Stage 2: Multilingual neural inference (FastText character subwords + 12 domain features).
      Stage 3: Evidentiary risk & promotional synthesis.
    """
    contacts = contacts or {}
    hdr_meta = parse_header_metadata(header)
    clean_msg = clean_and_normalize_multilingual(message)
    features = extract_structural_features(header, clean_msg)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: DETERMINISTIC GATEWAY (< 0.2 ms)
    # ══════════════════════════════════════════════════════════════════════════

    # 1. Address Book Whitelist Match
    norm_phone = normalize_phone(header)
    if norm_phone and norm_phone in contacts:
        contact_name = contacts[norm_phone]
        return {
            "verdict": "✅ SAFE (HAM)",
            "threat_score": 0.0000,
            "commercial_score": 0.0000,
            "category": "ham_contact",
            "header_info": hdr_meta,
            "indicators": [f"Known Contact: {contact_name}"],
            "explanation": f"Message originates from verified contact '{contact_name}'. Zero threat detected.",
            "stage": "Stage 1 (Contact Gateway)"
        }

    # 2. Verified DLT Transactional / Government / Service Gateway
    # If sent from verified bank/govt/service DLT AND contains no malicious APKs / IP urls
    if hdr_meta["is_dlt"] and hdr_meta["category"] in ("transactional", "service", "government", "shortcode"):
        if not features["has_apk_link"] and not features["has_ip_url"] and not features["is_extortion"]:
            return {
                "verdict": "✅ SAFE (HAM)",
                "threat_score": 0.0010,
                "commercial_score": 0.0500,
                "category": "ham_transactional",
                "header_info": hdr_meta,
                "indicators": [f"Verified TRAI DLT Sender ({hdr_meta['entity']})"],
                "explanation": f"Sent from registered DLT entity '{hdr_meta['entity']}' fulfilling transactional or service notifications.",
                "stage": "Stage 1 (DLT Gateway)"
            }

    # 3. Verified DLT Promotional Gateway
    # If sent from registered DLT promotional sender (-P or known retail brand)
    if hdr_meta["is_dlt"] and hdr_meta["category"] == "promotional":
        if not features["has_apk_link"] and not features["has_ip_url"] and not features["is_extortion"] and not features["is_kyc_phishing"]:
            return {
                "verdict": "📢 PROMOTIONAL (MARKETING / OFFER)",
                "threat_score": 0.0150,
                "commercial_score": 0.9800,
                "category": "promotional",
                "header_info": hdr_meta,
                "indicators": [f"Registered Retail DLT Sender ({hdr_meta['entity']})", "Commercial advertisement / offer"],
                "explanation": f"Legitimate retail marketing offer from registered entity '{hdr_meta['entity']}'. Not a security scam.",
                "stage": "Stage 1 (DLT Promo Gateway)"
            }

    # 4. Registered DLT General Enterprise Gateway
    # If sent from any registered DLT enterprise without overt cyber malware
    if hdr_meta["is_dlt"] and hdr_meta["category"] == "registered_dlt":
        if not features["has_apk_link"] and not features["has_ip_url"] and not features["is_extortion"] and not features["is_kyc_phishing"]:
            if features["is_promotional_content"]:
                return {
                    "verdict": "📢 PROMOTIONAL (MARKETING / OFFER)",
                    "threat_score": 0.0150,
                    "commercial_score": 0.9500,
                    "category": "promotional",
                    "header_info": hdr_meta,
                    "indicators": [f"Registered Enterprise DLT Sender ({hdr_meta['entity']})", "Commercial promotion"],
                    "explanation": f"Marketing notification from registered corporate sender '{hdr_meta['entity']}'.",
                    "stage": "Stage 1 (DLT General Promo Gateway)"
                }
            else:
                return {
                    "verdict": "✅ SAFE (HAM)",
                    "threat_score": 0.0050,
                    "commercial_score": 0.1000,
                    "category": "ham_registered_dlt",
                    "header_info": hdr_meta,
                    "indicators": [f"Registered Enterprise DLT Sender ({hdr_meta['entity']})"],
                    "explanation": f"Legitimate communication from registered corporate sender '{hdr_meta['entity']}'.",
                    "stage": "Stage 1 (DLT General Gateway)"
                }

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: MULTILINGUAL NEURAL INFERENCE (~1.2 ms on edge)
    # ══════════════════════════════════════════════════════════════════════════

    sample_ngrams = np.array(get_char_ngrams(clean_msg), dtype=np.int64)
    sample_offsets = np.array([0], dtype=np.int64)
    sample_tabular = np.array([extract_domain_features(header, clean_msg)], dtype=np.float32)

    raw_output = session.run(None, {
        "ngram_ids": sample_ngrams,
        "offsets": sample_offsets,
        "tabular": sample_tabular
    })
    logits = raw_output[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    base_ham_score = float(probs[0])
    base_promo_score = float(probs[1])
    base_spam_score = float(probs[2])
    base_threat_score = float(probs[3])
    base_commercial_score = base_promo_score + base_spam_score

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3: EVIDENTIARY RISK & COMMERCIALITY SYNTHESIS (< 1.0 ms)
    # ══════════════════════════════════════════════════════════════════════════

    # Calculate calibrated threat score and commercial score
    threat_score = base_threat_score
    commercial_score = base_commercial_score

    # Adjust for DLT header credibility
    if hdr_meta["is_dlt"]:
        if hdr_meta["category"] == "promotional":
            # If DLT registered promo, heavily penalize false scam inflation unless overt malware
            if not features["has_apk_link"] and not features["is_extortion"]:
                threat_score = min(threat_score, 0.15)
                commercial_score = max(commercial_score, 0.95)
        elif hdr_meta["category"] in ("transactional", "service", "government"):
            if not features["has_apk_link"]:
                threat_score = min(threat_score, 0.10)

    # Elevate threat score if high-risk attack indicators are detected from unverified senders
    if not hdr_meta["is_trusted"]:
        if features["is_extortion"]:
            threat_score = max(threat_score, 0.95)
        if features["is_kyc_phishing"] and (features["has_shortener"] or features["has_ip_url"] or features["has_apk_link"]):
            threat_score = max(threat_score, 0.96)
        if features["has_apk_link"]:
            threat_score = max(threat_score, 0.98)
        if features["is_fraud_lure"] and features["has_shortener"]:
            threat_score = max(threat_score, 0.90)

        # Dampen threat score for ordinary unsolicited marketing/cold outreach without malware/extortion
        if features["is_promotional_content"]:
            if not (features["is_extortion"] or features["is_kyc_phishing"] or features["has_apk_link"] or features["has_ip_url"] or features["is_fraud_lure"]):
                threat_score = min(threat_score, 0.30)
                commercial_score = max(commercial_score, 0.88)

    # Determine final verdict based on 4-Class Taxonomy
    if threat_score >= 0.75:
        verdict = "🚨 HIGH RISK SCAM / PHISHING"
        category = "scam"
        explanation = "Strong malicious indicators detected: unverified sender soliciting sensitive credentials, intimidation, or malware."
    elif commercial_score >= 0.70:
        if hdr_meta["is_dlt"]:
            verdict = "📢 PROMOTIONAL (MARKETING / OFFER)"
            category = "promotional"
            explanation = "Commercial marketing advertisement from registered sender. No malicious threats detected."
        else:
            verdict = "📵 UNSOLICITED SPAM (UCC)"
            category = "commercial_spam"
            explanation = "Unsolicited promotional marketing sent from an unregistered mobile number (Do-Not-Disturb violation)."
    elif threat_score >= 0.40:
        verdict = "🟡 BORDERLINE (EXERCISE CAUTION)"
        category = "borderline"
        explanation = "Suspicious message structure from unverified sender. Exercise caution before opening links or calling numbers."
    else:
        verdict = "✅ SAFE (HAM)"
        category = "ham"
        explanation = "Message appears legitimate. No security threats or unsolicited bulk spam detected."

    return {
        "verdict": verdict,
        "threat_score": float(np.clip(threat_score, 0.0, 1.0)),
        "commercial_score": float(np.clip(commercial_score, 0.0, 1.0)),
        "category": category,
        "header_info": hdr_meta,
        "indicators": features["indicators"],
        "explanation": explanation,
        "stage": "Stage 2 & 3 (ML + Heuristic Reasoner)"
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI Pretty-Printing & Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def print_result_card(res: Dict[str, Any]) -> None:
    """Renders a visual terminal report card for classified SMS."""
    threat = res["threat_score"]
    commercial = res["commercial_score"]
    hdr = res["header_info"]

    t_filled = round(threat * 20)
    t_bar = "█" * t_filled + "░" * (20 - t_filled)

    c_filled = round(commercial * 20)
    c_bar = "█" * c_filled + "░" * (20 - c_filled)

    print()
    print("┌" + "─" * 66 + "┐")
    print(f"│  Verdict         : {res['verdict']:<45}│")
    print(f"│  Threat Score    : {threat:.4f}  [{t_bar}]  │")
    print(f"│  Commerciality   : {commercial:.4f}  [{c_bar}]  │")
    print("├" + "─" * 66 + "┤")
    print(f"│  Sender Identity : {(hdr['emoji'] + '  ' + hdr['description']):<45}│")
    print(f"│  Decision Stage  : {res.get('stage', 'Pipeline'):<45}│")
    if res["indicators"]:
        print("├" + "─" * 66 + "┤")
        print("│  Detected Indicators:                                           │")
        for ind in res["indicators"][:3]:
            print(f"│   • {ind:<60}│")
    print("├" + "─" * 66 + "┤")
    print(f"│  Explanation     : {res['explanation'][:45]:<45}│")
    if len(res["explanation"]) > 45:
        print(f"│                    {res['explanation'][45:90]:<45}│")
    print("└" + "─" * 66 + "┘")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Silver Guard — Indian Multilingual SMS Scam & Threat Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive Mode:
    python silverguard.py

  Single Message Check:
    python silverguard.py --header "JK-SMRTBR" --message "SMART Bazaar Full Paisa Vasool Sale 50% off!"
    python silverguard.py --header "+919876543210" --message "This is CBI. Arrest warrant issued. Call now."
    python silverguard.py --header "AD-HDFCBK-T" --message "Rs 1,500 debited from A/c **4120. -HDFC" --json
        """
    )
    default_model = "model/silverguard_fasttext_fusion.onnx"
    p.add_argument("--model", default=default_model, help=f"Path to ONNX model file (default: {default_model})")
    p.add_argument("--contacts", default="data/contacts.csv", help="Path to contacts.csv or contacts.vcf (default: data/contacts.csv)")
    p.add_argument("--header", default="", help="Sender ID (DLT header or phone number)")
    p.add_argument("--message", default="", help="SMS text body to evaluate non-interactively")
    p.add_argument("--json", action="store_true", help="Output pure JSON format")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate model file
    if not os.path.exists(args.model):
        if args.json:
            print(json.dumps({"error": f"File not found: {args.model}"}))
        else:
            print(f"❌ File not found: {args.model}")
        sys.exit(1)

    try:
        session = ort.InferenceSession(args.model)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": f"Could not load ONNX model: {e}"}))
        else:
            print(f"❌ Could not load ONNX model:\n    {e}")
        sys.exit(1)

    contacts = load_contacts(args.contacts)

    # Non-interactive CLI mode
    if args.message:
        res = classify_sms(
            session=session,
            header=args.header,
            message=args.message,
            contacts=contacts
        )
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            print_result_card(res)
        return

    # Interactive REPL mode
    print(f"Loading Silver Guard ({args.model})...")
    print(f"Loaded {len(contacts)} contacts for whitelisting.")
    print("✅ Ready.\n")
    print("─" * 66)
    print("  Input format :  HEADER [SEP] message body")
    print("  Example DLT  :  JK-SMRTBR [SEP] 50% OFF on all groceries!")
    print("  Example Scam :  +919876543210 [SEP] Your Aadhaar is blocked by CBI. Call now!")
    print("  No Sender?   :  Just type the message directly.")
    print("  Quit         :  type 'quit' or 'exit'")
    print("─" * 66)

    while True:
        try:
            raw = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if "[SEP]" in raw:
            parts = raw.split("[SEP]", maxsplit=1)
            header = parts[0].strip()
            message = parts[1].strip()
        else:
            header = ""
            message = raw

        if not message:
            print("  ⚠️ Please include a message body.")
            continue

        res = classify_sms(
            session=session,
            header=header,
            message=message,
            contacts=contacts
        )
        print_result_card(res)


if __name__ == "__main__":
    main()
