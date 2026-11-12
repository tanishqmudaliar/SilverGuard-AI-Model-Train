"""
Silver Guard — Unified 4-Class Indic Dataset Builder
----------------------------------------------------
Compiles, normalizes, sanitizes PII, and deduplicates multi-source datasets:
  1. Atherion 30K (Kaggle: atherionsystems/indian-scam-sms-30k)
  2. ScamShield Smishing (Kaggle: siddharthrane07/scamshield-multilingual-smishing-social-engn)
  3. Personal Indian SMS Export (data/rawdata.json)
  4. JuniorAlive India SMS (data/external/junioralive/spam_ham_india.csv)

Output:
  data/silverguard_indic_unified.json (Clean 4-Class balanced training dataset)
"""

import csv
import json
import os
import re
import random
from pathlib import Path
from collections import Counter

from dataset_pipeline import redact_pii, classify_and_tokenize_urls, normalize_indic_text, deduplicate_dataset
from label_sms import parse_sender_info, normalize_phone, load_contacts, PROMO_KEYWORDS

def build_dataset(output_path="data/silverguard_indic_unified.json", max_scam=8000, max_ham=8000):
    print("=" * 70)
    print("SILVER GUARD — BUILDING UNIFIED 4-CLASS INDIC DATASET")
    print("=" * 70)

    dataset = []

    # 1. Ingest Personal Messages (Real-world Indian ground truth)
    rawdata_path = Path("data/rawdata.json")
    contacts_path = Path("data/contacts.csv")
    contacts = load_contacts(str(contacts_path)) if contacts_path.exists() else {}

    if rawdata_path.exists():
        with open(rawdata_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        msgs = raw.get("messages", []) if isinstance(raw, dict) else raw
        print(f"Loading personal messages: {len(msgs)} total...")

        personal_ham = 0
        personal_promo = 0
        for m in msgs:
            if m.get("kind") == "sent":
                continue
            addr = str(m.get("address", "")).strip()
            text = str(m.get("text", "")).strip()
            norm = normalize_phone(addr)
            info = parse_sender_info(addr)

            if norm and norm in contacts:
                dataset.append({"text": text, "address": addr, "label": 0, "source": "personal_contact"})
                personal_ham += 1
            elif not info["is_dlt"]:
                if any(re.search(pat, text.lower()) for pat in PROMO_KEYWORDS):
                    dataset.append({"text": text, "address": addr, "label": 2, "source": "personal_ucc_spam"})
                else:
                    dataset.append({"text": text, "address": addr, "label": 0, "source": "personal_phone_ham"})
                    personal_ham += 1
            elif info["is_dlt"] and info["category"] == "promotional":
                dataset.append({"text": text, "address": addr, "label": 1, "source": "personal_dlt_promo"})
                personal_promo += 1
            elif info["is_dlt"] and info["category"] in ("transactional", "service", "registered_dlt"):
                dataset.append({"text": text, "address": addr, "label": 0, "source": "personal_dlt_ham"})
                personal_ham += 1

        print(f"  • Ingested {personal_ham} Safe Ham (0) and {personal_promo} Promotional (1) from personal data.")

    # 2. Ingest Atherion 30K (Kaggle Indian Scam SMS)
    atherion_path = Path("data/external/atherion/dataset_30k_pure.jsonl")
    if atherion_path.exists():
        print(f"\nLoading Atherion 30K dataset from {atherion_path}...")
        ath_ham = []
        ath_scam = []
        ath_spam = []

        with open(atherion_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                msgs = row.get("messages", [])
                if len(msgs) < 3:
                    continue
                user_text = msgs[1].get("content", "").strip()
                asst = msgs[2].get("content", "").strip()
                parts = [p.strip() for p in asst.split(",")]

                if not user_text:
                    continue

                verdict = parts[-1].upper() if parts else ""
                category = parts[1].upper() if len(parts) >= 3 else ""

                if verdict == "LEGITIMATE":
                    # Check if text is promotional
                    if any(re.search(pat, user_text.lower()) for pat in PROMO_KEYWORDS):
                        ath_ham.append({"text": user_text, "address": "DLT-PROMO", "label": 1, "source": "atherion_promo"})
                    else:
                        ath_ham.append({"text": user_text, "address": "DLT-SERVICE", "label": 0, "source": "atherion_ham"})
                elif verdict == "SCAM":
                    # Check if it's commercial unsolicited spam vs high risk fraud
                    if category in ("JOB_OFFER_SCAM", "RENTAL_SCAM", "LOAN_SCAM") and not any(k in user_text.lower() for k in ["arrest", "cbi", "police", "kyc", "block", "freeze", "otp", "password"]):
                        ath_spam.append({"text": user_text, "address": "+919876543210", "label": 2, "source": "atherion_spam"})
                    else:
                        ath_scam.append({"text": user_text, "address": "+919876543210", "label": 3, "source": "atherion_scam"})

        # Subsample to keep balanced
        random.seed(42)
        random.shuffle(ath_ham)
        random.shuffle(ath_scam)
        random.shuffle(ath_spam)

        selected_ath_ham = ath_ham[:max_ham]
        selected_ath_scam = ath_scam[:max_scam]
        selected_ath_spam = ath_spam[:3500]

        dataset.extend(selected_ath_ham)
        dataset.extend(selected_ath_scam)
        dataset.extend(selected_ath_spam)
        print(f"  • Ingested {len(selected_ath_ham)} Ham/Promo, {len(selected_ath_spam)} Spam (2), and {len(selected_ath_scam)} Scam (3) from Atherion.")

    # 3. Ingest ScamShield Multilingual Smishing
    scamshield_path = Path("data/external/scamshield/train.jsonl")
    if scamshield_path.exists():
        print(f"\nLoading ScamShield dataset from {scamshield_path}...")
        ss_scam = 0
        ss_ham = 0
        with open(scamshield_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                txt = row.get("text", "").strip()
                is_scam = row.get("is_scam", 0)
                meta = row.get("metadata", {})

                if not txt:
                    continue

                if is_scam == 1 and ss_scam < 3000:
                    dataset.append({"text": txt, "address": "+919876543210", "label": 3, "source": "scamshield_phish"})
                    ss_scam += 1
                elif is_scam == 0 and ss_ham < 1500 and (meta.get("has_url") or meta.get("has_amount")):
                    dataset.append({"text": txt, "address": "AD-SERVICE-T", "label": 0, "source": "scamshield_ham"})
                    ss_ham += 1

        print(f"  • Ingested {ss_scam} Scams (3) and {ss_ham} Hams (0) from ScamShield.")

    # 4. Ingest JuniorAlive Indian Spam
    ja_path = Path("data/external/junioralive/spam_ham_india.csv")
    if ja_path.exists():
        with open(ja_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header_row = next(reader, None)
            ja_count = 0
            for row in reader:
                if len(row) < 2:
                    continue
                txt = row[0].strip()
                lbl_raw = row[1].strip().lower()
                if not txt:
                    continue
                if lbl_raw == "ham":
                    lbl = 0
                else:
                    if any(k in txt.lower() for k in ["kyc", "block", "arrest", "cbi", "freeze", "lottery"]):
                        lbl = 3
                    elif any(k in txt.lower() for k in ["cert-in", "gov.in", "advises"]):
                        lbl = 0
                    else:
                        lbl = 2
                dataset.append({"text": txt, "address": "+919876543210" if lbl in (2, 3) else "AD-SERVICE-T", "label": lbl, "source": "junioralive_india"})
                ja_count += 1
        print(f"\nLoaded {ja_count} records from JuniorAlive India SMS.")

    # 5. Ingest UCI SMS Spam Collection (Real personal English conversational Ham & Spam)
    uci_path = Path("data/external/uci/spam.csv")
    if uci_path.exists():
        print(f"\nLoading UCI SMS Spam Collection from {uci_path}...")
        uci_count = 0
        with open(uci_path, "r", encoding="latin-1", errors="replace") as f:
            reader = csv.reader(f)
            header_row = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                lbl_raw = row[0].strip().lower()
                txt = row[1].strip()
                if not txt:
                    continue
                if lbl_raw == "ham":
                    lbl = 0
                else:
                    if any(k in txt.lower() for k in ["won", "claim", "prize", "urgent", "cash", "lottery"]):
                        lbl = 3
                    else:
                        lbl = 2
                dataset.append({"text": txt, "address": "+919876543210" if lbl in (2, 3) else "AD-SERVICE-T", "label": lbl, "source": "uci_sms"})
                uci_count += 1
        print(f"  • Ingested {uci_count} records from UCI SMS Collection.")

    # 6. Synthesize Diverse Indian Retail Promotional Offers & Commercial Spam (UCC)
    print("\nSynthesizing diverse Indian promotional and commercial spam examples...")
    RETAIL_ENTITIES = ["SMRTBR", "AJIOIN", "MAXFSN", "DOMINO", "TRENDS", "SWIGGY", "ZOMATO",
                       "MYNTRA", "NYKAA", "FLPKRT", "BIGBSK", "ZEPTO", "BLINKT", "KFCIND",
                       "PIZZAH", "LENSKT", "TATACL", "PHARME", "APOLLO", "MEESHO", "WESTSD"]
    
    PROMO_TEMPLATES = [
        "Get flat {pct}% OFF on all your favorite meals today! Use code {code} at checkout on {brand}. Order now: {url}",
        "Flash Sale Alert! End of Season clearance sale is live @{brand}. Up to {pct}% discount on top apparel: {url}",
        "BOGO Treat! Buy 1 Get 1 FREE on medium & large pizzas today at {brand}. Apply coupon {code}: {url}",
        "Super Saver Wednesday! Fresh fruits & vegetables at unbelievable prices at {brand}. Visit your nearest store today!",
        "Special Festive Offer: Extra {pct}% cashback on orders above Rs.{amt} with code {code}. Valid till midnight: {url}",
        "Weekend Dhamaka! Flat Rs.{amt} OFF on groceries & home essentials at {brand}. Free delivery on orders over Rs.499: {url}",
        "Upgrade your wardrobe! Trending styles starting at just Rs.499 @{brand}. Exclusive preview sale: {url}",
        "Craving dessert? Enjoy 50% OFF up to Rs.100 on your next dessert order on {brand}. Code: {code}. T&C apply.",
        "Your cart misses you! Complete your order now and enjoy free express delivery + extra {pct}% discount: {url}",
        "Recharge with Rs.{amt} today and get unlimited 5G data for 28 days! Fast speeds only on {brand}. Visit {url}",
        "🎵अपना पसंदीदा गाना को बनाएँ फ्री कॉलर ट्यून 🎶 JioTunes में जाएँ और अपनी पसंद का ट्रैक चुनें।",
        "त्यौहारों की खुशियाँ {brand} के साथ! हर खरीदारी पर पाएँ {pct}% की भारी छूट। आज ही पधारें!"
    ]

    SPAM_TEMPLATES = [
        "Pre-approved Personal Loan of Rs. {loan_amt} Lakhs at just {rate}% interest rate. Zero processing fee. Apply today: {url}",
        "Exclusive Launch: Luxury 2 & 3 BHK Apartments on {location} starting at Rs.{price} Lakhs. Zero down payment, free modular kitchen. Call {phone} for site visit.",
        "Instant Home Loan approval in 24 hours! Interest rates starting at {rate}%. No guarantor required. Reply YES or call {phone}.",
        "Guaranteed Stock Market Tips! Earn 10% to 25% daily returns in Nifty & BankNifty options. Join our VIP Telegram channel: {url}",
        "Pre-approved Lifetime Free Credit Card with Rs. 5000 welcome gift voucher and 5% unlimited cashback. Check eligibility now: {url}",
        "Investment Opportunity: Prime commercial office spaces with 12% guaranteed annual rental yield in {location}. Book site visit: {phone}",
        "Direct Admissions open for MBA / B.Tech 2026 batch in top colleges with 100% placement assistance. Call counselor on {phone}.",
        "Special Car Loan Offer! Drive home your dream car at zero down payment and EMI starting Rs.4,999. Apply online: {url}",
        "Health Insurance Policy for entire family covering Rs.1 Crore at just Rs.599/month. Instant paperless issuance: {url}",
        "Instant Business Loan up to Rs.50 Lakhs for MSME / Retailers without collateral. Minimal documentation. Call {phone}."
    ]

    LOCATIONS = ["Noida Expressway", "Gurgaon Golf Course", "Thane West", "Whitefield Bangalore", "Hinjewadi Pune", "Navi Mumbai"]
    PROMO_CODES = ["SAVE50", "BOGO100", "FESTIVE25", "SUPERDEAL", "MEGASALE", "FREESHIP", "WEEKEND40"]

    promo_synth = []
    for _ in range(1200):
        brand = random.choice(RETAIL_ENTITIES)
        tmpl = random.choice(PROMO_TEMPLATES)
        text = tmpl.format(
            pct=random.choice([20, 30, 40, 50, 60, 70]),
            code=random.choice(PROMO_CODES),
            brand=brand,
            amt=random.choice([100, 150, 200, 500, 999]),
            url=f"http://{brand.lower()}.in/offer/{random.randint(100,999)}"
        )
        promo_synth.append({"text": text, "address": f"JD-{brand}-P", "label": 1, "source": "synth_promo"})

    spam_synth = []
    for _ in range(1200):
        tmpl = random.choice(SPAM_TEMPLATES)
        text = tmpl.format(
            loan_amt=random.choice([2, 5, 10, 15, 25]),
            rate=random.choice([8.5, 9.9, 10.5, 11.2]),
            location=random.choice(LOCATIONS),
            price=random.choice([35, 45, 65, 85, 120]),
            phone=f"98{random.randint(10000000, 99999999)}",
            url=f"http://easy-loan{random.randint(10,99)}.in/apply"
        )
        spam_synth.append({"text": text, "address": f"+91{random.randint(7000000000, 9999999999)}", "label": 2, "source": "synth_spam"})

    dataset.extend(promo_synth)
    dataset.extend(spam_synth)
    print(f"  • Added {len(promo_synth)} Promotional (1) and {len(spam_synth)} Commercial Spam (2) templates.")

    print(f"\nRaw total dataset records before sanitization: {len(dataset)}")

    # 6. Sanitize & Normalize
    print("\nApplying PII sanitization and Indic canonical normalization...")
    for item in dataset:
        clean_text = normalize_indic_text(item["text"])
        sanitized = redact_pii(clean_text)
        tokenized = classify_and_tokenize_urls(sanitized)
        item["text"] = tokenized

    # 6. Deduplicate template near-duplicates
    print("\nRunning Shingle Jaccard template deduplication (cutoff = 0.85)...")
    unique_dataset, duplicates_removed = deduplicate_dataset(dataset, similarity_threshold=0.85)
    print(f"Removed {duplicates_removed} duplicate templates. Final unique: {len(unique_dataset)}")

    # 7. Summary Statistics
    counts = Counter(m["label"] for m in unique_dataset)
    print("\n" + "=" * 70)
    print("FINAL UNIFIED DATASET CLASS BREAKDOWN")
    print("=" * 70)
    print(f"Total Unique Messages : {len(unique_dataset)}")
    print(f"  • Class 0 (Safe Ham)        : {counts.get(0, 0):5d} ({counts.get(0, 0)/len(unique_dataset)*100:5.1f}%)")
    print(f"  • Class 1 (Promotional)     : {counts.get(1, 0):5d} ({counts.get(1, 0)/len(unique_dataset)*100:5.1f}%)")
    print(f"  • Class 2 (Commercial Spam) : {counts.get(2, 0):5d} ({counts.get(2, 0)/len(unique_dataset)*100:5.1f}%)")
    print(f"  • Class 3 (Scam / Phishing) : {counts.get(3, 0):5d} ({counts.get(3, 0)/len(unique_dataset)*100:5.1f}%)")
    print("=" * 70)

    # 8. Save
    out_payload = {
        "dataset_name": "SilverGuard-Indic-Unified-4Class",
        "version": "2.0",
        "taxonomy": {
            "0": "ham",
            "1": "promotional",
            "2": "commercial_spam",
            "3": "scam"
        },
        "total": len(unique_dataset),
        "class_counts": dict(counts),
        "messages": unique_dataset
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print(f"💾 Successfully saved unified dataset to: {output_path}")
    return output_path

if __name__ == "__main__":
    build_dataset()
