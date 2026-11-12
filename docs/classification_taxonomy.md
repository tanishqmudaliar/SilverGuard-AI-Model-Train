# Classification Strategy & Semantic Taxonomy for SMS Security

## 1. Executive Summary

A primary finding of the Silver Guard repository audit is that **collapsing all unsolicited messages into a binary choice ($0 = \text{Ham}, 1 = \text{Scam}$) is mathematically and semantically unviable for real-world SMS processing**. In modern telecommunications—especially in India where the Telecom Commercial Communications Customer Preference Regulations (TRAI TCCCPR) govern hundreds of millions of commercial messages daily—an SMS message can be benign, annoying, commercial, or actively criminal.

Treating commercial advertisements (spam) and fraudulent attacks (scams) as the same class causes severe false positive alarms, degrades user trust, and trains the neural network on contradictory lexical signals.

This document establishes:
1. A rigorous **5-tier semantic taxonomy** defining the exact differences between Ham, Service, Promotional Spam, and Malicious Scams.
2. A formal architectural comparison between **Flat Binary**, **Flat Multi-Class**, and **Hierarchical Cascade** classification models.
3. An **Ambiguity Resolution Protocol** for handling edge cases like pre-approved bank loans, delivery OTPs from unknown numbers, and discount links.
4. Specific algorithmic techniques for reducing false positive rates on commercial traffic to $< 0.1\%$.

---

## 2. The Semantic Taxonomy: Defining Message Classes

In academic literature (Almeida et al., 2012; Mishra & Soni, 2020; Deligiannis et al., 2023), communication is partitioned by intent, authorization, and threat profile. For Silver Guard, we define a standardized 5-tier taxonomy:

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                    INCOMING SMS                        │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┴─────────────────────────────────────┐
                    ▼                                                                           ▼
         ┌─────────────────────┐                                                     ┌─────────────────────┐
         │    BENIGN TRAFFIC   │                                                     │   THREAT TRAFFIC    │
         └──────────┬──────────┘                                                     └──────────┬──────────┘
                    │                                                                           │
       ┌────────────┴────────────┐                                                 ┌────────────┴────────────┐
       ▼                         ▼                                                 ▼                         ▼
┌──────────────┐         ┌──────────────┐                                   ┌──────────────┐         ┌──────────────┐
│  CLASS 0     │         │  CLASS 1     │                                   │  CLASS 2     │         │  CLASS 3     │
│  HAM         │         │  PROMOTIONAL │                                   │  COMMERCIAL  │         │  SCAM /      │
│  (Personal / │         │  (Registered │                                   │  SPAM (UCC)  │         │  PHISHING    │
│  Service /   │         │   Marketing) │                                   │  (Unsolicited│         │  (Malicious/ │
│  Transaction)│         │              │                                   │   10-digit)  │         │   Criminal)  │
└──────────────┘         └──────────────┘                                   └──────────────┘         └──────────────┘
```

### Class 0: Ham (Legitimate & Benign)
Benign messages are communications where the sender has a legitimate personal or business relationship with the recipient, or is fulfilling an explicit user request.
*   **0A. Personal P2P**: Direct interpersonal messages between individuals (friends, family, colleagues).
    *   *Example*: *"Hey, are you free for lunch today at 1?"*
*   **0B. Transactional A2P**: Critical operational notifications triggered directly by user action. Includes banking debits/credits, One-Time Passwords (OTPs), 2FA codes, flight boarding passes, and e-commerce delivery dispatches.
    *   *Example*: `AD-HDFCBK` — *"Rs. 1,450.00 debited from A/c **4120 at Zomato on 24-Feb-26. Not you? SMS BLOCK to 56767."*
*   **0C. Service / Informational A2P**: Non-transactional updates sent to existing subscribers or citizens, such as utility bill generation, telecom validity expiry warnings, or government advisories (e.g. TRAI Sanchar Saathi or disaster alerts).
    *   *Example*: `JM-JioPay-S` — *"Your monthly postpaid bill for mobile 98XXXXXXXX of Rs. 470.82 is generated. Due date: 15-Mar."*

### Class 1: Legitimate Promotional / Marketing (Registered DLT)
Commercial messages sent by registered commercial enterprises (Principal Entities) with a DLT-approved promotional header or template. They contain advertisements, discount vouchers, sales announcements, and loyalty point updates.
*   **Key Characteristic**: They seek commercial profit, but they **do not engage in identity theft, credential phishing, malware distribution, or extortion**.
*   *Example*: `JK-SMRTBR` — *"Get ready for TOMORROW! SMART Bazaar FULL PAISA VASOOL Preview Day SALE Early Access to 33% off on groceries!"*
*   *Example*: `JM-AJIOIN` — *"Payday Special! Get Extra 20% off on footwear above Rs. 1999. Use code FEB20. Shop now on AJIO."*

### Class 2: Commercial Spam / Unsolicited Commercial Communication (UCC)
Unsolicited advertising sent from unregistered senders (such as standard 10-digit mobile SIMs or grey-route bulk aggregators) violating regulatory DND (Do-Not-Disturb) mandates.
*   **Key Characteristic**: Annoying, persistent, and unauthorized advertising (real estate leads, loan brokerage, local coaching classes), but still **lacking direct criminal intent** (i.e. the product is real, but the communication channel is unsolicited).
*   *Example*: `+919820112233` — *"Pre-launch luxury 2&3 BHK flats in Thane starting 85 Lacs. Zero stamp duty. Call 9820112233 for site visit."*

### Class 3: Malicious Fraud, Scam & Phishing (Smishing)
Criminal communications engineered to deceive, defraud, extort, or compromise the security of the victim.
*   **3A. Credential Phishing / Smishing**: Impersonating banks, government portals, or utility boards to steal credentials via fake websites.
    *   *Example*: `+917890123456` — *"Dear customer, your SBI NetBanking is locked today. Update your PAN card immediately to prevent permanent deactivation: bit.ly/sbi-pan-kyc"*
*   **3B. Digital Arrest & Cyber Extortion**: Impersonating law enforcement agencies (CBI, Mumbai Police, Customs, Narcotics Bureau, ED) claiming a parcel contains illegal contraband or an arrest warrant is pending.
    *   *Example*: `+919123456780` — *"CBI NOTICE: Your Aadhaar is implicated in an international money laundering case in Mumbai. Call IO Kumar immediately at 9123456780 or digital arrest will follow."*
*   **3C. Advance-Fee & Prize Fraud**: Luring victims with fake lottery prizes (KBC), part-time job schemes (YouTube like/Telegram task scams), or pre-approved loans requiring an "advance processing fee".
    *   *Example*: `+918877665544` — *"Congratulations! You are selected for Part-time Work from Home. Earn 3000-8000 daily by rating Google hotels. Join Telegram: t.me/daily_task_in"*
*   **3D. Financial Impersonation & Malicious APKs**: Sending fake electricity disconnection threats or courier delivery failure notices instructing users to download an `.apk` file (such as a remote desktop RAT like AnyDesk or a credential-stealing Trojan).
    *   *Example*: `+919001122334` — *"Dear consumer your electricity power will be disconnected tonight 9:30 PM from power office because your previous bill was not update. Please contact our officer 9001122334."*

---

## 3. Comparison of Classification Architectural Strategies

We analyze three fundamental classification paradigms for Silver Guard:

| Criterion | Architecture 1: Flat Binary | Architecture 2: Flat Multi-Class | Architecture 3: Hierarchical Cascade |
| :--- | :--- | :--- | :--- |
| **Classes Modeled** | 2 (`Ham` vs `Scam`) | 3 or 4 (`Ham`, `Promo`, `Spam`, `Scam`) | Tier 1: Benign vs Suspicious<br>Tier 2: Promo vs Phishing/Scam |
| **Model Footprint** | Single model (~105 MB FP32) | Single model (slightly larger head) | Hybrid: Rule Engine + Multi-head Model |
| **False Positive Handling** | **Extremely Poor**: Promos collapse into Scams | **Moderate**: Requires high class separation | **Superior**: Promos explicitly isolated |
| **Inference Latency** | Full neural forward pass every SMS | Full neural forward pass every SMS | 80% of traffic handled in $<1\text{ ms}$ (Rule engine) |
| **Explainability** | Single scalar `threat_score` | Class probability vector | Structured audit trail & threat indicators |
| **Suitability for Silver Guard** | ❌ **Rejected** (Current flawed setup) | 🟡 **Acceptable** (Step forward) | ✅ **Recommended Production Standard** |

### Why Flat Binary Fails
Flat binary classification suffers from **semantic compression**: it forces promotional offers and malicious phishing into a single hyperplane partition. Because promotional messages share lexical features with scams (urgency words like `"HURRY"`, financial terms like `"Rs."`, discounts, and links), a binary classifier trained with standard Cross-Entropy Loss cannot distinguish between a legitimate 50% discount and a fraudulent phishing lure without memorizing specific sender IDs.

### The Recommended Architecture: The 3-Stage Hierarchical Cascade

To achieve industrial reliability, Silver Guard should adopt a **Hierarchical Cascade Architecture**:

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                 INCOMING SMS MESSAGE                   │
                               │               (Header + Body + Metadata)               │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │           STAGE 1: DETERMINISTIC GATEWAY               │
                               │             (Rules, DLT & Contact Engine)              │
                               │                Execution Latency: < 1 ms               │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                     ┌────────────────────────────────────┴────────────────────────────────────┐
                     │                                                                         │
        [Matches Known Contact]                                                   [Unknown Sender OR No Match]
        [OR Verified DLT Transactional/Service]                                                │
                     │                                                                         │
                     ▼                                                                         ▼
      ┌─────────────────────────────┐                                           ┌─────────────────────────────┐
      │  DIRECT VERDICT: SAFE (HAM) │                                           │     STAGE 2: MULTILINGUAL   │
      │   (Bypass Neural Model)     │                                           │        INTENT CLASSIFIER    │
      └─────────────────────────────┘                                           │   Execution Latency: ~30 ms │
                                                                                └──────────────┬──────────────┘
                                                                                               │
                                            ┌──────────────────────────────────────────────────┴───────────────┐
                                            ▼                                                                  ▼
                                 [Intent: Commercial Promo]                                         [Intent: Security Risk / Phishing]
                                            │                                                                  │
                                            ▼                                                                  ▼
                             ┌─────────────────────────────┐                                    ┌─────────────────────────────┐
                             │  DIRECT VERDICT: PROMOTIONAL│                                    │    STAGE 3: RISK & EVIDENCE │
                             │  (Flag as Ad / Junk folder) │                                    │          ANALYSIS ENGINE    │
                             └─────────────────────────────┘                                    │   Execution Latency: < 2 ms │
                                                                                                └──────────────┬──────────────┘
                                                                                                               │
                                                                                                               ▼
                                                                                                ┌─────────────────────────────┐
                                                                                                │  FINAL VERDICT: SCAM/PHISH  │
                                                                                                │  Threat Score + Actionable  │
                                                                                                │  Indicator Breakdown        │
                                                                                                └─────────────────────────────┘
```

#### Advantages of the Hierarchical Cascade:
1. **Zero False Positives on Verified Contacts**: Messages from family and friends in `contacts.csv` bypass the neural network entirely.
2. **Computational Efficiency on Mobile**: Over 70% of messages received by an Indian user are standard transactional alerts (`AD-HDFCBK-T`) or known contacts. Processing these via Stage 1 saves device battery and eliminates latency.
3. **Explicit Isolation of Promotions**: A message flagged as commercial is routed to a "Promotions / Offers" bucket rather than sounding high-risk scam alarms.

---

## 4. Differentiating Unsolicited Marketing from Malicious Fraud

To prevent ordinary advertisements from being flagged as scams, the system must evaluate four orthogonal feature axes:

### Axis 1: Sender ID Identity & DLT Registration
*   **Promotional / Legitimate**: Sent from an alphanumeric registered Principal Entity header (e.g. `JK-SMRTBR`, `VM-DOMINO`). Under TRAI regulations, legitimate businesses pay substantial registration fees and bind their telemarketer chains to access these headers.
*   **Scam / Malicious**: Scammers overwhelmingly operate from standard 10-digit SIM cards (`+9198XXXXXXXX`), virtual VoIP numbers, or spoofed overseas numbers, because telecom carriers immediately terminate DLT accounts found propagating fraud.

### Axis 2: Call-to-Action (CTA) and Behavioral Intent
*   **Promotional CTA**: Encourages purchasing a product, visiting a legitimate retail store, redeeming a loyalty voucher, or browsing an official e-commerce app (*"Shop now @TRENDS"*, *"Visit nearest SMART Bazaar"*).
*   **Scam CTA**: Demands immediate corrective security action under duress (*"Call immediately to avoid arrest"*, *"Click link to verify PAN"*, *"Install APK to restore power"*).

### Axis 3: Link Target & URL Architecture
*   **Promotional Links**: Direct to registered domains, official subdomains, or TRAI-whitelisted shorteners (e.g. `jio.com`, `t.jio/...`, `nykaa.com`, `amzn.to`).
*   **Scam Links**: Direct to generic link shorteners (`bit.ly`, `tinyurl.com`, `is.gd`, `rb.gy`), free hosting providers (`.firebaseapp.com`, `.pages.dev`), newly registered suspicious TLDs (`.top`, `.xyz`, `.online`, `.support`), or direct download links for Android application packages (`.apk`).

### Axis 4: Financial Transaction Directionality
*   **Promotional Messages**: Offer discounts on user expenditure (*"Get 500 OFF on shopping of 1500"*).
*   **Scam Messages**: Solicit sensitive credentials (debit card PIN, CVV, NetBanking password), demand urgent fees to release non-existent lottery funds, or instruct the victim to enter their UPI PIN to "receive" money (a classic UPI reverse-payment scam).

---

## 5. Ambiguity Resolution Protocol for Edge Cases

Real-world SMS classification involves boundary cases where lexical features overlap. The following rules govern ambiguity resolution:

```text
+-------------------------------------------------------+----------------------+-----------------------+
| Edge Case Scenario                                    | Traditional Flaw     | Silver Guard Protocol |
+-------------------------------------------------------+----------------------+-----------------------+
| 1. Legitimate bank sends loan advertisement           | Flagged as Scam      | Classified as Promo   |
|    "HDFC Bank: Pre-approved personal loan @ 10.5%.   | (Financial terms +   | (Sender is registered |
|     Apply in NetBanking."                              |  loan keywords)      |  DLT; no phishing URL)|
+-------------------------------------------------------+----------------------+-----------------------+
| 2. Delivery OTP sent from 10-digit mobile number      | Flagged as Scam      | Classified as Ham/OTP |
|    "Your delivery OTP for Amazon order is 4821."      | (Unknown phone       | (Passive OTP; does not|
|                                                       |  number sender)      |  solicit callback/URL)|
+-------------------------------------------------------+----------------------+-----------------------+
| 3. High-urgency brand sale with shortlink             | Flagged as Scam      | Classified as Promo   |
|    "HURRY! Last 2 hours! 70% off at Zara.             | (Urgency words +     | (DLT header verified; |
|     Shop: bit.ly/zara-sale"                           |  shortened link)     |  domain resolves safe)|
+-------------------------------------------------------+----------------------+-----------------------+
| 4. Personal message containing a link or phone number | Flagged as Scam      | Classified as Ham     |
|    "Check out this cafe: https://maps.app.goo.gl/..." | (Unknown URL)        | (Contact whitelist    |
|                                                       |                      |  bypasses inspection) |
+-------------------------------------------------------+----------------------+-----------------------+
| 5. Order cancellation from legitimate brand           | Flagged as Scam      | Classified as Ham/Txn |
|    "AJIO order FN8771609821 has been cancelled."      | (Negative sentiment  | (Known DLT entity;    |
|                                                       |  + order keywords)   |  standard service msg)|
+-------------------------------------------------------+----------------------+-----------------------+
| 6. Sanchar Saathi government security alert in Marathi| Flagged as Scam      | Classified as Ham/Gov |
|    "तुमच्या नावावर किती सिम आहेत ते तपासायचे आहे का?"   | (Devanagari script + | (Government DLT suffix|
|                                                       |  telecom keywords)   |  -G / verified header)|
+-------------------------------------------------------+----------------------+-----------------------+
```

---

## 6. False Positive Defense Architecture

To ensure the false positive rate on legitimate messages remains below **0.05%**, the classification engine must enforce the following safeguards:

1.  **Strict Separate Output Thresholds**:
    *   Do not present a binary threat slider. Output two distinct indices:
        $$\text{Commerciality Score } S_{\text{promo}} \in [0, 1]$$
        $$\text{Malicious Threat Score } S_{\text{scam}} \in [0, 1]$$
    *   A message can only be flagged as a scam if $S_{\text{scam}} > \tau_{\text{scam}}$ (where $\tau_{\text{scam}} \ge 0.85$).
2.  **DLT Header High-Confidence Veto**:
    *   If a message originates from a cryptographically verified or telecom-verified DLT header (e.g. `AD-HDFCBK`, `JD-SBINOT`, `DL-GOVTNG`), it **cannot be classified as a high-risk scam unless it contains a blacklisted phishing domain or a direct `.apk` payload link**.
3.  **Explainability and Evidence Surfacing**:
    *   Instead of an ambiguous "🚨 HIGH RISK SCAM" banner, the UI must display concrete evidentiary rationale:
        *   *"Why this was flagged: Sent from an unverified 10-digit number claiming to be CBI, contains high-urgency arrest threats, and demands a call to an unknown number."*
