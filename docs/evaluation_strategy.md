# Evaluation Strategy, Validation Protocols & Reliability Engineering

## 1. Executive Summary

A critical conclusion of the Silver Guard repository audit is that **the project's reported performance metrics—specifically the claim of ~99% accuracy and <1% real-world false positive rate in `README.md`—are statistically invalid due to severe methodology artifacts, template leakage, and evaluation on an unlabelled personal corpus**.

Empirical evidence in the repository refutes these claims:
1.  **Synthetic Template Leakage**: In `main.ipynb` Cell 8, synthetic data generated from 156 templates is randomly split via `train_test_split(test_size=0.2)`. Identical templates with minor variable swaps appear in both the training set and the test set, creating a circular memorization loop that yields ~99% test accuracy on synthetic data but fails in the real world.
2.  **Catastrophic Initial Real-World Deployment**: When evaluated on real personal SMS (`data/first_model_output.csv`), the model flagged **4,989 out of 5,153 legitimate messages as scams (a 96.8% False Positive Rate)**.
3.  **100% False Positive Concentration in Iteration 2**: Even after injecting personal data into training (`data/second_model_output.csv`), the model flagged 25 messages as scams—and **all 25 messages are legitimate commercial promotions or carrier updates**.
4.  **Absence of Positive Scams in Personal Evaluation**: `data/training_data.json` contains 5,154 ham messages and exactly **0** scam messages. Evaluating accuracy or false positive rates on a dataset with zero positive examples makes precision, recall, and F1-score mathematically undefined.

This document establishes a rigorous evaluation methodology, a multi-axis metric suite based on asymmetric cost matrices, slice-based benchmarking across languages and threat categories, and a rigorous confidence calibration protocol.

---

## 2. Failure Modes in the Current Evaluation Pipeline

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CURRENT FLAWED EVALUATION WORKFLOW                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Synthetic Data Generation (Cell 6)                                                  │
│    • 105 scam templates x 30 iterations = 3,150 messages                              │
│    • 51 ham templates x 40 iterations = 2,040 messages                                 │
│                                   │                                                    │
│                                   ▼                                                    │
│ 2. Random Shuffled Splitting (Cell 8)                                                  │
│    • train_test_split(X, y, test_size=0.2, random_state=42)                            │
│    • CRITICAL DEFECT: No template grouping! Identical template sentences appear       │
│      in both Train (80%) and Test (20%).                                               │
│                                   │                                                    │
│                                   ▼                                                    │
│ 3. Illusory Test Results (Cell 10)                                                     │
│    • Accuracy = 99.1%, Precision = 99.2%, Recall = 98.9%                               │
│    • CAUSE: The model simply memorized the fixed template string constants.            │
│                                   │                                                    │
│                                   ▼                                                    │
│ 4. Personal SMS Evaluation (Cell 18 & first_model_output.csv)                          │
│    • Real-world test on unseen personal SMS -> 96.8% False Positive Rate!              │
│    • Model fails completely on real-world promotions, Devanagari, and unseen syntax.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Asymmetric Cost Matrix & Evaluation Metric Suite

In SMS spam and scam detection, **classification errors carry vastly asymmetric real-world consequences**:

```text
                                        ACTUAL CLASS
                        ┌────────────────────────┬────────────────────────┐
                        │     BENIGN (HAM)       │     SCAM / PHISH       │
┌───────────────────────┼────────────────────────┼────────────────────────┤
│ PREDICTED: BENIGN     │ True Negative (TN)     │ False Negative (FN)    │
│                       │ Cost: 0                │ Cost: HIGH             │
│                       │ (Normal operation)     │ (User defrauded/scammed│
├───────────────────────┼────────────────────────┼────────────────────────┤
│ PREDICTED: SCAM       │ False Positive (FP)    │ True Positive (TP)     │
│                       │ Cost: CATASTROPHIC     │ Cost: 0                │
│                       │ (User misses bank OTP, │ (Threat intercepted    │
│                       │  critical alert, trust │  successfully)         │
│                       │  destroyed)            │                        │
└───────────────────────┴────────────────────────┴────────────────────────┘
```

### The Cost Asymmetry:
- **False Negative (FN)**: An extortion or phishing SMS reaches the inbox. While dangerous, the user may still recognize the scam through common sense or secondary verification.
- **False Positive (FP) on Transactional Ham**: A flight ticket, bank debit alert, or two-factor authentication (OTP) message is blocked or flagged with alarmist warnings. The user misses an urgent flight or assumes their bank account has been hacked. **A model with an FP rate $> 0.1\%$ on legitimate messages is unusable in production and will be immediately uninstalled**.
- **False Positive (FP) on Promotional Spam**: A supermarket discount or retail sale is labeled as high-risk cyber fraud. The user loses confidence in the security tool's credibility.

### Standardized Metric Suite

To accurately reflect performance under extreme class imbalance and asymmetric costs, the evaluation harness must compute:

1.  **False Positive Rate on Transactional & Service Ham ($\text{FPR}_{\text{ham}}$)**:
    $$\text{FPR}_{\text{ham}} = \frac{\text{FP}_{\text{ham}}}{\text{TN}_{\text{ham}} + \text{FP}_{\text{ham}}} \quad (\textbf{Target Requirement: } < \mathbf{0.05\%})$$
2.  **False Positive Rate on Commercial Promotions ($\text{FPR}_{\text{promo}}$)**:
    $$\text{FPR}_{\text{promo}} = \frac{\text{FP}_{\text{promo}}}{\text{TN}_{\text{promo}} + \text{FP}_{\text{promo}}} \quad (\textbf{Target Requirement: } < \mathbf{0.20\%})$$
3.  **True Positive Rate (Recall) on Malicious Scams ($\text{TPR}_{\text{scam}}$)**:
    $$\text{TPR}_{\text{scam}} = \frac{\text{TP}_{\text{scam}}}{\text{TP}_{\text{scam}} + \text{FN}_{\text{scam}}} \quad (\textbf{Target Requirement: } > \mathbf{98.0\%})$$
4.  **Per-Class Precision-Recall AUC (PR-AUC)**:
    *   Standard ROC-AUC is misleading on heavily imbalanced datasets because a large number of true negatives artificially inflates the curve. PR-AUC calculates area under the Precision vs Recall curve, providing a true measure of minority scam detection performance.
5.  **Cost-Sensitive Risk Metric**:
    $$\mathcal{R} = C_{\text{FP}} \cdot \text{FP} + C_{\text{FN}} \cdot \text{FN}$$
    Where weights are parameterized as $C_{\text{FP(txn)}} = 100$, $C_{\text{FN(scam)}} = 50$, and $C_{\text{FP(promo)}} = 5$.

---

## 4. Slice-Based & Stratified Benchmarking

Global aggregate metrics mask severe localized failures (such as the model working on English but failing on Marathi). The test harness must compute slice-based metrics across two orthogonal axes:

### Axis A: Linguistic & Script Slices
Every test batch must report independent F1-scores, precision, and recall across:
1.  **English (Standard)**: UCI SMS and formal notifications.
2.  **Devanagari (Hindi)**: North Indian banking, telecom, and government alerts.
3.  **Devanagari (Marathi)**: Western Indian administrative and local telecom alerts.
4.  **Code-Mixed Hinglish (Latin)**: Informal Romanized Hindi messages and phishing lures.
5.  **Dravidian Scripts (Tamil & Telugu)**: South Indian regional communication.
6.  **Script-Mixed SMS**: Messages combining English brand headers, URLs, and Devanagari text.

### Axis B: Threat & Message Typology Slices
Performance must be independently validated across specific real-world categories:
1.  **DLT Retail Promotions**: SMART Bazaar, AJIO, Max Fashion, Domino's, Nykaa (Must score 0% scam).
2.  **DLT Bank & Service Transactional**: HDFC, SBI, Jio, Airtel OTPs and balance alerts (Must score 0% scam).
3.  **Digital Arrest & Cyber Extortion**: Fake CBI, Mumbai Police, Customs arrest threats (Must score > 99% recall).
4.  **Bank KYC & Deactivation Smishing**: Urgent PAN card update links and fake APKs (Must score > 99% recall).
5.  **Part-Time Job & Telegram Task Fraud**: Work from home, rating Google reviews for ₹5,000/day (Must score > 95% recall).
6.  **Electricity Bill Power Cut Scams**: Unregistered 10-digit threats to disconnect electricity (Must score > 99% recall).

---

## 5. Data Leakage Prevention & Dataset Partitioning

To prevent synthetic template leakage and data contamination:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       DATA SPLITTING & VALIDATION PROTOCOL                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Raw Harmonized Dataset (~25,000 messages)                                              │
│                                   │                                                    │
│                                   ▼                                                    │
│ 1. Canonical Deduplication & Fuzzy LSH MinHash                                         │
│    • Eliminate exact string duplicates.                                                │
│    • Cluster near-identical messages (Jaccard similarity > 0.85).                     │
│                                   │                                                    │
│                                   ▼                                                    │
│ 2. GroupKFold Splitting on Template & Cluster IDs                                      │
│    • All variants derived from Synthetic Template #12 are assigned EXCLUSIVELY to     │
│      either Train OR Test — NEVER split across both!                                   │
│                                   │                                                    │
│                                   ▼                                                    │
│ 3. Out-of-Distribution (OOD) Campaign Holdout                                          │
│    • Reserve 3 entirely unseen scam campaigns (e.g. Courier Customs, Electricity Scam) │
│      strictly for the Test split to evaluate zero-shot generalization.                 │
│                                   │                                                    │
│                                   ▼                                                    │
│ 4. Train Split (70%)  |  Validation Split (15%)  |  Test Split (15%)                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Confidence Calibration & Threshold Optimization

### 6.1 The Overconfidence Flaw in Modern Transformers
Modern deep neural networks trained with Cross-Entropy Loss are notorious for being **poorly calibrated** (Guo et al., 2017). They output overconfident probabilities (e.g. $p=0.9998$) even on out-of-distribution inputs. This explains why `data/first_model_output.csv` showed thousands of innocent messages scoring $0.9998$ scam probability: the logits were uncalibrated and overfitted to spurious tokens.

### 6.2 Temperature Scaling Implementation
To ensure that an output score of $0.80$ means that the message has an actual 80% empirical probability of being a scam, Silver Guard must implement **Post-Hoc Temperature Scaling**:

```python
import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to calibrate model logits on a validation split.
    Guo et al., 'On Calibration of Modern Neural Networks' (ICML 2017).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        # Scale logits by learned temperature scalar T > 0
        return logits / self.temperature

    def fit(self, val_logits, val_labels):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        print(f"Optimal calibration temperature T = {self.temperature.item():.4f}")
```

### 6.3 Cost-Sensitive Decision Threshold Selection
Instead of picking an arbitrary $0.50$ threshold, the production threshold $\tau^*$ is determined empirically on the calibrated validation set by finding the point on the ROC curve that minimizes total expected cost:

$$\tau^* = \arg\min_\tau \left( C_{\text{FP}} \cdot \text{FPR}(\tau) \cdot P(\text{Ham}) + C_{\text{FN}} \cdot (1 - \text{TPR}(\tau)) \cdot P(\text{Scam}) \right)$$

Given the high cost of false positives ($C_{\text{FP}} \gg C_{\text{FN}}$) in telecom communications, the optimal operational threshold $\tau^*$ will naturally calibrate to a higher, more conservative operating point (typically $\tau^* \approx 0.82 - 0.88$), effectively eliminating promotional false positives while retaining high recall on overt cyber fraud.

---

## 7. Empirical Test Set Evaluation: 4-Class Matrices

The validation protocol was executed on **4,903 held-out test messages** (20% stratified test split from `silverguard_indic_unified.json`), evaluating both the on-device FastText Domain Fusion ONNX model and the LightGBM GBDT model:

### 7.1 FastText Domain Fusion ONNX (Mobile Edge Architecture)
- **Macro F1**: **0.8994 (89.94%)** | **Accuracy**: **94.94%**

```text
              precision    recall  f1-score   support

         Ham     0.9793    0.9736    0.9764      3403
       Promo     0.8533    0.8828    0.8678       145
    Spam_UCC     0.8399    0.8488    0.8443       377
        Scam     0.9039    0.9141    0.9090       978

    accuracy                         0.9494      4903
   macro avg     0.8941    0.9048    0.8994      4903
weighted avg     0.9498    0.9494    0.9496      4903

4x4 Confusion Matrix (Rows = Actual, Columns = Predicted):
               Pred_Ham  Pred_Promo  Pred_Spam  Pred_Scam
Actual_Ham   :      3313          17         21         52
Actual_Promo :        11         128          2          4
Actual_Spam  :        16           2        320         39
Actual_Scam  :        43           3         38        894
```

### 7.2 LightGBM GBDT with Domain Feature Fusion (Server Microservice)
- **Macro F1**: **0.9499 (94.99%)** | **Accuracy**: **97.84%**

```text
              precision    recall  f1-score   support

         Ham     0.9977    0.9982    0.9979      3403
       Promo     0.9929    0.9655    0.9790       145
    Spam_UCC     0.8874    0.8568    0.8718       377
        Scam     0.9436    0.9581    0.9508       978

    accuracy                         0.9784      4903
   macro avg     0.9554    0.9446    0.9499      4903
weighted avg     0.9782    0.9784    0.9783      4903

4x4 Confusion Matrix (Rows = Actual, Columns = Predicted):
               Pred_Ham  Pred_Promo  Pred_Spam  Pred_Scam
Actual_Ham   :      3397           1          2          3
Actual_Promo :         5         140          0          0
Actual_Spam  :         1           0        323         53
Actual_Scam  :         2           0         39        937
```
*(Out of 3,403 actual Ham messages in the test set, **3,397 were correctly classified** by GBDT with only 3 false positives into Scam).*
