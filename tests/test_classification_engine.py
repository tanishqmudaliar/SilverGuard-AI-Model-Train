"""
Integration tests for Silver Guard 3-stage cascaded inference engine.
"""

import os
import unittest
import onnxruntime as ort
from silverguard import classify_sms, load_contacts


class TestClassificationEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path = "model/silverguard_fasttext_fusion.onnx"
        cls.contacts_path = "data/contacts.csv"

        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest("FastText ONNX model file not found in model/ directory")

        cls.session = ort.InferenceSession(cls.model_path)
        cls.contacts = load_contacts(cls.contacts_path) if os.path.exists(cls.contacts_path) else {}

    def test_stage1_contact_whitelisting(self):
        test_contacts = {"9876543210": "Aarav Sharma"}
        res = classify_sms(
            session=self.session,
            header="+91 98765 43210",
            message="Hey, can you please transfer Rs 500 on GPay?",
            contacts=test_contacts
        )
        self.assertEqual(res["verdict"], "✅ SAFE (HAM)")
        self.assertEqual(res["threat_score"], 0.0)
        self.assertEqual(res["category"], "ham_contact")
        self.assertEqual(res["stage"], "Stage 1 (Contact Gateway)")

    def test_stage1_dlt_transactional_banking(self):
        res = classify_sms(
            session=self.session,
            header="AD-HDFCBK-T",
            message="Rs 2,450.00 debited from A/c **4120 towards Amazon Pay. Avail bal: Rs 14,210.00. -HDFC Bank",
            contacts=self.contacts
        )
        self.assertEqual(res["verdict"], "✅ SAFE (HAM)")
        self.assertLess(res["threat_score"], 0.05)
        self.assertEqual(res["category"], "ham_transactional")
        self.assertEqual(res["stage"], "Stage 1 (DLT Gateway)")

    def test_stage1_dlt_government_advisory_marathi(self):
        res = classify_sms(
            session=self.session,
            header="JX-ISATHI-G",
            message="तुमच्या नावावर किती सिम आहेत ते तपासायचे आहे का? तपासण्यासाठी संचार साथी डाउनलोड करा. Department of Telecom",
            contacts=self.contacts
        )
        self.assertEqual(res["verdict"], "✅ SAFE (HAM)")
        self.assertLess(res["threat_score"], 0.05)
        self.assertEqual(res["category"], "ham_transactional")

    def test_stage1_dlt_promotional_retail(self):
        res = classify_sms(
            session=self.session,
            header="JK-SMRTBR",
            message="SMART Bazaar Full Paisa Vasool Sale! Get 50% OFF on all groceries and apparel today only. Visit now!",
            contacts=self.contacts
        )
        self.assertEqual(res["verdict"], "📢 PROMOTIONAL (MARKETING / OFFER)")
        self.assertLess(res["threat_score"], 0.05)
        self.assertGreater(res["commercial_score"], 0.90)
        self.assertEqual(res["category"], "promotional")
        self.assertEqual(res["stage"], "Stage 1 (DLT Promo Gateway)")

    def test_stage1_dlt_promotional_hindi_caller_tune(self):
        res = classify_sms(
            session=self.session,
            header="JK-620040-P",
            message="🎵अपना पसंदीदा गाना को बनाएँ फ्री कॉलर ट्यून 🎶 JioTunes में जाएँ और अपनी पसंद का ट्रैक चुनें।",
            contacts=self.contacts
        )
        self.assertEqual(res["verdict"], "📢 PROMOTIONAL (MARKETING / OFFER)")
        self.assertLess(res["threat_score"], 0.05)
        self.assertEqual(res["category"], "promotional")

    def test_stage2_personal_ham_unknown_mobile(self):
        res = classify_sms(
            session=self.session,
            header="+919876543210",
            message="Can we meet at Starbucks tomorrow at 5 PM for coffee?",
            contacts={}
        )
        self.assertEqual(res["verdict"], "✅ SAFE (HAM)")
        self.assertLess(res["threat_score"], 0.30)
        self.assertEqual(res["category"], "ham")

    def test_stage3_unsolicited_commercial_spam(self):
        res = classify_sms(
            session=self.session,
            header="+919876543210",
            message="Exclusive Launch: 2BHK and 3BHK Luxury Flats in Noida Expressway starting 45 Lakhs. Zero down payment, free site visit. Call today.",
            contacts={}
        )
        self.assertEqual(res["verdict"], "📵 UNSOLICITED SPAM (UCC)")
        self.assertEqual(res["category"], "commercial_spam")
        self.assertLess(res["threat_score"], 0.35)
        self.assertGreaterEqual(res["commercial_score"], 0.85)

    def test_stage3_high_risk_cyber_extortion(self):
        res = classify_sms(
            session=self.session,
            header="+919876543210",
            message="CBI NOTICE: An arrest warrant has been issued in your name for money laundering. Contact Mumbai Police cyber cell officer immediately.",
            contacts={}
        )
        self.assertEqual(res["verdict"], "🚨 HIGH RISK SCAM / PHISHING")
        self.assertEqual(res["category"], "scam")
        self.assertGreater(res["threat_score"], 0.90)

    def test_stage3_malicious_apk_download(self):
        res = classify_sms(
            session=self.session,
            header="+919876543210",
            message="Your electricity will be disconnected tonight at 9 PM. Update bill immediately by downloading our support app: http://192.168.1.100/bill.apk",
            contacts={}
        )
        self.assertEqual(res["verdict"], "🚨 HIGH RISK SCAM / PHISHING")
        self.assertEqual(res["category"], "scam")
        self.assertGreater(res["threat_score"], 0.95)


if __name__ == "__main__":
    unittest.main()
