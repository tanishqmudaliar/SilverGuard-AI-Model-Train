"""
Unit tests for Silver Guard dataset preparation and preprocessing pipeline.
"""

import unittest
from dataset_pipeline import (
    redact_pii,
    classify_and_tokenize_urls,
    get_word_shingles,
    jaccard_similarity,
    deduplicate_dataset,
    align_external_dataset
)


class TestDatasetPipeline(unittest.TestCase):

    def test_redact_phone_numbers(self):
        text1 = "Call me at +91 98765 43210 immediately."
        self.assertEqual(redact_pii(text1), "Call me at [PHONE_NUM] immediately.")

        text2 = "Contact agent on 9876543210 for details."
        self.assertEqual(redact_pii(text2), "Contact agent on [PHONE_NUM] for details.")

    def test_redact_account_numbers(self):
        text = "Your A/c *******4120 credited with Rs 1,500."
        self.assertEqual(redact_pii(text), "Your A/c [ACCT_NUM] credited with Rs 1,500.")

        text2 = "Account 5010023456789 balance enquiry."
        self.assertEqual(redact_pii(text2), "Account [ACCT_NUM] balance enquiry.")

    def test_redact_otp_codes(self):
        text = "Your SBI OTP is 482910 for txn of Rs 5000."
        self.assertEqual(redact_pii(text), "Your SBI OTP is [OTP_CODE] for txn of Rs 5000.")

        text2 = "One time password is 123456. Do not share."
        self.assertEqual(redact_pii(text2), "One time password is [OTP_CODE]. Do not share.")

    def test_url_tokenization_risk_categories(self):
        # Shorteners
        t_short = classify_and_tokenize_urls("Click here: http://bit.ly/claim-offer today")
        self.assertIn("[URL_SHORTENER]", t_short)

        # Direct IP
        t_ip = classify_and_tokenize_urls("Login at http://192.168.1.1/admin to verify")
        self.assertIn("[URL_IP]", t_ip)

        # APK download
        t_apk = classify_and_tokenize_urls("Download app from https://evil-site.com/banking.apk now")
        self.assertIn("[URL_APK]", t_apk)

        # Whitelisted domains
        t_bank = classify_and_tokenize_urls("Visit official portal at https://www.hdfcbank.com/netbanking")
        self.assertIn("[URL_WHITELISTED]", t_bank)

        t_govt = classify_and_tokenize_urls("Check telecom portal https://sancharsaathi.gov.in")
        self.assertIn("[URL_WHITELISTED]", t_govt)

        # Generic URL
        t_gen = classify_and_tokenize_urls("Visit my blog at https://myportfolio.io/about")
        self.assertIn("[URL]", t_gen)

    def test_shingling_and_jaccard_deduplication(self):
        msg1 = "Your OTP for HDFC Bank login is 112233. Valid for 10 mins."
        msg2 = "Your OTP for HDFC Bank login is 998877. Valid for 10 mins."
        msg3 = "Flat 50% discount on shoes at Puma store today only."

        sh1 = get_word_shingles(msg1)
        sh2 = get_word_shingles(msg2)
        sh3 = get_word_shingles(msg3)

        sim_near = jaccard_similarity(sh1, sh2)
        sim_diff = jaccard_similarity(sh1, sh3)

        # msg1 and msg2 differ only by OTP digits; shingle similarity should be high (> 0.65)
        self.assertGreater(sim_near, 0.65)
        self.assertLess(sim_diff, 0.15)

        dataset = [
            {"text": msg1, "label": 0},
            {"text": msg2, "label": 0},
            {"text": msg3, "label": 1}
        ]
        unique, removed = deduplicate_dataset(dataset, similarity_threshold=0.65)
        self.assertEqual(len(unique), 2)
        self.assertEqual(removed, 1)


if __name__ == "__main__":
    unittest.main()
