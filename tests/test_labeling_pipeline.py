"""
Unit tests for label_sms.py and merge_labels.py in the 4-Class Taxonomy.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from label_sms import parse_sender_info, get_heuristic_review_hint


class TestLabelingPipeline(unittest.TestCase):

    def test_parse_sender_info_dlt(self):
        # Promotional DLT header
        p_info = parse_sender_info("JM-RTRNDS-P")
        self.assertTrue(p_info["is_dlt"])
        self.assertEqual(p_info["category"], "promotional")

        # Retail entity without -P
        smart_info = parse_sender_info("JK-SMRTBR")
        self.assertTrue(smart_info["is_dlt"])
        self.assertEqual(smart_info["category"], "promotional")

        # Bank transactional DLT header
        b_info = parse_sender_info("AD-HDFCBK-T")
        self.assertTrue(b_info["is_dlt"])
        self.assertEqual(b_info["category"], "transactional")

        # Phone number
        ph_info = parse_sender_info("+919876543210")
        self.assertFalse(ph_info["is_dlt"])
        self.assertEqual(ph_info["category"], "phone_number")

    def test_heuristic_review_hints(self):
        # Extortion threat keywords -> suggested 3 (scam)
        scam_txt = "Delhi Police Cyber Crime: Digital arrest warrant issued for money laundering. Call immediately."
        sugg_scam, hint_scam = get_heuristic_review_hint(scam_txt, phone_freq=1)
        self.assertEqual(sugg_scam, 3)
        self.assertIn("threat_indicators", hint_scam)

        # Commercial offer keywords -> suggested 2 (spam)
        promo_txt = "Pre-approved personal loan of Rs 5 Lakhs with zero interest rate. Apply today."
        sugg_promo, hint_promo = get_heuristic_review_hint(promo_txt, phone_freq=1)
        self.assertEqual(sugg_promo, 2)
        self.assertIn("commercial_offer", hint_promo)

        # Frequent contact without suspicious keywords -> suggested 0 (ham)
        ham_txt = "Hey, are we still meeting for lunch today at 1 PM?"
        sugg_ham, hint_ham = get_heuristic_review_hint(ham_txt, phone_freq=5)
        self.assertEqual(sugg_ham, 0)
        self.assertIn("frequent_sender", hint_ham)


if __name__ == "__main__":
    unittest.main()
