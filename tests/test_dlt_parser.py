"""
Unit tests for TRAI DLT header parsing and sender identification.
"""

import unittest
from silverguard import parse_header_metadata, normalize_phone


class TestDLTParser(unittest.TestCase):

    def test_normalize_phone(self):
        self.assertEqual(normalize_phone("+919876543210"), "9876543210")
        self.assertEqual(normalize_phone("+91 98765 43210"), "9876543210")
        self.assertEqual(normalize_phone("09876543210"), "9876543210")
        self.assertEqual(normalize_phone("9876543210"), "9876543210")
        self.assertIsNone(normalize_phone(""))
        self.assertIsNone(normalize_phone(None))

    def test_dlt_suffixed_headers(self):
        # Transactional suffix (-T)
        meta_t = parse_header_metadata("AD-HDFCBK-T")
        self.assertTrue(meta_t["is_dlt"])
        self.assertEqual(meta_t["category"], "transactional")
        self.assertTrue(meta_t["is_trusted"])

        # Service suffix (-S)
        meta_s = parse_header_metadata("VM-AIRTEL-S")
        self.assertTrue(meta_s["is_dlt"])
        self.assertEqual(meta_s["category"], "service")
        self.assertTrue(meta_s["is_trusted"])

        # Government suffix (-G)
        meta_g = parse_header_metadata("JX-ISATHI-G")
        self.assertTrue(meta_g["is_dlt"])
        self.assertEqual(meta_g["category"], "government")
        self.assertTrue(meta_g["is_trusted"])

        # Promotional suffix (-P)
        meta_p = parse_header_metadata("JK-620040-P")
        self.assertTrue(meta_p["is_dlt"])
        self.assertEqual(meta_p["category"], "promotional")
        self.assertTrue(meta_p["is_trusted"])

    def test_dlt_unsuffixed_commercial_headers(self):
        # Retail brands without explicit -P suffix
        meta_smrtbr = parse_header_metadata("JK-SMRTBR")
        self.assertTrue(meta_smrtbr["is_dlt"])
        self.assertEqual(meta_smrtbr["category"], "promotional")

        meta_ajio = parse_header_metadata("JM-AJIOIN")
        self.assertTrue(meta_ajio["is_dlt"])
        self.assertEqual(meta_ajio["category"], "promotional")

        meta_max = parse_header_metadata("CP-MAXFSN")
        self.assertTrue(meta_max["is_dlt"])
        self.assertEqual(meta_max["category"], "promotional")

    def test_telecom_shortcodes(self):
        meta_sc = parse_header_metadata("1909")
        self.assertTrue(meta_sc["is_dlt"])
        self.assertEqual(meta_sc["category"], "shortcode")
        self.assertTrue(meta_sc["is_trusted"])

        meta_sc5 = parse_header_metadata("56767")
        self.assertTrue(meta_sc5["is_dlt"])
        self.assertEqual(meta_sc5["category"], "shortcode")

    def test_unverified_mobile_senders(self):
        meta_mob = parse_header_metadata("+919876543210")
        self.assertFalse(meta_mob["is_dlt"])
        self.assertEqual(meta_mob["category"], "phone_number")
        self.assertFalse(meta_mob["is_trusted"])
        self.assertEqual(meta_mob["entity"], "9876543210")

    def test_empty_or_unknown_sender(self):
        meta_empty = parse_header_metadata("")
        self.assertFalse(meta_empty["is_dlt"])
        self.assertEqual(meta_empty["category"], "personal")
        self.assertFalse(meta_empty["is_trusted"])


if __name__ == "__main__":
    unittest.main()
