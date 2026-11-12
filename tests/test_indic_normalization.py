"""
Unit tests for Indic text normalization and multilingual preservation.
"""

import unittest
from silverguard import clean_and_normalize_multilingual, is_indic_mark


class TestIndicNormalization(unittest.TestCase):

    def test_is_indic_mark(self):
        # Devanagari vowel sign AA (ा: 0x093E)
        self.assertTrue(is_indic_mark("\u093e"))
        # Devanagari virama (्: 0x094D)
        self.assertTrue(is_indic_mark("\u094d"))
        # Tamil vowel sign I (ி: 0x0BC0)
        self.assertTrue(is_indic_mark("\u0bc0"))
        # Telugu vowel sign AA (ా: 0x0C3E)
        self.assertTrue(is_indic_mark("\u0c3e"))
        # Latin combining acute (́: 0x0301) - Western accent, should be False
        self.assertFalse(is_indic_mark("\u0301"))

    def test_preserves_devanagari_matras_and_conjuncts(self):
        # Hindi: "अपना पसंदीदा गाना को बनाएँ फ्री कॉलर ट्यून"
        raw = "अपना पसंदीदा गाना को बनाएँ फ्री कॉलर ट्यून"
        normalized = clean_and_normalize_multilingual(raw)
        self.assertIn("पसंदीदा", normalized)
        self.assertIn("गाना", normalized)
        self.assertIn("फ्री", normalized)
        self.assertIn("कॉलर", normalized)
        self.assertIn("ट्यून", normalized)

    def test_preserves_marathi_text(self):
        # Marathi: "तुमच्या नावावर किती सिम आहेत ते तपासायचे आहे का?"
        raw = "तुमच्या नावावर किती सिम आहेत ते तपासायचे आहे का?"
        normalized = clean_and_normalize_multilingual(raw)
        self.assertIn("तुमच्या", normalized)
        self.assertIn("तपासायचे", normalized)

    def test_preserves_dravidian_scripts(self):
        # Tamil: வணக்கம் (Vanakkam)
        raw_ta = "வணக்கம்"
        norm_ta = clean_and_normalize_multilingual(raw_ta)
        self.assertEqual(norm_ta, "வணக்கம்")

        # Telugu: నమస్కారం (Namaskaram)
        raw_te = "నమస్కారం"
        norm_te = clean_and_normalize_multilingual(raw_te)
        self.assertEqual(norm_te, "నమస్కారం")

    def test_emoji_and_symbol_isolation(self):
        # Emojis adjacent to words should be isolated with spaces to prevent [UNK] collapses
        raw = "🎵गाना🎶ऑफ़र50%🎉"
        normalized = clean_and_normalize_multilingual(raw)
        self.assertIn("🎵", normalized)
        self.assertIn("🎶", normalized)
        self.assertIn("🎉", normalized)
        # Verify space separation
        tokens = normalized.split()
        self.assertIn("🎵", tokens)
        self.assertIn("🎶", tokens)
        self.assertIn("🎉", tokens)

    def test_indic_danda_isolation(self):
        raw = "सत्यमेव जयते।शुभकामनाएं॥"
        normalized = clean_and_normalize_multilingual(raw)
        tokens = normalized.split()
        self.assertIn("।", tokens)
        self.assertIn("॥", tokens)


if __name__ == "__main__":
    unittest.main()
