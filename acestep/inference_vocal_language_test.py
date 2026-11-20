"""Tests for propagating the LM-detected vocal language into DiT conditioning."""

import unittest

from acestep.inference import (
    _is_unset_vocal_language,
    _lm_vocal_language,
    _update_metadata_from_lm,
)


class LmVocalLanguageTests(unittest.TestCase):
    """The LM CoT names the field ``language``; it must reach the DiT inputs."""

    def test_reads_language_key_emitted_by_lm(self):
        self.assertEqual(_lm_vocal_language({"language": "en"}), "en")

    def test_accepts_legacy_vocal_language_key(self):
        self.assertEqual(_lm_vocal_language({"vocal_language": "zh"}), "zh")

    def test_unknown_or_missing_is_none(self):
        for meta in ({}, {"language": "unknown"}, {"language": ""}, {"language": "N/A"}):
            self.assertIsNone(_lm_vocal_language(meta), meta)

    def test_unset_detection(self):
        for v in (None, "", "unknown", "Unknown", " unknown "):
            self.assertTrue(_is_unset_vocal_language(v), v)
        self.assertFalse(_is_unset_vocal_language("ja"))

    def _update(self, user_language, metadata):
        out = _update_metadata_from_lm(
            metadata=metadata, bpm=None, key_scale="", time_signature="",
            audio_duration=30.0, vocal_language=user_language, caption="c", lyrics="l",
        )
        return out[4]

    def test_lm_language_fills_unknown_user_language(self):
        self.assertEqual(self._update("unknown", {"language": "en"}), "en")

    def test_lm_language_fills_empty_user_language(self):
        self.assertEqual(self._update("", {"language": "ja"}), "ja")

    def test_user_language_wins_over_lm(self):
        self.assertEqual(self._update("zh", {"language": "en"}), "zh")

    def test_unknown_lm_language_keeps_user_value(self):
        self.assertEqual(self._update("unknown", {"language": "unknown"}), "unknown")


if __name__ == "__main__":
    unittest.main()
