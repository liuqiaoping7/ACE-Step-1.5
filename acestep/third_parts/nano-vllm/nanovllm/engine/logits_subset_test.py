"""Tests for audio-code vocab-row restricted logits (lm_head subset)."""

import os
import types
import unittest

import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.models.qwen3 import Qwen3ForCausalLM


class _Proc:
    def __init__(self, ids, state="CODES_GENERATION", eos=7, enabled=True):
        self.audio_code_token_ids = set(ids)
        self.state = types.SimpleNamespace(name=state)
        self.eos_token_id = eos
        self.enabled = enabled
        self.non_audio_code_mask = torch.zeros(1)


def _runner():
    r = ModelRunner.__new__(ModelRunner)
    r.world_size = 1
    return r


class LogitsSubsetTests(unittest.TestCase):
    def test_rows_match_full_logits_bitwise(self):
        torch.manual_seed(0)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.bfloat16 if dev == "cuda" else torch.float32
        w = torch.randn(300, 64, device=dev, dtype=dt)
        x = torch.randn(2, 64, device=dev, dtype=dt)
        host = types.SimpleNamespace(lm_head=types.SimpleNamespace(weight=w))
        sub = Qwen3ForCausalLM.compute_logits_rows(host, x, 100, 250, (7,))
        full = torch.nn.functional.linear(x, w)
        self.assertTrue(torch.equal(sub[:, 100:250], full[:, 100:250]))
        self.assertTrue(torch.equal(sub[:, 7], full[:, 7]))
        self.assertEqual(int((sub[:, 250:] != 0).sum()), 0)

    def test_vocab_rows_only_in_codes_phase(self):
        r = _runner()
        seq = types.SimpleNamespace(logits_processor=_Proc(range(100, 250)))
        self.assertEqual(r._codes_vocab_rows(seq), (100, 250, (7,)))
        self.assertIsNone(r._codes_vocab_rows(types.SimpleNamespace(logits_processor=_Proc(range(100, 250), state="CAPTION_VALUE"))))
        self.assertIsNone(r._codes_vocab_rows(types.SimpleNamespace(logits_processor=None)))
        self.assertIsNone(r._codes_vocab_rows(types.SimpleNamespace(logits_processor=_Proc(range(100, 250), enabled=False))))

    def test_non_contiguous_codes_disable_subset(self):
        r = _runner()
        seq = types.SimpleNamespace(logits_processor=_Proc([1, 2, 5]))
        self.assertIsNone(r._codes_vocab_rows(seq))

    def test_env_switch(self):
        r = _runner()
        os.environ["NANOVLLM_LOGITS_SUBSET"] = "0"
        try:
            self.assertIsNone(r._codes_vocab_rows(types.SimpleNamespace(logits_processor=_Proc(range(100, 250)))))
        finally:
            del os.environ["NANOVLLM_LOGITS_SUBSET"]


if __name__ == "__main__":
    unittest.main()
