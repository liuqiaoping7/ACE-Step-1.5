"""Batch sizing happens before the LM writes per-item codes, and vllm batches never preempt.

Before: the DiT-side VRAM guard shrank the batch *after* the LM had generated codes for
every requested item, silently dropping the extra codes; and nano-vllm, when its KV cache
ran out, preempted single sequences, splitting CFG pairs (``The size of tensor a (2)
must match the size of tensor b (3)``) - e.g. 3 x 240 s on an 8 GB card.
"""

from __future__ import annotations

import types
import unittest
from typing import Any, List, Optional
from unittest.mock import MagicMock

from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler


class PlanningDitHandler:
    """Stand-in DiT handler whose planner allows only ``limit`` items."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.plan_calls: List[tuple] = []
        self.generate_kwargs: Optional[dict] = None
        self.lora_loaded = False
        self.use_lora = False
        self.lora_scale = 1.0

    def plan_batch_size(self, batch_size: int, audio_duration=None) -> int:
        self.plan_calls.append((batch_size, audio_duration))
        return min(batch_size, self.limit)

    def prepare_seeds(self, batch_size: int, seed: str, use_random_seed: bool):
        return list(range(batch_size)), None

    def generate_music(self, seed=None, use_random_seed=None, batch_size=None, **kwargs: Any):
        self.generate_kwargs = {"batch_size": batch_size, **kwargs}
        return {"success": False, "status_message": "stub", "error": "stub", "audios": [], "extra_outputs": {}}


def _llm() -> MagicMock:
    llm = MagicMock()
    llm.llm_initialized = True
    llm.generate_with_stop_condition.return_value = {"success": False, "error": "stub LM"}
    return llm


class PlanBeforeLmTests(unittest.TestCase):
    def _run(self, limit: int, batch: int, use_lm: bool = True):
        dit = PlanningDitHandler(limit)
        llm = _llm()
        params = GenerationParams(task_type="text2music", caption="synth pop", lyrics="[Verse]\nla",
                                  duration=240, thinking=use_lm, use_cot_metas=use_lm,
                                  use_cot_caption=use_lm, use_cot_language=use_lm)
        cfg = GenerationConfig(batch_size=batch, use_random_seed=False, seeds=list(range(batch)))
        generate_music(dit, llm, params, cfg)
        return dit, llm

    def test_lm_is_asked_for_the_planned_batch(self):
        dit, llm = self._run(limit=2, batch=4)
        self.assertEqual(dit.plan_calls, [(4, 240)])
        kwargs = llm.generate_with_stop_condition.call_args.kwargs
        self.assertEqual(kwargs["batch_size"], 2)
        self.assertEqual(kwargs["seeds"], [0, 1])

    def test_batch_within_budget_is_unchanged(self):
        dit, llm = self._run(limit=8, batch=4)
        self.assertEqual(llm.generate_with_stop_condition.call_args.kwargs["batch_size"], 4)

    def test_dit_receives_the_planned_batch_without_lm(self):
        """Without the LM the DiT-side guard decides; inference forwards the requested batch."""
        dit, llm = self._run(limit=2, batch=4, use_lm=False)
        self.assertEqual(dit.plan_calls, [])
        llm.generate_with_stop_condition.assert_not_called()
        self.assertEqual(dit.generate_kwargs["batch_size"], 4)


class VllmItemsPerCallTests(unittest.TestCase):
    def _handler(self, total_blocks: int, block_size: int = 256, prompt_tokens: int = 500):
        h = LLMHandler.__new__(LLMHandler)
        block_manager = types.SimpleNamespace(block_size=block_size, blocks=[None] * total_blocks)
        h.llm = types.SimpleNamespace(scheduler=types.SimpleNamespace(block_manager=block_manager))
        h.llm_tokenizer = types.SimpleNamespace(encode=lambda text: [0] * prompt_tokens)
        return h

    def test_240s_cfg_on_8gb_runs_two_at_a_time(self):
        # (500 + 1200 + 1) tokens -> 7 blocks, x2 for the CFG pair -> 14 per item; 30 // 14 = 2
        h = self._handler(total_blocks=30)
        self.assertEqual(h._vllm_items_per_call(["p"] * 3, "u", max_tokens=1200), 2)

    def test_short_songs_fit_in_one_call(self):
        h = self._handler(total_blocks=30)
        self.assertEqual(h._vllm_items_per_call(["p"] * 4, "u", max_tokens=150), 4)

    def test_without_cfg_each_item_is_one_sequence(self):
        h = self._handler(total_blocks=30)
        self.assertEqual(h._vllm_items_per_call(["p"] * 5, None, max_tokens=1200), 4)

    def test_at_least_one_item_and_unknown_engine_keeps_single_call(self):
        self.assertEqual(self._handler(total_blocks=3)._vllm_items_per_call(["p"] * 2, "u", 1200), 1)
        h = LLMHandler.__new__(LLMHandler)
        h.llm = object()
        h.llm_tokenizer = None
        self.assertEqual(h._vllm_items_per_call(["p"] * 3, "u", 1200), 3)


if __name__ == "__main__":
    unittest.main()
