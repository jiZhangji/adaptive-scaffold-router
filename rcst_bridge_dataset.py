#!/usr/bin/env python3
"""RL dataset that exposes a verified subproblem as counterfactual context.

The policy rollout and actor update still use the original root prompt.  The
extra ``bridge_prompt_ids`` field is consumed only by the bridge-advantage
trainer to score the *same response* under root+subproblem context.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

# VERL loads custom datasets directly from their file path.  In that mode the
# file's parent directory is not guaranteed to be present in sys.path, so make
# sibling project modules importable before resolving bridge_prompt_utils.
MODULE_DIR = str(Path(__file__).resolve().parent)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from bridge_prompt_utils import build_bridge_messages
from verl.utils.dataset.rl_dataset import RLHFDataset


class RCSTBridgeDataset(RLHFDataset):
    def __getitem__(self, item):
        source_row = self.dataframe[int(item)]
        original_messages = copy.deepcopy(source_row[self.prompt_key])
        row = super().__getitem__(item)
        extra = row.get("extra_info") or {}
        subproblem = str(extra.get("bridge_subproblem") or "").strip()
        answer = str(extra.get("bridge_subproblem_answer") or "").strip()
        enabled = bool(extra.get("bridge_enabled", False) and subproblem)

        if enabled:
            messages = build_bridge_messages(original_messages, subproblem, answer)
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_ids) > self.max_prompt_length:
                if self.truncation == "left":
                    prompt_ids = prompt_ids[-self.max_prompt_length :]
                elif self.truncation == "right":
                    prompt_ids = prompt_ids[: self.max_prompt_length]
                elif self.truncation == "middle":
                    left = self.max_prompt_length // 2
                    prompt_ids = prompt_ids[:left] + prompt_ids[-(self.max_prompt_length - left) :]
                else:
                    raise RuntimeError(
                        f"Bridge prompt length {len(prompt_ids)} exceeds {self.max_prompt_length}"
                    )
        else:
            prompt_ids = []

        row["bridge_prompt_ids"] = prompt_ids
        row["bridge_enabled"] = enabled
        return row
