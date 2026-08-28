#!/usr/bin/env python3
"""Prompt helpers for system-preserving subproblem counterfactuals."""

from __future__ import annotations

import copy
from typing import Any


BRIDGE_PROMPT_VERSION = "system_preserving_bridge_v2"


def bridge_suffix(subproblem: str, answer: str) -> str:
    subproblem = str(subproblem).strip()
    answer = str(answer).strip()
    if not subproblem:
        raise ValueError("subproblem must be non-empty")
    return (
        f"\n\nVerified prerequisite subproblem: {subproblem}\n"
        f"Prerequisite answer: {answer}\n\n"
        "Now solve the original problem independently."
    )


def build_bridge_messages(
    original_messages: list[dict[str, Any]], subproblem: str, answer: str
) -> list[dict[str, Any]]:
    """Append support to the final user turn without changing other messages.

    Keeping the system prompt and all prior turns identical is essential: the
    resulting log-probability difference should isolate the added subproblem,
    rather than confounding it with a different chat template.
    """

    messages = copy.deepcopy(list(original_messages))
    suffix = bridge_suffix(subproblem, answer)
    for message in reversed(messages):
        if str(message.get("role", "")) != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            raise TypeError("bridge prompts currently require text-only user content")
        message["content"] = content.rstrip() + suffix
        return messages
    raise ValueError("original prompt does not contain a user message")
