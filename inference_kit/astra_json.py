"""Astra-only Loop reasoning via Codex; never falls back to a student model."""

import json
import time

from inference_kit import codex

MODEL = "gpt-6-astra"


def complete(system, payload, *, schema=None, timeout=300, effort="high"):
    effort = effort or "high"
    wrapped = schema is None
    if wrapped:
        schema = {
            "type": "object",
            "properties": {"result_json": {"type": "string"}},
            "required": ["result_json"],
            "additionalProperties": False,
        }
        system += "\nEncode your complete JSON response as the result_json string."
    started = time.monotonic()
    value = codex.run_extraction(
        system,
        schema=schema,
        text=json.dumps(payload, ensure_ascii=False),
        model=MODEL,
        timeout=timeout,
        isolated=True,
        executable=codex.replay_identity()["executable"],
        **({"reasoning_effort": effort} if effort is not None else {}),
    )
    if wrapped:
        value = json.loads(value["result_json"])
    if not isinstance(value, dict):
        raise ValueError("Astra must return a JSON object")
    return value, {
        "model": MODEL,
        "provider": "codex",
        "reasoningEffort": effort,
        "durationSeconds": time.monotonic() - started,
        # The CLI extraction adapter does not expose token/billing telemetry.
        "usage": None,
        "costUsd": None,
    }
