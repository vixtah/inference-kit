"""Claude (Anthropic API) as an alternative reviewer to Astra.

One schema-guided completion via the official SDK: adaptive thinking,
effort control, a Pydantic output model as the requested JSON schema. Returns
``(value, metadata)`` in the same shape as ``core.sources.astra_json.complete``
so callers can switch reviewers without touching the rest of the pipeline.
The value is the parsed JSON object, NOT validated against the model here:
the pipeline repairs known deviations (a duplicate game phase) before its own
strict validation, and validating first would discard the whole review. Credentials come from
the environment (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).
"""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic
from anthropic.lib._parse._transform import transform_schema  # what messages.parse applies
from pydantic import BaseModel

MODEL = "claude-opus-5"
# USD per million tokens (claude-opus-5); cache reads/writes are not modelled.
_PRICE_IN, _PRICE_OUT = 5.0, 25.0
# Output ceiling shared by adaptive thinking and the JSON answer. 16k was too
# small: at high effort the model spent most of it thinking and the report was
# cut mid-string, which the SDK's parse helper turned into a JSON error before
# stop_reason could be inspected. Streaming lets the ceiling be generous.
_MAX_TOKENS = 64000


def complete(
    system: str,
    payload: dict,
    *,
    output_model: type[BaseModel],
    model: str = MODEL,
    timeout: float = 480,
    effort: str = "high",
) -> tuple[dict, dict[str, Any]]:
    started = time.monotonic()
    client = anthropic.Anthropic(timeout=timeout, max_retries=2)
    # The schema goes in as a raw output format so the SDK does not parse the
    # text block itself; we check how the turn ended before validating.
    schema = transform_schema(output_model.model_json_schema())
    with client.messages.stream(
        model=model,
        max_tokens=_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise RuntimeError(
            f"{model} declined the request"
            + (f" ({details.category}: {details.explanation})" if details else "")
        )
    usage = response.usage
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"{model} ran out of output tokens ({usage.output_tokens} of {_MAX_TOKENS}) "
            "before finishing the review"
        )
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise ValueError(f"{model} returned no JSON")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{model} returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{model} must return a JSON object")
    cost = (usage.input_tokens * _PRICE_IN + usage.output_tokens * _PRICE_OUT) / 1_000_000
    return value, {
        "model": model,
        "provider": "anthropic",
        "reasoningEffort": effort,
        "durationSeconds": time.monotonic() - started,
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        "costUsd": round(cost, 4),
        "stopReason": response.stop_reason,
    }
