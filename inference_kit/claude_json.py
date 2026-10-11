"""Tool-free, model-verified Claude completion using the existing OAuth session.

Sonnet remains the default for existing experiments; callers may pin another model.
"""

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile

MODEL = "claude-sonnet-5"

# Image mime types the CLI's stream-json input accepts as a content block
# (verified for real against the running `claude` binary — see story notes).
_IMAGE_MIMES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _image_content_block(path) -> dict | None:
    """Base64-encode one image file into an Anthropic content block, or
    ``None`` (best-effort — a caller's bad path must not fail the whole
    call) if it can't be read or isn't a recognized image type."""
    ext = os.path.splitext(str(path))[1].lower()
    mime = _IMAGE_MIMES.get(ext) or mimetypes.guess_type(str(path))[0]
    if not mime or not mime.startswith("image/"):
        return None
    try:
        with open(path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
    except OSError:
        return None
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}


def complete(
    system, payload, *, schema=None, model=MODEL, effort="low", timeout=120, max_turns=3,
    images=None,
):
    schema = schema or (
        {
            "type": "object",
            "properties": {
                "annotations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_ref": {"type": "string"},
                            "fired": {"type": "boolean"},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["item_ref", "fired", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["annotations"],
            "additionalProperties": False,
        }
        if "cases" in payload
        else {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "rationale": {"type": "string"},
                "hypothesis": {"type": "string"},
            },
            "required": ["prompt", "rationale", "hypothesis"],
            "additionalProperties": False,
        }
    )
    executable = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    args = [executable, "-p"]
    stdin_text = None
    if images:
        # Images ride over stdin as a stream-json user message (the prompt
        # text becomes its leading text block) rather than the positional
        # prompt argument, which only takes plain text.
        content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        for path in images:
            block = _image_content_block(path)
            if block is not None:
                content.append(block)
        stdin_text = json.dumps(
            {"type": "user", "message": {"role": "user", "content": content}},
            ensure_ascii=False,
        ) + "\n"
        args += ["--input-format", "stream-json"]
    else:
        args.append(json.dumps(payload, ensure_ascii=False))
    args += [
        "--model",
        model,
        "--effort",
        effort,
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--setting-sources",
        "",
        "--settings",
        '{"disableAllHooks":true}',
        "--disable-slash-commands",
        "--no-session-persistence",
        "--output-format",
        "stream-json",
        "--verbose",
        "--system-prompt",
        system,
        "--json-schema",
        json.dumps(schema),
        "--max-turns",
        str(max_turns),
    ]
    with tempfile.TemporaryDirectory(prefix="loop-judge-") as cwd:
        result = subprocess.run(
            args, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout,
            input=stdin_text,
        )
    events = []
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    envelope = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result.returncode or (envelope and envelope.get("is_error")):
        failure = envelope or {}
        detail = failure.get("errors") or failure.get("result") or result.stderr.strip()
        subtype = failure.get("subtype") or f"exit {result.returncode}"
        raise RuntimeError(f"{model} request failed ({subtype}): {str(detail)[-500:]}")
    if envelope is None:
        raise RuntimeError(f"{model} returned no completion result")
    actual = {
        e["message"]["model"]
        for e in events
        if e.get("type") == "assistant" and e.get("message", {}).get("model")
    }
    if actual != {model}:
        raise RuntimeError("Unexpected judge model: " + str(actual))
    text = envelope["result"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    value = envelope.get("structured_output")
    if value is None:
        value = json.loads(text)
    return value, {
        "model": model,
        "costUsd": envelope.get("total_cost_usd"),
        "usage": envelope.get("usage"),
    }
