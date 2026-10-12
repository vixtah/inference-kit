"""Codex CLI runner — structured extraction via the authenticated ``codex`` agent.

A thin wrapper around ``codex exec`` (OpenAI Codex CLI, logged in via ChatGPT).
Used to read documents and images that the local Ollama model can't — in
particular vision: ``codex exec -i <image>`` understands picture content
directly, with no local vision model to pull.

``run_extraction`` runs one non-interactive call: it pipes the document text
on stdin (if any), attaches image files with ``-i``, constrains the answer to
a JSON Schema with ``--output-schema``, and returns the parsed JSON object the
agent emits as its final message.

Kept dependency-free (just ``subprocess`` + ``json``) and synchronous; async
callers should wrap it in ``asyncio.to_thread`` like other ``core/sources``
blocking primitives. Under a job, the child process is registered with
``core.runctx`` so cancelling the job kills it instead of leaving it holding
the worker slot for the rest of its timeout.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from orbit_jobs import runctx

log = logging.getLogger(__name__)

CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
# Codex calls can be slow (model latency + tool overhead). Bound it so a
# wedged call can't pin a worker slot forever.
DEFAULT_TIMEOUT = int(os.environ.get("CODEX_TIMEOUT_SECONDS", "300"))


class CodexError(RuntimeError):
    """Raised when the codex CLI fails or returns unparseable output."""


def _parse_final_json(stdout: str) -> dict[str, Any]:
    """Pull the structured result out of codex stdout.

    With ``--output-schema`` the agent's final message is the JSON object,
    written to stdout (logs go to stderr). Be forgiving: try the whole blob
    first, then fall back to the last balanced ``{...}`` span.
    """
    text = stdout.strip()
    if not text:
        raise CodexError("codex produced no output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning(
            "codex stdout was not pure JSON (%d chars); falling back to last {...} scan",
            len(text),
        )
    # Fall back: scan for the last top-level JSON object in the output.
    start = text.rfind("{")
    while start != -1:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            start = text.rfind("{", 0, start)
    raise CodexError(f"could not parse JSON from codex output: {text[:500]!r}")


def run_extraction(
    instructions: str,
    *,
    schema: dict[str, Any],
    text: str | None = None,
    image_paths: list[str] | None = None,
    model: str | None = None,
    timeout: int | None = None,
    isolated: bool = False,
    executable: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Run one ``codex exec`` structured-extraction call and return the JSON.

    ``instructions`` is the prompt; ``text`` (optional) is piped on stdin and
    appears to the agent as a ``<stdin>`` block; ``image_paths`` are attached
    with ``-i``. ``schema`` constrains the response. Raises :class:`CodexError`
    on non-zero exit, timeout, or unparseable output.
    """
    image_paths = image_paths or []
    if reasoning_effort is not None and reasoning_effort not in (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    ):
        raise ValueError("Unsupported reasoning effort")
    timeout = timeout or DEFAULT_TIMEOUT

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as fh:
        json.dump(schema, fh)
        schema_path = fh.name

    # NOTE: ``-i/--image`` is variadic — it greedily consumes every following
    # bare token. So image paths must come first, with a flag (not the prompt)
    # immediately after them; the positional prompt goes last.
    cmd = [executable or CODEX_BIN, "exec"]
    for p in image_paths:
        cmd += ["-i", p]
    cmd += ["--skip-git-repo-check", "--output-schema", schema_path]
    if isolated:
        cmd += ["--ignore-user-config", "--ephemeral", "--sandbox", "read-only"]
        for setting in (
            'approval_policy="never"',
            'web_search="disabled"',
            "features.shell_tool=false",
            "features.unified_exec=false",
            "features.apps=false",
            "features.plugins=false",
            "features.image_generation=false",
            "features.multi_agent=false",
            "agents.enabled=false",
            "tools.view_image=false",
            "project_doc_max_bytes=0",
        ):
            cmd += ["-c", setting]
    else:
        cmd += ["--dangerously-bypass-approvals-and-sandbox"]
    if model:
        cmd += ["--model", model]
    if reasoning_effort is not None:
        cmd += ["-c", f'model_reasoning_effort="{reasoning_effort}"']
    cmd.append(instructions)

    try:
        with tempfile.TemporaryDirectory(prefix="prompt-replay-") as replay_dir:
            # async-ok: codex CLI child with timeout/kill; waited before return
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **({"cwd": replay_dir} if isolated else {}),
            )
            handle = runctx.current()
            if handle is not None:
                handle.register_cancel_callback(proc.kill)
            try:
                stdout, stderr = proc.communicate(input=text or "", timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
    except subprocess.TimeoutExpired as exc:
        raise CodexError(f"codex timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise CodexError(f"codex binary not found ({CODEX_BIN!r}); is the CLI installed?") from exc
    finally:
        try:
            os.unlink(schema_path)
        except OSError:
            pass

    if proc.returncode != 0:
        tail = (stderr or "").strip()[-800:]
        raise CodexError(f"codex exited {proc.returncode}: {tail}")

    return _parse_final_json(stdout)


def run_readonly_structured(
    prompt: str, *, system_prompt: str, schema: dict[str, Any], model: str,
    effort: str = "high", timeout: int = 15 * 60, cwd: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Investigate a repository without writes or network and return a schema-shaped plan.

    Uses the ChatGPT CLI login, ignores user configuration, and never falls back to
    another provider. Billing data is unavailable from this CLI output.
    """
    with tempfile.TemporaryDirectory(prefix="codex-plan-") as scratch:
        contract, output = Path(scratch) / "schema.json", Path(scratch) / "result.json"
        contract.write_text(json.dumps(schema))
        cmd = [resolve_codex_bin(), "exec", "--ignore-user-config", "--ephemeral",
               "--sandbox", "read-only", "--skip-git-repo-check",
               "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
               "-c", "features.apps=false", "-c", "features.plugins=false",
               "-C", str(cwd), "--model", model,
               "-c", f'model_reasoning_effort="{effort}"',
               "--output-schema", str(contract), "--output-last-message", str(output), "-"]
        env = {k: os.environ[k] for k in ("HOME", "PATH", "LANG", "TMPDIR") if k in os.environ}
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)

        def cancel():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        handle = runctx.current()
        if handle is not None:
            handle.register_cancel_callback(cancel)
        try:
            proc.communicate(system_prompt + "\n\n" + prompt, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            cancel()
            proc.communicate()
            raise CodexError(f"Codex planning timed out after {timeout}s") from exc
        if proc.returncode or not output.exists():
            raise CodexError("Codex planning failed or produced no structured result")
        try:
            value = json.loads(output.read_text())
        except (ValueError, OSError) as exc:
            raise CodexError("Codex planning produced an invalid structured result") from exc
        if not isinstance(value, dict):
            raise CodexError("Codex planning must return a JSON object")
        return value, {"models": [model], "cost_usd": None}


def replay_identity():
    """Pin replay to the configured model and an explicit or app-bundled runner."""
    bundled = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
    executable = (
        os.environ.get("CODEX_REPLAY_BIN")
        or os.environ.get("CODEX_BIN")
        or (str(bundled) if bundled.is_file() else CODEX_BIN)
    )
    config = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
    settings = tomllib.loads(config.read_text()) if config.is_file() else {}
    model = settings.get("model")
    stat = Path(executable).stat() if Path(executable).is_file() else None
    return {
        "model": model,
        "executable": executable,
        "runner_mtime_ns": stat.st_mtime_ns if stat else None,
        "runner_code_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


# ``codex exec`` writes this when the ChatGPT plan's quota is used up; the run
# exits quickly without doing anything.
USAGE_LIMIT_MARKERS = ("you've hit your usage limit", "usage limit reached")
_FALLBACK_CODEX = ("/opt/homebrew/bin/codex", "~/.local/bin/codex", "/usr/local/bin/codex")


def resolve_codex_bin() -> str:
    """``codex`` on PATH, else a known install location (launchd's PATH is
    minimal)."""
    found = shutil.which(CODEX_BIN)
    if found:
        return found
    for candidate in _FALLBACK_CODEX:
        path = os.path.expanduser(candidate)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise CodexError(f"codex binary not found ({CODEX_BIN!r}); is the CLI installed?")


def headless_command(
    prompt: str, *, writable_roots: list[str], model: str, reasoning_effort: str,
) -> list[str]:
    """The workspace-write ``codex exec`` command used for unattended coding
    runs (the request fixer): no approvals, network on, and write access
    only to the worktree (its cwd) plus ``writable_roots``."""
    return [
        resolve_codex_bin(), "exec",
        "--sandbox", "workspace-write",
        "-c", 'approval_policy="never"',
        "-c", "sandbox_workspace_write.network_access=true",
        "-c", f"sandbox_workspace_write.writable_roots={json.dumps(writable_roots)}",
        "--skip-git-repo-check",
        "--model", model,
        "-c", f'model_reasoning_effort="{reasoning_effort}"',
        prompt,
    ]


def run_headless(
    prompt: str,
    *,
    cwd: Path,
    log_path: Path,
    writable_roots: list[str],
    model: str,
    reasoning_effort: str = "medium",
    timeout: int = 45 * 60,
    env_extra: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one unattended ``codex exec`` coding session in ``cwd`` (its own
    process group, output to ``log_path``) and wait up to ``timeout``.

    Returns ``{"exit_code", "timed_out", "usage_limited", "log_tail"}``; never
    raises for the agent's own failure. Cancelling the job kills the group.
    ``OPENAI_API_KEY`` is removed so Codex uses the ChatGPT login.
    """
    cmd = headless_command(
        prompt, writable_roots=writable_roots, model=model, reasoning_effort=reasoning_effort,
    )
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "VIRTUAL_ENV")}
    env.update(env_extra or {})
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _kill_group(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    timed_out = False
    with log_path.open("w") as log_fh:
        # async-ok: codex CLI child in its own group, killed on job cancel; waited before return
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log_fh,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        handle = runctx.current()
        if handle is not None:
            handle.register_cancel_callback(lambda: _kill_group(proc))
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
    tail = log_path.read_text(errors="replace")[-4000:]
    return {
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "usage_limited": any(m in tail.lower() for m in USAGE_LIMIT_MARKERS),
        "log_tail": tail[-1500:],
    }
