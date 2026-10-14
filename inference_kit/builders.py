"""Interchangeable coding builders: one unattended coding task in, commits on
a local git branch out.

Every builder has the same three calls:

- ``start(workdir, base_branch, branch, prompt, base_sha=...) -> Handle``:
  start the task. ``workdir`` is a clean checkout (usually a worktree) of the
  repository at ``base_sha``; ``branch`` names where the result lands.
- ``status(handle, workdir) -> Status``: ``running``, ``done`` or ``failed``
  (with an error).
- ``collect(handle, workdir, local_branch, message) -> sha``: put the finished
  result on ``local_branch`` in ``workdir`` and return its tip.

Implementations:

- ``ClaudeCloud``: a Claude Code cloud session (``claude --cloud``). The CLI
  only starts cloud sessions from a terminal, so it runs under ``script``'s
  pseudo-terminal and exits once the session exists. The session clones the
  workdir's GitHub remote and pushes ``branch``; we watch that branch with
  ``git ls-remote``. A finished push carries a per-start token in its last
  commit message, so a branch left by an earlier attempt never counts.
- ``CodexCloud``: a Codex cloud task (``codex cloud exec`` in an environment
  set up at chatgpt.com/codex). Its result is a diff (``codex cloud diff``),
  which ``collect`` applies and commits locally.
- ``CodexLocal``: today's headless ``codex exec`` in the workdir
  (``codex.run_headless``); ``start`` blocks until it ends.

Commands and their output never include credentials; errors carry only the
CLI's own short message.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from inference_kit import codex

State = Literal["running", "done", "failed"]
# (cmd, *, cwd, timeout, input, env) -> CompletedProcess-like (returncode, stdout, stderr)
Runner = Callable[..., Any]

DEFAULT_TIMEOUT = 2 * 60 * 60  # a cloud build that hasn't finished in 2 h has failed
_ANSI = re.compile(r"\x1b\[[0-9;?<>=]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[78()][0-9A-B]?")


class BuilderError(RuntimeError):
    """A builder couldn't start or collect a task (the message says why)."""


@dataclass
class Handle:
    """A started task, JSON-serialisable (``to_dict`` / ``from_dict``) so the
    caller can store it and poll from another process."""

    builder: str
    id: str
    branch: str
    base_branch: str
    base_sha: str | None
    started_at: float
    url: str | None = None
    token: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> Handle:
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


@dataclass
class Status:
    state: State
    error: str | None = None


class Builder(Protocol):
    name: str

    def start(self, workdir: Path, base_branch: str, branch: str, prompt: str, *,
              base_sha: str | None = None) -> Handle: ...

    def status(self, handle: Handle, workdir: Path) -> Status: ...

    def collect(self, handle: Handle, workdir: Path, local_branch: str, message: str) -> str: ...


def _run(cmd: list[str], *, cwd: Path, timeout: float, input: str | None = None,
         env: dict | None = None):
    try:
        return subprocess.run(cmd, cwd=cwd, input=input, capture_output=True, text=True,
                              timeout=timeout, env=env, check=False)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"timed out after {timeout:g}s")
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def _run_in_tty(cmd: list[str], *, cwd: Path, timeout: float, input: str | None = None,
                env: dict | None = None):
    """Run ``cmd`` under ``script``'s pseudo-terminal. Its stdin stays open
    (an EOF would reach the program as a keypress) until it exits; output goes
    through a file so a long transcript can't fill a pipe."""
    with tempfile.TemporaryFile("w+", errors="replace") as out:
        try:
            proc = subprocess.Popen(["script", "-q", "/dev/null", *cmd], cwd=cwd, env=env,
                                    stdin=subprocess.PIPE, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        except OSError as exc:
            return subprocess.CompletedProcess(cmd, 127, "", str(exc))
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            code = 124
        finally:
            if proc.stdin:
                proc.stdin.close()
        out.seek(0)
        return subprocess.CompletedProcess(cmd, code, out.read(), "")


def clean_output(text: str, limit: int = 300) -> str:
    """Terminal output as one short plain line (no escape codes)."""
    text = _ANSI.sub("", text or "").replace("\r", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " · ".join(lines)[-limit:]


def _git(runner: Runner, workdir: Path, *args: str, timeout: float = 120, input=None) -> str:
    proc = runner(["git", *args], cwd=workdir, timeout=timeout, input=input)
    if proc.returncode != 0:
        raise BuilderError(f"git {args[0]} failed: {clean_output(proc.stderr or proc.stdout)}")
    return (proc.stdout or "").strip()


def _resolve(name: str, env_var: str) -> str:
    return os.environ.get(env_var) or shutil.which(name) or name


# Appended to the caller's prompt: where the work goes and how we know it's finished.
CLAUDE_EPILOGUE = """

## Delivering the result (required)

- Work on a branch named `{branch}` created from commit `{base_sha}`:
  `git fetch {remote} && git checkout -B {branch} {base_sha}`.
- Commit your change there. Do not open a pull request, and never push to any
  other branch.
- When everything is committed, add one last commit (it may be empty:
  `git commit --allow-empty`) whose message ends with this line, exactly:
  `{trailer}`
  then push once: `git push -f {remote} {branch}`.
- If you cannot finish, push nothing to `{branch}`. Instead create a branch
  `{failed_branch}` from `{base_sha}` with one empty commit whose message is
  `builder failed: <one line saying why>` followed by the line `{trailer}`,
  and push it: `git push -f {remote} {failed_branch}`.
"""


class ClaudeCloud:
    """A Claude Code cloud session per task (see the module docstring).

    The workdir must be a worktree of a checkout Claude Code already trusts
    (run ``claude`` in the main checkout once) and its ``remote`` must be on
    GitHub, reachable from claude.ai (Claude GitHub App or ``/web-setup``).
    """

    name = "claude"

    def __init__(self, *, remote: str = "origin", timeout: float = DEFAULT_TIMEOUT,
                 launch_timeout: float = 180, claude_bin: str | None = None,
                 runner: Runner = _run, tty_runner: Runner = _run_in_tty,
                 clock: Callable[[], float] = time.time):
        self.remote = remote
        self.timeout = timeout
        self.launch_timeout = launch_timeout
        self.claude_bin = claude_bin or _resolve("claude", "CLAUDE_BIN")
        self.runner, self.tty_runner, self.clock = runner, tty_runner, clock

    def failed_branch(self, branch: str) -> str:
        return f"{branch}-failed"

    def start(self, workdir: Path, base_branch: str, branch: str, prompt: str, *,
              base_sha: str | None = None) -> Handle:
        base_sha = base_sha or _git(self.runner, workdir, "rev-parse", "HEAD")
        token = secrets.token_hex(8)
        full = prompt.rstrip() + CLAUDE_EPILOGUE.format(
            branch=branch, base_sha=base_sha, remote=self.remote, trailer=self.trailer(token),
            failed_branch=self.failed_branch(branch))
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        proc = self.tty_runner([self.claude_bin, "--cloud", full], cwd=workdir,
                               timeout=self.launch_timeout, env=env)
        out = _ANSI.sub("", proc.stdout or "")
        if "trust this folder" in out:
            raise BuilderError(
                f"Claude Code asks whether to trust {workdir}; run `claude` once in the "
                "repository's main checkout and trust it")
        m = re.search(r"\b(session_[A-Za-z0-9]+)\b", out)
        if not m:
            raise BuilderError(f"claude --cloud started no session (exit {proc.returncode}): "
                               f"{clean_output(out)}")
        return Handle(builder=self.name, id=m[1], branch=branch, base_branch=base_branch,
                      base_sha=base_sha, started_at=self.clock(), token=token,
                      url=f"https://claude.ai/code/{m[1]}")

    @staticmethod
    def trailer(token: str) -> str:
        return f"Builder-Done: {token}"

    def _remote_heads(self, workdir: Path, *branches: str) -> dict[str, str]:
        out = _git(self.runner, workdir, "ls-remote", self.remote,
                   *[f"refs/heads/{b}" for b in branches])
        heads = {}
        for line in out.splitlines():
            sha, _, ref = line.partition("\t")
            heads[ref.removeprefix("refs/heads/")] = sha
        return heads

    def _fetch(self, workdir: Path, branch: str) -> str:
        _git(self.runner, workdir, "fetch", "--no-tags", self.remote,
             f"+refs/heads/{branch}:refs/remotes/{self.remote}/{branch}", timeout=300)
        return _git(self.runner, workdir, "rev-parse", f"refs/remotes/{self.remote}/{branch}")

    def _message(self, workdir: Path, sha: str) -> str:
        return _git(self.runner, workdir, "log", "-1", "--format=%B", sha)

    def status(self, handle: Handle, workdir: Path) -> Status:
        failed = self.failed_branch(handle.branch)
        trailer = self.trailer(handle.token)
        heads = self._remote_heads(workdir, handle.branch, failed)
        if failed in heads:
            msg = self._message(workdir, self._fetch(workdir, failed))
            if trailer in msg:
                why = msg.replace(trailer, "").strip().removeprefix("builder failed:").strip()
                return Status("failed", f"the cloud session gave up: {why[:300] or 'no reason'}")
        if handle.branch in heads and trailer in self._message(
                workdir, self._fetch(workdir, handle.branch)):
            return Status("done")
        if self.clock() - handle.started_at > self.timeout:
            return Status("failed", f"no finished branch {handle.branch} after "
                                    f"{self.timeout / 60:.0f} minutes")
        return Status("running")

    def collect(self, handle: Handle, workdir: Path, local_branch: str, message: str) -> str:
        sha = self._fetch(workdir, handle.branch)
        if handle.base_sha:
            ok = self.runner(["git", "merge-base", "--is-ancestor", handle.base_sha, sha],
                             cwd=workdir, timeout=60)
            if ok.returncode != 0:
                raise BuilderError(f"{handle.branch} doesn't start from {handle.base_sha[:12]}")
        _git(self.runner, workdir, "checkout", "-B", local_branch, sha)
        return sha


_TASK_ID = re.compile(r"\b(task_[A-Za-z0-9_]+)\b")
_TASK_STATUS = re.compile(r"^\[(PENDING|READY|APPLIED|ERROR)\]\s*(.*)$", re.MULTILINE)


class CodexCloud:
    """A Codex cloud task per request, in a Codex environment (created once
    per repository at chatgpt.com/codex/settings/environments). ``env`` is the
    environment's id or its label."""

    name = "codex"

    def __init__(self, *, env: str, timeout: float = DEFAULT_TIMEOUT,
                 codex_bin: str | None = None, runner: Runner = _run,
                 clock: Callable[[], float] = time.time):
        if not env:
            raise BuilderError("no Codex cloud environment configured")
        self.env = env
        self.timeout = timeout
        self._bin = codex_bin
        self.runner, self.clock = runner, clock

    @property
    def codex_bin(self) -> str:
        return self._bin or codex.resolve_codex_bin()

    def _codex(self, workdir: Path, *args: str, timeout: float = 120):
        env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY",)}
        return self.runner([self.codex_bin, "cloud", *args], cwd=workdir, timeout=timeout,
                           env=env)

    def start(self, workdir: Path, base_branch: str, branch: str, prompt: str, *,
              base_sha: str | None = None) -> Handle:
        base_sha = base_sha or _git(self.runner, workdir, "rev-parse", "HEAD")
        proc = self._codex(workdir, "exec", "--env", self.env, "--branch", base_branch, prompt)
        m = _TASK_ID.search(proc.stdout or "")
        if proc.returncode != 0 or not m:
            raise BuilderError(f"codex cloud exec failed (exit {proc.returncode}): "
                               f"{clean_output(proc.stderr or proc.stdout)}")
        url = next((u for u in (proc.stdout or "").split() if u.startswith("https://")), None)
        return Handle(builder=self.name, id=m[1], branch=branch, base_branch=base_branch,
                      base_sha=base_sha, started_at=self.clock(), url=url)

    def status(self, handle: Handle, workdir: Path) -> Status:
        # ``status`` exits 1 whenever the task isn't READY: read the line, not the code.
        proc = self._codex(workdir, "status", handle.id)
        m = _TASK_STATUS.search(_ANSI.sub("", proc.stdout or ""))
        if m and m[1] in ("READY", "APPLIED"):
            return Status("done")
        if m and m[1] == "ERROR":
            return Status("failed", f"the Codex cloud task failed: {m[2][:200]}")
        if self.clock() - handle.started_at > self.timeout:
            why = "" if m else f" (last status check: {clean_output(proc.stderr or proc.stdout)})"
            return Status("failed", f"the Codex cloud task didn't finish in "
                                    f"{self.timeout / 60:.0f} minutes{why}")
        return Status("running")

    def collect(self, handle: Handle, workdir: Path, local_branch: str, message: str) -> str:
        proc = self._codex(workdir, "diff", handle.id, timeout=300)
        if proc.returncode != 0:
            raise BuilderError(f"codex cloud diff failed: {clean_output(proc.stderr or proc.stdout)}")
        diff = proc.stdout or ""
        if not diff.strip():
            raise BuilderError("the Codex cloud task changed nothing")
        _git(self.runner, workdir, "checkout", "-B", local_branch, handle.base_sha or "HEAD")
        _git(self.runner, workdir, "apply", "--index", "--whitespace=nowarn", "-", input=diff)
        _git(self.runner, workdir, "commit", "-q", "-m",
             f"{message.rstrip()}\n\nBuilt by Codex cloud task {handle.url or handle.id}")
        return _git(self.runner, workdir, "rev-parse", "HEAD")


class CodexLocal:
    """Headless ``codex exec`` in the workdir (``codex.run_headless``). The
    workdir must already be on the result branch; ``start`` waits for the run,
    and its result (``exit_code``, ``timed_out``, ``usage_limited``,
    ``log_tail``) is in ``handle.data["run"]``."""

    name = "local"

    def __init__(self, *, log_path: Path, writable_roots: list[str], model: str,
                 reasoning_effort: str = "medium", timeout: int = 45 * 60,
                 env_extra: dict[str, str] | None = None,
                 run: Callable[..., dict] | None = None, runner: Runner = _run,
                 clock: Callable[[], float] = time.time):
        self.log_path, self.writable_roots = log_path, writable_roots
        self.model, self.reasoning_effort, self.timeout = model, reasoning_effort, timeout
        self.env_extra = env_extra
        self._run = run or codex.run_headless
        self.runner, self.clock = runner, clock

    def start(self, workdir: Path, base_branch: str, branch: str, prompt: str, *,
              base_sha: str | None = None) -> Handle:
        started = self.clock()
        result = self._run(prompt, cwd=workdir, log_path=self.log_path,
                           writable_roots=self.writable_roots, model=self.model,
                           reasoning_effort=self.reasoning_effort, timeout=self.timeout,
                           env_extra=self.env_extra)
        return Handle(builder=self.name, id=f"local-{int(started)}", branch=branch,
                      base_branch=base_branch, base_sha=base_sha, started_at=started,
                      data={"run": result, "log": str(self.log_path)})

    def status(self, handle: Handle, workdir: Path) -> Status:
        run = handle.data.get("run") or {}
        if run.get("usage_limited"):
            return Status("failed", "Codex hit its usage limit")
        return Status("done")

    def collect(self, handle: Handle, workdir: Path, local_branch: str, message: str) -> str:
        return _git(self.runner, workdir, "rev-parse", "HEAD")


__all__ = [
    "CLAUDE_EPILOGUE",
    "DEFAULT_TIMEOUT",
    "Builder",
    "BuilderError",
    "ClaudeCloud",
    "CodexCloud",
    "CodexLocal",
    "Handle",
    "Status",
    "clean_output",
]
