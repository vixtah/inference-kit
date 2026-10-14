import subprocess
from pathlib import Path

import pytest

from inference_kit import builders
from inference_kit.builders import (
    BuilderError,
    ClaudeCloud,
    CodexCloud,
    CodexLocal,
    Handle,
)


def done(stdout="", code=0, stderr=""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


class FakeGit:
    """A scripted runner: ``responses`` maps a command prefix to a result."""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    def __call__(self, cmd, *, cwd, timeout, input=None, env=None):
        self.calls.append({"cmd": cmd, "cwd": cwd, "input": input, "env": env})
        for prefix, result in self.responses.items():
            if tuple(cmd[: len(prefix)]) == prefix:
                return result(cmd) if callable(result) else result
        return done()


WD = Path("/tmp/wt")
CREATED = ("\x1b[>4mCreated cloud session: Fix thing\r\n"
           "View: https://claude.ai/code/session_01AbC?from=cli&m=0\r\n"
           "Resume with: claude --teleport session_01AbC\r\n")


def claude(runner=None, tty=None, now=1000.0, **kw):
    clock = {"t": now}
    c = ClaudeCloud(remote="github", claude_bin="claude", runner=runner or FakeGit(),
                    tty_runner=tty or FakeGit({("claude",): done(CREATED)}),
                    clock=lambda: clock["t"], **kw)
    return c, clock


def test_claude_start_runs_in_a_terminal_and_asks_for_the_named_branch(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-for-the-cli")
    tty = FakeGit({("claude",): done(CREATED)})
    c, _ = claude(tty=tty)
    h = c.start(WD, "main", "claude/fix-7", "Fix the bug.", base_sha="abc123")
    cmd = tty.calls[0]["cmd"]
    assert cmd[:2] == ["claude", "--cloud"] and tty.calls[0]["cwd"] == WD
    prompt = cmd[2]
    assert prompt.startswith("Fix the bug.")
    assert "git checkout -B claude/fix-7 abc123" in prompt
    # the cloud clone's remote is ``origin`` even when ours is called ``github``
    assert "git fetch origin &&" in prompt and "github" not in prompt
    assert "git push -f origin claude/fix-7" in prompt and "claude/fix-7-failed" in prompt
    assert f"Builder-Done: {h.token}" in prompt and len(h.token) == 16
    assert "ANTHROPIC_API_KEY" not in tty.calls[0]["env"]
    assert (h.builder, h.id, h.url) == ("claude", "session_01AbC",
                                        "https://claude.ai/code/session_01AbC")
    assert Handle.from_dict(h.to_dict()) == h


def test_claude_start_fails_clearly_on_the_trust_prompt_or_no_session():
    c, _ = claude(tty=FakeGit({("claude",): done("Quick safety check: Yes, I trust this folder",
                                                 124)}))
    with pytest.raises(BuilderError, match="trust"):
        c.start(WD, "main", "claude/fix-7", "p", base_sha="abc")
    c, _ = claude(tty=FakeGit({("claude",): done("\x1b[31mError: not logged in\x1b[0m", 1)}))
    with pytest.raises(BuilderError, match="started no session.*Error: not logged in"):
        c.start(WD, "main", "claude/fix-7", "p", base_sha="abc")


def _remote(heads: dict, messages: dict):
    def ls_remote(cmd):
        return done("".join(f"{sha}\trefs/heads/{b}\n" for b, sha in heads.items()))

    def rev_parse(cmd):
        return done(heads[cmd[-1].split("/", 3)[-1]])

    return FakeGit({
        ("git", "ls-remote"): ls_remote,
        ("git", "rev-parse"): rev_parse,
        ("git", "log"): lambda cmd: done(messages[cmd[-1]]),
    })


def test_claude_status_running_done_failed_and_timeout():
    c, clock = claude()
    h = c.start(WD, "main", "claude/fix-7", "p", base_sha="base")
    trailer = c.trailer(h.token)

    c.runner = _remote({}, {})
    assert c.status(h, WD).state == "running"
    # An earlier attempt's branch (another token) is not this run's result.
    c.runner = _remote({"claude/fix-7": "old"}, {"old": "fix\n\nBuilder-Done: 0000"})
    assert c.status(h, WD).state == "running"
    c.runner = _remote({"claude/fix-7": "new"}, {"new": f"done\n\n{trailer}"})
    assert c.status(h, WD).state == "done"
    c.runner = _remote({"claude/fix-7-failed": "f"},
                       {"f": f"builder failed: tests need a database\n\n{trailer}"})
    st = c.status(h, WD)
    assert st.state == "failed" and "tests need a database" in st.error
    c.runner = _remote({}, {})
    clock["t"] += builders.DEFAULT_TIMEOUT + 1
    st = c.status(h, WD)
    assert st.state == "failed" and "no finished branch claude/fix-7" in st.error


def test_claude_collect_checks_out_the_branch_only_when_it_starts_from_base():
    c, _ = claude()
    h = c.start(WD, "main", "claude/fix-7", "p", base_sha="base")
    git = _remote({"claude/fix-7": "tip"}, {})
    c.runner = git
    assert c.collect(h, WD, "fix/7", "msg") == "tip"
    assert ["git", "checkout", "-B", "fix/7", "tip"] in [x["cmd"] for x in git.calls]
    git.responses[("git", "merge-base")] = done(code=1)
    with pytest.raises(BuilderError, match="doesn't start from base"):
        c.collect(h, WD, "fix/7", "msg")


def codex_cloud(responses, now=0.0):
    clock = {"t": now}
    return CodexCloud(env="automations", codex_bin="codex", runner=FakeGit(responses),
                      clock=lambda: clock["t"]), clock


def test_codex_cloud_start_status_and_collect(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-for-the-cli")
    url = "https://chatgpt.com/codex/tasks/task_e_123abc"
    statuses = iter(["[PENDING] Fix\n", "[READY] Fix\nautomations\n+3/-1"])
    cx, _ = codex_cloud({
        ("codex", "cloud", "exec"): done(url + "\n"),
        ("codex", "cloud", "status"): lambda cmd: done(next(statuses), 1),
        ("codex", "cloud", "diff"): done("diff --git a/x b/x\n"),
        ("git", "rev-parse"): done("newsha\n"),
    })
    h = cx.start(WD, "main", "codex/fix-7", "Fix it", base_sha="base")
    exec_call = cx.runner.calls[0]
    assert exec_call["cmd"] == ["codex", "cloud", "exec", "--env", "automations",
                                "--branch", "main", "Fix it"]
    assert "OPENAI_API_KEY" not in exec_call["env"]
    assert (h.id, h.url) == ("task_e_123abc", url)
    assert cx.status(h, WD).state == "running"
    assert cx.status(h, WD).state == "done"
    assert cx.collect(h, WD, "fix/7", "fix: thing (request #7)") == "newsha"
    cmds = [c["cmd"] for c in cx.runner.calls]
    assert ["git", "checkout", "-B", "fix/7", "base"] in cmds
    apply = next(c for c in cx.runner.calls if c["cmd"][:2] == ["git", "apply"])
    assert apply["input"].startswith("diff --git")
    commit = next(c["cmd"] for c in cx.runner.calls if c["cmd"][:2] == ["git", "commit"])
    assert "fix: thing (request #7)" in commit[-1] and url in commit[-1]


def test_codex_cloud_failures():
    cx, clock = codex_cloud({("codex", "cloud", "exec"): done("", 1,
                             "Error: environment 'x' not found; run `codex cloud`")})
    with pytest.raises(BuilderError, match="environment 'x' not found"):
        cx.start(WD, "main", "codex/fix-7", "p", base_sha="b")
    with pytest.raises(BuilderError, match="no Codex cloud environment"):
        CodexCloud(env="")

    cx, clock = codex_cloud({("codex", "cloud", "status"): done("[ERROR] Fix\n", 1),
                             ("codex", "cloud", "diff"): done("")})
    h = Handle("codex", "task_e_1", "codex/fix-7", "main", "b", 0.0)
    st = cx.status(h, WD)
    assert st.state == "failed" and "Codex cloud task failed" in st.error
    with pytest.raises(BuilderError, match="changed nothing"):
        cx.collect(h, WD, "fix/7", "m")

    cx, clock = codex_cloud({("codex", "cloud", "status"): done("", 1, "network down")})
    assert cx.status(h, WD).state == "running"
    clock["t"] = builders.DEFAULT_TIMEOUT + 5
    st = cx.status(h, WD)
    assert st.state == "failed" and "didn't finish" in st.error and "network down" in st.error


def test_codex_local_wraps_run_headless(tmp_path):
    seen = {}

    def run(prompt, **kw):
        seen.update(prompt=prompt, **kw)
        return {"exit_code": 0, "timed_out": False, "usage_limited": False, "log_tail": ""}

    local = CodexLocal(log_path=tmp_path / "log", writable_roots=["/tmp"], model="m",
                       reasoning_effort="high", timeout=60, env_extra={"A": "1"}, run=run,
                       runner=FakeGit({("git", "rev-parse"): done("head\n")}))
    h = local.start(tmp_path, "main", "fix/7", "Do it", base_sha="base")
    assert seen["cwd"] == tmp_path and seen["model"] == "m" and seen["env_extra"] == {"A": "1"}
    assert h.data["run"]["exit_code"] == 0 and local.status(h, tmp_path).state == "done"
    assert local.collect(h, tmp_path, "fix/7", "m") == "head"
    h.data["run"]["usage_limited"] = True
    assert local.status(h, tmp_path).state == "failed"


def test_clean_output_strips_escapes():
    assert builders.clean_output("\x1b[1mHello\x1b[0m\r\nworld\n") == "Hello · world"
