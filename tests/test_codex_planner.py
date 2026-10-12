import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_kit import codex


def test_planner_is_readonly_high_and_excludes_service_credentials(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setenv('GITHUB_TOKEN', 'not-for-model')
    monkeypatch.setenv('OPENAI_API_KEY', 'not-for-model')
    monkeypatch.setattr(codex, 'resolve_codex_bin', lambda: 'codex')
    monkeypatch.setattr(codex.runctx, 'current', lambda: None)

    def spawn(cmd, **kwargs):
        seen.update(cmd=cmd, **kwargs)
        def communicate(prompt, timeout):
            seen['prompt'] = prompt
            Path(cmd[cmd.index('--output-last-message')+1]).write_text(json.dumps({'kind':'bug'}))
            return '', ''
        return SimpleNamespace(communicate=communicate, returncode=0)

    monkeypatch.setattr(codex.subprocess, 'Popen', spawn)
    value, meta = codex.run_readonly_structured('Investigate', system_prompt='Read code',
        schema={'type':'object'}, model='gpt-6.1-sol', cwd=tmp_path)
    assert value == {'kind':'bug'} and meta['cost_usd'] is None
    assert seen['cmd'][seen['cmd'].index('--sandbox')+1] == 'read-only'
    assert 'model_reasoning_effort="high"' in seen['cmd']
    assert '--ignore-user-config' in seen['cmd']
    assert '--dangerously-bypass-approvals-and-sandbox' not in seen['cmd']
    assert not {'GITHUB_TOKEN','OPENAI_API_KEY'} & seen['env'].keys()


def test_failed_planner_does_not_fall_back_or_publish_raw_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(codex, 'resolve_codex_bin', lambda: 'codex')
    monkeypatch.setattr(codex.runctx, 'current', lambda: None)
    monkeypatch.setattr(codex.subprocess, 'Popen', lambda *a, **kw: SimpleNamespace(
        communicate=lambda *a, **kw: ('private output', 'private error'), returncode=1))
    with pytest.raises(codex.CodexError, match='^Codex planning failed or produced no structured result$'):
        codex.run_readonly_structured('Investigate', system_prompt='Read code', schema={},
                                     model='gpt-6.1-sol', cwd=tmp_path)
