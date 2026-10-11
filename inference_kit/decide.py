"""Typed decisions (yes/no, choice, score) with probabilities instead of text.

Not to be confused with ``core.decisions`` (the decision log and
shadow/cascade harness that routes named decisions between Claude and a
cheap backend); this package is a local-model scorer.

Questions and answers use TypeSafe Jev's ``/v1/systemone`` shape, so a remote
Jev backend can be added later without changing callers::

    await decide("Star Trek S4E3 recap", {
        "about_set": {"type": "noul",
                      "instructions": "Is this about the MTG set 'Star Trek'?"},
    })
    # {"about_set": {"type": "noul", "noul": 0.04}}

A local MLX model (``DECIDE_MLX_MODEL``) scores each option's label token in
one forward pass — no generation — so the distribution is the model's own.
One pass per question; runs behind the shared Ollama admission slot.

Confidence is ``1 - normalised entropy`` of the distribution: 1 when one
option holds all the mass, 0 when it is uniform.
"""

from __future__ import annotations

import json
import math

from orbit_jobs.runctx import current

from inference_kit.mlx_decide import MODEL, Scorer

__all__ = ["decide", "model", "release"]

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SYSTEM = (
    "You label content. Read the content and the question, then answer with the "
    "letter of the best option only. Follow the question and option wording literally."
)
_scorer: Scorer | None = None


async def decide(state, questions: dict[str, dict]) -> dict:
    """Answer each Jev-shaped question about ``state``; returns answers keyed by id."""
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    return {qid: await _local(text, q) for qid, q in questions.items()}


def model() -> str:
    """Name of the local model that answers questions."""
    return _scorer.model if _scorer else MODEL


async def release() -> None:
    """Stop the model process and free its memory; the next question reloads it."""
    global _scorer
    if _scorer is not None:
        scorer, _scorer = _scorer, None
        await scorer.close()


async def _local(text: str, question: dict) -> dict:
    global _scorer
    kind = question["type"]
    keys, descriptions = _options(question)
    if len(keys) > len(_LETTERS):
        raise ValueError(f"Local backend supports at most {len(_LETTERS)} options")
    labels = list(_LETTERS[: len(keys)])
    body = (
        f"Content:\n{text}\n\nQuestion: {question['instructions']}\nOptions:\n"
        + "\n".join(f"{label}. {d}" for label, d in zip(labels, descriptions))
        + "\nAnswer with the letter only."
    )
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": body}]
    _scorer = _scorer or Scorer()
    result = await _scorer.score(messages, labels)
    probs = dict(zip(keys, result["probabilities"]))
    handle = current()
    if handle is not None:
        handle.record_llm(
            prompt=body,
            response=json.dumps({k: round(p, 4) for k, p in probs.items()}),
            ms=int(result["ms"]),
            model=_scorer.model,
            label=f"decide:{kind}",
        )
    if kind == "noul":
        return {"type": "noul", "noul": probs["true"]}
    if kind == "choice":
        return {
            "type": "choice",
            "choice": max(probs, key=probs.get),
            "probabilities": probs,
            "confidence": _confidence(probs),
        }
    legend = {level: i / (len(keys) - 1) for i, level in enumerate(keys)}
    return {
        "type": "score",
        "score": sum(legend[k] * p for k, p in probs.items()),
        "legend": legend,
        "probabilities": probs,
        "confidence": _confidence(probs),
    }


def _options(question: dict) -> tuple[list[str], list[str]]:
    kind, criteria = question["type"], question.get("criteria")
    if kind == "noul":
        criteria = criteria or {}
        return ["true", "false"], [criteria.get("true", "Yes"), criteria.get("false", "No")]
    if kind == "choice":
        return list(criteria), [f"{k}: {v}" if v else k for k, v in criteria.items()]
    if kind == "score":
        if not 2 <= len(criteria) <= 10:
            raise ValueError("Score questions need 2-10 levels")
        return list(criteria), list(criteria)
    raise ValueError(f"Unknown question type: {kind}")


def _confidence(probs: dict[str, float]) -> float:
    if len(probs) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return max(0.0, 1 - entropy / math.log(len(probs)))
