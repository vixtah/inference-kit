"""JSON-lines option scorer, run only with the isolated MLX interpreter.

Each request is a chat prompt plus one label per option. The worker runs a
single forward pass (no generation) and returns the softmax over the first
token of each label, renormalised across the options.
"""

import json
import sys
import time

import mlx.core as mx
from mlx_lm import load


def main():
    try:
        model, tokenizer = load(sys.argv[1])
    except Exception as exc:
        print(
            json.dumps({"ready": False, "error": type(exc).__name__ + ": " + str(exc)}), flush=True
        )
        return
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            start = time.perf_counter()
            prompt = tokenizer.apply_chat_template(
                req["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in req["labels"]]
            if len(set(ids)) != len(ids):
                raise ValueError("Option labels must start with distinct tokens")
            logits = model(mx.array(prompt)[None])[0, -1, :].astype(mx.float32)
            picked = mx.softmax(logits[mx.array(ids)], axis=-1)
            mx.eval(picked)
            print(
                json.dumps(
                    {
                        "probabilities": picked.tolist(),
                        "inputTokens": len(prompt),
                        "ms": (time.perf_counter() - start) * 1000,
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            print(json.dumps({"error": type(exc).__name__ + ": " + str(exc)}), flush=True)


if __name__ == "__main__":
    main()
