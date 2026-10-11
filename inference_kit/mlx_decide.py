"""Long-lived local MLX option scorer behind the shared Ollama admission slot."""

import asyncio
import json
import os
from pathlib import Path

from inference_kit import inference_queue
from inference_kit import runtime as agent_runtime

PYTHON = Path(
    os.environ.get("MLX_PYTHON", str(Path.home() / ".local/share/automations/mlx-env/bin/python"))
)
MODEL = os.environ.get("DECIDE_MLX_MODEL", "mlx-community/Qwen3.5-9B-4bit")
_TIMEOUT = 60


class Scorer:
    """One model subprocess; requests are serialised and admitted like Ollama calls."""

    def __init__(self, model: str = MODEL):
        self.model = model
        self.process = None
        self._lock = asyncio.Lock()

    async def score(self, messages: list[dict], labels: list[str]) -> dict:
        """Return ``{"probabilities": [...], "inputTokens", "ms"}`` for ``labels``."""
        async with self._lock:
            async with inference_queue.slot(
                endpoint="http://localhost:11434",  # Same local GPU as Ollama; shared admission.
                model=self.model,
                label="decide",
            ):
                try:
                    async with asyncio.timeout(_TIMEOUT):
                        if self.process is None or self.process.returncode is not None:
                            await self._start()
                        request = json.dumps({"messages": messages, "labels": labels})
                        self.process.stdin.write((request + "\n").encode())
                        await self.process.stdin.drain()
                        line = await self.process.stdout.readline()
                except BaseException:
                    await self.close()
                    raise
        if not line:
            await self.close()
            raise RuntimeError("MLX decide process exited")
        result = json.loads(line)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    async def _start(self):
        agent_runtime.require_production_runtime("start local decide model worker")
        self.process = await asyncio.create_subprocess_exec(
            str(PYTHON),
            "-I",
            str(Path(__file__).with_name("mlx_decide_worker.py")),
            self.model,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
        line = await self.process.stdout.readline()
        loaded = json.loads(line) if line else {}
        if not loaded.get("ready"):
            raise RuntimeError(loaded.get("error") or "MLX decide model did not load")

    async def close(self):
        if self.process and self.process.returncode is None:
            agent_runtime.require_production_runtime("stop local decide model worker")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None
