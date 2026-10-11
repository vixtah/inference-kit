# inference-kit

Shared provider implementations extracted from Tasks: structured CLI/API calls,
local option scoring, and cross-process GPU admission. Neither Tasks nor CardScanner
is imported. Product job kinds belong in their own applications.

Tasks adapters retain its telemetry database and production-runtime guard. Standalone
workers use `INFERENCE_DB_PATH` (default `~/.local/share/automations/inference.db`)
and must explicitly set `INFERENCE_MODEL_WORKER=1` to start local model processes.
The historical state directory keeps the GPU lock files shared with existing Tasks
processes. Credentials stay in the environment and the user's provider login stores.
