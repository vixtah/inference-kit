# inference-kit

Shared provider implementations extracted from Tasks: structured CLI/API calls,
local option scoring, and cross-process GPU admission. Neither Tasks nor CardScanner
is imported. Product job kinds belong in their own applications.

Tasks adapters retain its telemetry database and production-runtime guard. Standalone
workers use `INFERENCE_DB_PATH` (default `~/.local/share/automations/inference.db`)
and must explicitly set `INFERENCE_MODEL_WORKER=1` to start local model processes.
The historical state directory keeps the GPU lock files shared with existing Tasks
processes. Credentials stay in the environment and the user's provider login stores.

`inference_kit.builders` runs one unattended coding task on Claude Code in the
cloud (`claude --cloud`), Codex in the cloud (`codex cloud exec`) or headless
`codex exec` here, behind one interface (`start` / `status` / `collect`); the
result always ends up as commits on a local branch.
