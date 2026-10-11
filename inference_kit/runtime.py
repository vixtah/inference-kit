"""Standalone storage and explicit process permission, replaceable by host adapters."""
import os
import sqlite3
import threading
from pathlib import Path

_local = threading.local()


def get_db():
    path = Path(os.environ.get("INFERENCE_DB_PATH", str(
        Path.home() / ".local/share/automations/inference.db")))
    if getattr(_local, "path", None) != path:
        path.parent.mkdir(parents=True, exist_ok=True)
        from inference_kit.inference_queue import register_schema
        conn = sqlite3.connect(path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        register_schema(conn)
        conn.commit()
        _local.conn, _local.path = conn, path
    return _local.conn


def require_production_runtime(action):
    if os.environ.get("INFERENCE_MODEL_WORKER") != "1":
        raise RuntimeError(f"{action}: model workers must be explicitly enabled")
