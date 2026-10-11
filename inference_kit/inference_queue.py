"""Cross-process Ollama admission, with an OS lock and bounded metadata history.

One in-flight request per endpoint, shared by jobs, reflection and voice. Other
applications calling Ollama directly are outside this app-owned queue.
"""

import asyncio
import fcntl
import hashlib
import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from pathlib import Path
from urllib.parse import urlsplit

from orbit_jobs.runctx import current

from inference_kit.runtime import get_db

_active = ContextVar("inference_active", default=None)
_scope = ContextVar("inference_scope", default=(None, 0))
_STALE_SECONDS = 120


def register_schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS inference_requests (
        id TEXT PRIMARY KEY, endpoint TEXT NOT NULL, model TEXT NOT NULL,
        label TEXT, job_id INTEGER, goal_id INTEGER, priority INTEGER NOT NULL,
        status TEXT NOT NULL, created REAL NOT NULL, started REAL, finished REAL,
        heartbeat REAL NOT NULL, error TEXT)""")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS inference_active ON inference_requests(status,endpoint)"
    )


def _endpoint(url):
    p = urlsplit(url)
    host = "127.0.0.1" if p.hostname in ("localhost", "127.0.0.1", "::1") else p.hostname
    return f"{host}:{p.port or (443 if p.scheme == 'https' else 80)}"


@contextmanager
def context(goal_id, *, priority=-10):
    token = _scope.set((goal_id, priority))
    try:
        yield
    finally:
        _scope.reset(token)


def _touch(rid):
    with get_db() as conn:
        conn.execute("UPDATE inference_requests SET heartbeat=? WHERE id=?", (time.time(), rid))


async def _heartbeat(rid):
    while True:
        await asyncio.sleep(5)
        _touch(rid)


@asynccontextmanager
async def slot(*, endpoint, model, label=None, goal_id=None, priority=None):
    key = _endpoint(endpoint)
    active = _active.get()
    if (
        active
        and active[0] == key
        and active[1] is asyncio.current_task()
        and not active[2].get("released")
    ):
        yield active[2]
        return
    scoped_goal, scoped_priority = _scope.get()
    goal_id = scoped_goal if goal_id is None else goal_id
    priority = scoped_priority if priority is None else priority
    handle = current()
    conn = get_db()
    db_file = conn.execute("PRAGMA database_list").fetchone()[2]
    lock_dir = Path(db_file).parent / "inference-locks"
    lock_dir.mkdir(exist_ok=True)
    lock_path = lock_dir / (hashlib.sha256(key.encode()).hexdigest() + ".lock")
    rid, started, token, heartbeat = uuid.uuid4().hex, None, None, None
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    status, error = "cancelled", None
    created = time.time()
    try:
        with conn:
            conn.execute(
                "INSERT INTO inference_requests VALUES (?,?,?,?,?,?,?,'queued',?,NULL,NULL,?,NULL)",
                (
                    rid,
                    key,
                    model,
                    label,
                    handle.job_id if handle else None,
                    goal_id,
                    priority,
                    created,
                    created,
                ),
            )
        # async-ok: admission heartbeat, cancelled when the request leaves the queue
        heartbeat = asyncio.create_task(_heartbeat(rid))
        last_sweep = 0
        while True:
            if handle and handle.is_cancelled():
                raise asyncio.CancelledError()
            now = time.time()
            if now - last_sweep >= 5:
                with conn:
                    conn.execute(
                        "UPDATE inference_requests SET status='interrupted',finished=?,"
                        "error='Process stopped responding' "
                        "WHERE status IN ('queued','running') AND heartbeat<?",
                        (now, now - _STALE_SECONDS),
                    )
                    conn.execute("DELETE FROM inference_requests WHERE finished<?", (now - 86400,))
                last_sweep = now
            own = conn.execute(
                "SELECT status FROM inference_requests WHERE id=?", (rid,)
            ).fetchone()
            if not own or own[0] != "queued":
                raise RuntimeError("Inference admission interrupted; retry the request")
            row = conn.execute(
                "SELECT id FROM inference_requests WHERE endpoint=? AND status='queued' "
                "ORDER BY priority + CAST((?-created)/60 AS INTEGER) DESC,created,id LIMIT 1",
                (key, now),
            ).fetchone()
            if row and row[0] == rid:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    started = time.time()
                    with conn:
                        conn.execute(
                            "UPDATE inference_requests SET status='interrupted',finished=?,"
                            "error='Previous owner released admission without finishing' "
                            "WHERE endpoint=? AND status='running'",
                            (started, key),
                        )
                        conn.execute(
                            "UPDATE inference_requests SET status='running',started=?,heartbeat=? "
                            "WHERE id=?",
                            (started, started, rid),
                        )
                    break
            await asyncio.sleep(0.1)
        info = {"id": rid, "wait_ms": int((started - created) * 1000)}
        token = _active.set((key, asyncio.current_task(), info))
        try:
            yield info
            status = "succeeded"
        except Exception as exc:
            status, error = "failed", type(exc).__name__ + ": " + str(exc)[:300]
            raise
    except Exception as exc:
        status, error = "failed", type(exc).__name__ + ": " + str(exc)[:300]
        raise
    finally:
        if token is not None:
            # Closing from another context (an abandoned async generator's
            # finalizer runs aclose() in a fresh task) can't reset the var
            # the entering task still sees; the flag stops that task from
            # re-entering a slot that no longer exists.
            info["released"] = True
            try:
                _active.reset(token)
            except ValueError:
                pass
        if heartbeat:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
        try:
            with conn:
                conn.execute(
                    "UPDATE inference_requests SET status=?,finished=?,error=? WHERE id=?",
                    (status, time.time(), error, rid),
                )
        finally:
            os.close(descriptor)  # The kernel releases admission even after process termination.


def snapshot(goal_id=None):
    now = time.time()
    rows = [
        dict(r)
        for r in get_db().execute(
            "SELECT * FROM inference_requests WHERE status IN ('queued','running') OR "
            "finished>? ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 "
            "ELSE 2 END,created DESC LIMIT 200",
            (now - 3600,),
        )
    ]
    for row in rows:
        if row["status"] in ("queued", "running") and row["heartbeat"] < now - _STALE_SECONDS:
            row["status"] = "interrupted"
        row["waitSeconds"] = round((row["started"] or row["finished"] or now) - row["created"], 1)
        row["runSeconds"] = (
            round((row["finished"] or now) - row["started"], 1) if row["started"] else 0
        )
    counts = dict(
        get_db().execute(
            "SELECT status,COUNT(*) FROM inference_requests WHERE status IN "
            "('queued','running') AND heartbeat>=? GROUP BY status",
            (now - _STALE_SECONDS,),
        )
    )
    return {
        "active": [r for r in rows if r["status"] == "running"],
        "capacity": 1,
        "running": counts.get("running", 0),
        "queued": counts.get("queued", 0),
        "requests": [r for r in rows if goal_id is None or r["goal_id"] == goal_id][:20],
    }
