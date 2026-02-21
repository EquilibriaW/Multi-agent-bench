"""
loopbench.coordination

SQLite-backed coordination bus for planner/coder team communication.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .time_utils import now_ms


@dataclass(frozen=True)
class ClaimedTask:
    task_id: str
    phase: str
    assigned_role: str
    title: str
    payload: Dict[str, Any]
    claim_token: str


class CoordinationDB:
    """
    Small, append-auditable team protocol store:
    - tasks (assign/claim/complete lifecycle)
    - messages (top-down + peer updates)
    - claim events (history for debugging contention)
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).resolve()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    phase TEXT NOT NULL,
                    assigned_role TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','claimed','completed','failed')),
                    payload_json TEXT NOT NULL,
                    claim_token TEXT,
                    claimed_by TEXT,
                    claimed_at_ms INTEGER,
                    completed_at_ms INTEGER,
                    result_json TEXT,
                    created_at_ms INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_lookup
                ON tasks(phase, assigned_role, status)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    round_index INTEGER,
                    from_role TEXT NOT NULL,
                    to_role TEXT,
                    kind TEXT NOT NULL,
                    body_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    claim_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details_json TEXT NOT NULL
                )
                """
            )

    def seed_tasks(self, *, phase: str, subtasks: Iterable[Dict[str, Any]]) -> List[str]:
        """
        Insert new pending tasks for a phase.
        Returns the inserted task_ids.
        """
        inserted: List[str] = []
        seen: set[str] = set()
        now = self._now_ms()

        with self._connect() as conn:
            for idx, item in enumerate(subtasks, start=1):
                assigned_role = str(item.get("role") or "").strip()
                if not assigned_role:
                    continue

                raw_id = str(item.get("id") or f"{phase}_{idx}")
                task_id = f"{phase}:{raw_id}"
                while task_id in seen or self._task_exists(conn, task_id):
                    task_id = f"{task_id}:{idx}"
                seen.add(task_id)

                title = str(item.get("title") or raw_id)
                payload = dict(item)
                payload["phase"] = phase

                conn.execute(
                    """
                    INSERT INTO tasks(
                        task_id, phase, assigned_role, title, status,
                        payload_json, claim_token, claimed_by, claimed_at_ms,
                        completed_at_ms, result_json, created_at_ms
                    ) VALUES (?, ?, ?, ?, 'pending', ?, NULL, NULL, NULL, NULL, NULL, ?)
                    """,
                    (task_id, phase, assigned_role, title, json.dumps(payload), now),
                )
                inserted.append(task_id)

        return inserted

    def post_message(
        self,
        *,
        phase: str,
        from_role: str,
        to_role: Optional[str],
        kind: str,
        body: Dict[str, Any],
        round_index: Optional[int] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO messages(ts_ms, phase, round_index, from_role, to_role, kind, body_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._now_ms(),
                    phase,
                    round_index,
                    from_role,
                    to_role,
                    kind,
                    json.dumps(body),
                ),
            )
            message_id = cur.lastrowid
            return int(message_id) if message_id is not None else 0

    def claim_next_task(self, *, phase: str, role: str) -> Optional[ClaimedTask]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT task_id, phase, assigned_role, title, payload_json
                FROM tasks
                WHERE phase = ? AND assigned_role = ? AND status = 'pending'
                ORDER BY created_at_ms ASC, task_id ASC
                LIMIT 1
                """,
                (phase, role),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None

            claim_token = str(uuid.uuid4())
            now = self._now_ms()
            updated = conn.execute(
                """
                UPDATE tasks
                SET status = 'claimed', claim_token = ?, claimed_by = ?, claimed_at_ms = ?
                WHERE task_id = ? AND status = 'pending'
                """,
                (claim_token, role, now, row["task_id"]),
            )
            if updated.rowcount != 1:
                conn.execute("ROLLBACK")
                return None

            conn.execute(
                """
                INSERT INTO claims(ts_ms, task_id, role, action, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    now,
                    row["task_id"],
                    role,
                    "claimed",
                    json.dumps({"claim_token": claim_token}),
                ),
            )
            conn.execute("COMMIT")

            return ClaimedTask(
                task_id=str(row["task_id"]),
                phase=str(row["phase"]),
                assigned_role=str(row["assigned_role"]),
                title=str(row["title"]),
                payload=json.loads(str(row["payload_json"])),
                claim_token=claim_token,
            )

    def complete_task(
        self,
        *,
        task_id: str,
        role: str,
        claim_token: str,
        result: Dict[str, Any],
    ) -> bool:
        with self._connect() as conn:
            now = self._now_ms()
            updated = conn.execute(
                """
                UPDATE tasks
                SET status = 'completed', completed_at_ms = ?, result_json = ?
                WHERE task_id = ? AND status = 'claimed' AND claimed_by = ? AND claim_token = ?
                """,
                (now, json.dumps(result), task_id, role, claim_token),
            )
            ok = updated.rowcount == 1
            conn.execute(
                """
                INSERT INTO claims(ts_ms, task_id, role, action, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    now,
                    task_id,
                    role,
                    "completed" if ok else "complete_rejected",
                    json.dumps({"claim_token": claim_token}),
                ),
            )
            return ok

    def fail_task(
        self,
        *,
        task_id: str,
        role: str,
        claim_token: str,
        error: Dict[str, Any],
    ) -> bool:
        with self._connect() as conn:
            now = self._now_ms()
            updated = conn.execute(
                """
                UPDATE tasks
                SET status = 'failed', completed_at_ms = ?, result_json = ?
                WHERE task_id = ? AND status = 'claimed' AND claimed_by = ? AND claim_token = ?
                """,
                (now, json.dumps(error), task_id, role, claim_token),
            )
            ok = updated.rowcount == 1
            conn.execute(
                """
                INSERT INTO claims(ts_ms, task_id, role, action, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    now,
                    task_id,
                    role,
                    "failed" if ok else "fail_rejected",
                    json.dumps({"claim_token": claim_token}),
                ),
            )
            return ok

    def latest_messages(
        self,
        *,
        role: str,
        phase: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT message_id, ts_ms, phase, round_index, from_role, to_role, kind, body_json
                FROM messages
                WHERE phase = ? AND (to_role IS NULL OR to_role = ? OR from_role = ?)
                ORDER BY message_id DESC
                LIMIT ?
                """,
                (phase, role, role, int(limit)),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for row in reversed(rows):
            out.append(
                {
                    "message_id": int(row["message_id"]),
                    "ts_ms": int(row["ts_ms"]),
                    "phase": str(row["phase"]),
                    "round_index": row["round_index"],
                    "from_role": str(row["from_role"]),
                    "to_role": row["to_role"],
                    "kind": str(row["kind"]),
                    "body": json.loads(str(row["body_json"])),
                }
            )
        return out

    def summary(self) -> Dict[str, int]:
        with self._connect() as conn:
            counts_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS n
                FROM tasks
                GROUP BY status
                """
            ).fetchall()
            message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
            claim_count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()

        counts = {str(row["status"]): int(row["n"]) for row in counts_rows}
        return {
            "tasks_pending": counts.get("pending", 0),
            "tasks_claimed": counts.get("claimed", 0),
            "tasks_completed": counts.get("completed", 0),
            "tasks_failed": counts.get("failed", 0),
            "messages_total": int(message_count[0]) if message_count else 0,
            "claim_events_total": int(claim_count[0]) if claim_count else 0,
        }

    def _task_exists(self, conn: sqlite3.Connection, task_id: str) -> bool:
        row = conn.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return row is not None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _now_ms(self) -> int:
        return now_ms()
