"""Stateful SQLite-backed chat history database and ETL artifact checkpoint manager."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any


class ConversationalCheckpointManager:
    """Manages stateful conversational chat history and versioned ETL artifact checkpoints."""

    def __init__(self, db_path: str = "chat_history.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS etl_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    entity_name TEXT NOT NULL,
                    pyspark_code TEXT NOT NULL,
                    sparksql_code TEXT NOT NULL,
                    ci_status TEXT DEFAULT 'APPROVED',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
                """
            )
            conn.commit()

    def save_checkpoint(
        self,
        thread_id: str = "default",
        entity_name: str = "facilities",
        pyspark_code: str = "",
        sparksql_code: str = "",
        ci_status: str = "APPROVED",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save a new versioned ETL artifact checkpoint for a conversation thread.

        Automatically increments version number for the entity/thread.
        """
        clean_thread = (thread_id or "default").strip()
        clean_entity = (entity_name or "facilities").strip().lower()
        meta_json = json.dumps(metadata or {})

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT MAX(version) FROM etl_checkpoints
                WHERE thread_id = ? AND entity_name = ?
                """,
                (clean_thread, clean_entity),
            )
            row = cur.fetchone()
            current_max = row[0] if row and row[0] is not None else 0
            new_version = current_max + 1

            chk_id = f"chk-{clean_entity}-v{new_version}-{uuid.uuid4().hex[:6]}"
            now_iso = datetime.now(timezone.utc).isoformat()

            cur.execute(
                """
                INSERT INTO etl_checkpoints (
                    checkpoint_id, thread_id, version, entity_name,
                    pyspark_code, sparksql_code, ci_status, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chk_id,
                    clean_thread,
                    new_version,
                    clean_entity,
                    pyspark_code,
                    sparksql_code,
                    ci_status,
                    now_iso,
                    meta_json,
                ),
            )
            conn.commit()

        return {
            "checkpoint_id": chk_id,
            "thread_id": clean_thread,
            "version": new_version,
            "entity_name": clean_entity,
            "pyspark_code": pyspark_code,
            "sparksql_code": sparksql_code,
            "ci_status": ci_status,
            "created_at": now_iso,
            "metadata": metadata or {},
        }

    def list_checkpoints(
        self,
        thread_id: str = "default",
        entity_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """List saved ETL checkpoints sorted by version descending."""
        clean_thread = (thread_id or "default").strip()
        query = "SELECT * FROM etl_checkpoints WHERE thread_id = ?"
        params: list[Any] = [clean_thread]

        if entity_name:
            query += " AND entity_name = ?"
            params.append(entity_name.strip().lower())

        query += " ORDER BY version DESC, created_at DESC"

        results: list[dict[str, Any]] = []
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
            for r in rows:
                results.append(
                    {
                        "checkpoint_id": r["checkpoint_id"],
                        "thread_id": r["thread_id"],
                        "version": r["version"],
                        "entity_name": r["entity_name"],
                        "pyspark_code": r["pyspark_code"],
                        "sparksql_code": r["sparksql_code"],
                        "ci_status": r["ci_status"],
                        "created_at": r["created_at"],
                        "metadata": json.loads(r["metadata"] or "{}"),
                    }
                )
        return results

    def get_checkpoint(
        self,
        version_or_id: int | str,
        thread_id: str = "default",
        entity_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve a specific ETL checkpoint by checkpoint_id or version integer/string."""
        clean_thread = (thread_id or "default").strip()

        with self._get_connection() as conn:
            cur = conn.cursor()
            # If version passed as int or numeric string
            if isinstance(version_or_id, int) or (
                isinstance(version_or_id, str) and str(version_or_id).isdigit()
            ):
                ver_num = int(version_or_id)
                if entity_name:
                    cur.execute(
                        """
                        SELECT * FROM etl_checkpoints
                        WHERE thread_id = ? AND entity_name = ? AND version = ?
                        """,
                        (clean_thread, entity_name.strip().lower(), ver_num),
                    )
                else:
                    cur.execute(
                        """
                        SELECT * FROM etl_checkpoints
                        WHERE thread_id = ? AND version = ?
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (clean_thread, ver_num),
                    )
            else:
                # Retrieve by checkpoint_id
                cur.execute(
                    "SELECT * FROM etl_checkpoints WHERE checkpoint_id = ?",
                    (str(version_or_id).strip(),),
                )

            row = cur.fetchone()
            if not row:
                return None

            return {
                "checkpoint_id": row["checkpoint_id"],
                "thread_id": row["thread_id"],
                "version": row["version"],
                "entity_name": row["entity_name"],
                "pyspark_code": row["pyspark_code"],
                "sparksql_code": row["sparksql_code"],
                "ci_status": row["ci_status"],
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata"] or "{}"),
            }

    def restore_checkpoint(
        self,
        version_or_id: int | str,
        thread_id: str = "default",
        entity_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve and return checkpoint payload to restore as active conversation state."""
        return self.get_checkpoint(
            version_or_id=version_or_id, thread_id=thread_id, entity_name=entity_name
        )


_default_checkpoint_manager: ConversationalCheckpointManager | None = None


def get_checkpoint_manager(
    db_path: str | None = None,
) -> ConversationalCheckpointManager:
    """Singleton getter for ConversationalCheckpointManager."""
    global _default_checkpoint_manager
    target_path = db_path or "chat_history.db"
    if _default_checkpoint_manager is None:
        _default_checkpoint_manager = ConversationalCheckpointManager(db_path=target_path)
    elif db_path is not None and _default_checkpoint_manager.db_path != db_path:
        _default_checkpoint_manager = ConversationalCheckpointManager(db_path=db_path)
    return _default_checkpoint_manager
