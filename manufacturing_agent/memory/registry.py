from __future__ import annotations

import sqlite3
import uuid
import datetime as _dt
from contextlib import closing

from manufacturing_agent.config import LONGTERM_DB, CHECKPOINT_DB

# ConversationStore가 LONGTERM_DB 안에 두는 user_id/thread_id 종속 테이블들.
# delete_user / delete_thread 시 CASCADE 정리 대상.
_CONVERSATION_TABLES = ("turns", "summaries", "diagnosis_contexts", "context_state")
# LangGraph SqliteSaver가 CHECKPOINT_DB 안에 두는 thread_id 종속 테이블들.
_CHECKPOINT_TABLES = ("checkpoints", "writes")


class UserThreadRegistry:
    """user_id / thread_id 의 생성·조회·삭제(CASCADE)를 담당하는 가벼운 레지스트리.

    LONGTERM_DB(SQLite)에 users / threads 테이블을 만들고,
    삭제 시 ConversationStore(같은 DB) 및 LangGraph 체크포인트(CHECKPOINT_DB) 행까지 함께 정리한다.
    """

    def __init__(self, db_path: str = LONGTERM_DB, checkpoint_db: str = CHECKPOINT_DB):
        self.db_path = db_path
        self.checkpoint_db = checkpoint_db
        with closing(sqlite3.connect(self.db_path)) as c, c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS users(
                    user_id TEXT PRIMARY KEY,
                    created_at TEXT);
                CREATE TABLE IF NOT EXISTS threads(
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TEXT,
                    title TEXT);
                """
            )

    @staticmethod
    def _now() -> str:
        return _dt.datetime.now().isoformat(timespec="seconds")

    # --- users ---
    def create_user(self, user_id: str | None = None) -> dict:
        if user_id is None:
            user_id = uuid.uuid4().hex
        with closing(sqlite3.connect(self.db_path)) as c, c:
            row = c.execute(
                "SELECT user_id, created_at FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
            if row is not None:
                # idempotent: 이미 있으면 기존 값을 그대로 반환
                return {"user_id": row[0], "created_at": row[1]}
            created_at = self._now()
            c.execute(
                "INSERT INTO users(user_id, created_at) VALUES(?, ?)",
                (user_id, created_at),
            )
        return {"user_id": user_id, "created_at": created_at}

    def user_exists(self, user_id: str) -> bool:
        with closing(sqlite3.connect(self.db_path)) as c, c:
            row = c.execute(
                "SELECT 1 FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
        return row is not None

    def delete_user(self, user_id: str) -> bool:
        """user 및 그에 속한 모든 thread / 대화 / 체크포인트 행을 CASCADE 삭제."""
        with closing(sqlite3.connect(self.db_path)) as c, c:
            existed = (
                c.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone()
                is not None
            )
            thread_ids = [
                r[0]
                for r in c.execute(
                    "SELECT thread_id FROM threads WHERE user_id=?", (user_id,)
                ).fetchall()
            ]
            c.execute("DELETE FROM threads WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM users WHERE user_id=?", (user_id,))
            for table in _CONVERSATION_TABLES:
                try:
                    c.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
                except sqlite3.OperationalError:
                    # 해당 테이블이 아직 없을 수 있음(방어적).
                    pass
        for tid in thread_ids:
            self._delete_checkpoints(tid)
        return existed

    # --- threads ---
    def create_thread(self, user_id: str, thread_id: str | None = None, title: str = "") -> dict:
        if not self.user_exists(user_id):
            raise ValueError("user_not_found")
        if thread_id is None:
            thread_id = uuid.uuid4().hex
        with closing(sqlite3.connect(self.db_path)) as c, c:
            row = c.execute(
                "SELECT thread_id, user_id, created_at, title FROM threads WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
            if row is not None:
                # idempotent: 동일 thread_id가 있으면 기존 값을 반환
                return {
                    "thread_id": row[0],
                    "user_id": row[1],
                    "created_at": row[2],
                    "title": row[3] or "",
                }
            created_at = self._now()
            c.execute(
                "INSERT INTO threads(thread_id, user_id, created_at, title) VALUES(?, ?, ?, ?)",
                (thread_id, user_id, created_at, title),
            )
        return {
            "thread_id": thread_id,
            "user_id": user_id,
            "created_at": created_at,
            "title": title,
        }

    def thread_exists(self, user_id: str, thread_id: str) -> bool:
        with closing(sqlite3.connect(self.db_path)) as c, c:
            row = c.execute(
                "SELECT 1 FROM threads WHERE thread_id=? AND user_id=?",
                (thread_id, user_id),
            ).fetchone()
        return row is not None

    def list_threads(self, user_id: str) -> list[dict]:
        with closing(sqlite3.connect(self.db_path)) as c, c:
            rows = c.execute(
                "SELECT thread_id, user_id, created_at, title FROM threads "
                "WHERE user_id=? ORDER BY created_at DESC, rowid DESC",
                (user_id,),
            ).fetchall()
        return [
            {
                "thread_id": r[0],
                "user_id": r[1],
                "created_at": r[2],
                "title": r[3] or "",
            }
            for r in rows
        ]

    def delete_thread(self, user_id: str, thread_id: str) -> bool:
        with closing(sqlite3.connect(self.db_path)) as c, c:
            existed = (
                c.execute(
                    "SELECT 1 FROM threads WHERE thread_id=? AND user_id=?",
                    (thread_id, user_id),
                ).fetchone()
                is not None
            )
            if not existed:
                return False
            c.execute(
                "DELETE FROM threads WHERE thread_id=? AND user_id=?",
                (thread_id, user_id),
            )
            for table in _CONVERSATION_TABLES:
                try:
                    c.execute(
                        f"DELETE FROM {table} WHERE user_id=? AND thread_id=?",
                        (user_id, thread_id),
                    )
                except sqlite3.OperationalError:
                    pass
        self._delete_checkpoints(thread_id)
        return True

    # --- helpers ---
    def _delete_checkpoints(self, thread_id: str) -> None:
        """CHECKPOINT_DB에서 thread_id에 묶인 LangGraph 체크포인트 행을 삭제.

        SqliteSaver 테이블이 아직 생성되지 않았을 수 있으므로 테이블별로 try/except.
        """
        with closing(sqlite3.connect(self.checkpoint_db)) as c, c:
            for table in _CHECKPOINT_TABLES:
                try:
                    c.execute(f"DELETE FROM {table} WHERE thread_id=?", (thread_id,))
                except sqlite3.OperationalError:
                    pass


registry = UserThreadRegistry()
