from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.contracts.context import DiagnosisContext

class ConversationStore:
    """user_id/thread_id 기준 대화와 재사용 가능한 DiagnosisContext snapshot 저장소."""

    def __init__(self, db_path: str = LONGTERM_DB):
        self.db_path = db_path
        with self._conn() as c:
            self._drop_if_legacy(c, "turns")
            self._drop_if_legacy(c, "machine_values")
            self._drop_if_legacy(c, "summaries")
            c.executescript("""
            CREATE TABLE IF NOT EXISTS turns(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, thread_id TEXT, role TEXT, content TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS machine_values(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, thread_id TEXT, name TEXT, value TEXT, unit TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS summaries(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, thread_id TEXT, kind TEXT, content TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS diagnosis_contexts(
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                features_json TEXT NOT NULL,
                failure_types_json TEXT,
                prediction_summary TEXT,
                is_safe_to_reuse INTEGER DEFAULT 1,
                created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS context_state(
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                active_context_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, thread_id));
            """)
            self._ensure_column(c, "turns", "thread_id", "TEXT")
            self._ensure_column(c, "machine_values", "thread_id", "TEXT")
            self._ensure_column(c, "summaries", "thread_id", "TEXT")

    @contextmanager
    def _conn(self):
        # with 블록에서 commit/rollback 후 반드시 close 한다(핸들 누수 방지).
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        try:
            with c:
                yield c
        finally:
            c.close()

    @staticmethod
    def _now() -> str:
        return _dt.datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _drop_if_legacy(conn, table: str):
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if cols and "user_id" not in cols:
            conn.execute(f"DROP TABLE IF EXISTS {table}")

    @staticmethod
    def _ensure_column(conn, table: str, column: str, ddl: str):
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if cols and column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    @staticmethod
    def _context_from_row(row) -> DiagnosisContext:
        return DiagnosisContext(
            id=row["id"],
            user_id=row["user_id"],
            thread_id=row["thread_id"],
            turn_id=row["turn_id"],
            features=json.loads(row["features_json"] or "{}"),
            failure_types=json.loads(row["failure_types_json"] or "[]"),
            prediction_summary=row["prediction_summary"] or "",
            is_safe_to_reuse=bool(row["is_safe_to_reuse"]),
            created_at=row["created_at"],
        )

    # --- write ---
    def add_turn(self, user_id, role, content, thread_id=None):
        with self._conn() as c:
            c.execute("INSERT INTO turns(user_id,thread_id,role,content,created_at) VALUES(?,?,?,?,?)",
                      (user_id, thread_id, role, content, self._now()))

    def add_machine_values(self, user_id, values: dict, thread_id=None):
        """Legacy observation log. Prediction input 보완에는 사용하지 않는다."""
        with self._conn() as c:
            for name, v in values.items():
                unit = v.get("unit") if isinstance(v, dict) else None
                val = v.get("value") if isinstance(v, dict) else v
                c.execute("INSERT INTO machine_values(user_id,thread_id,name,value,unit,created_at) VALUES(?,?,?,?,?,?)",
                          (user_id, thread_id, name, str(val), unit, self._now()))

    def add_summary(self, user_id, kind, content, thread_id=None):
        if not content:
            return
        with self._conn() as c:
            c.execute("INSERT INTO summaries(user_id,thread_id,kind,content,created_at) VALUES(?,?,?,?,?)",
                      (user_id, thread_id, kind, content, self._now()))

    def save_diagnosis_context(self, user_id: str, thread_id: str, context: DiagnosisContext) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO diagnosis_contexts(
                    id,user_id,thread_id,turn_id,features_json,failure_types_json,prediction_summary,is_safe_to_reuse,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    context.id,
                    user_id,
                    thread_id,
                    context.turn_id,
                    json.dumps(context.features, ensure_ascii=False),
                    json.dumps(context.failure_types, ensure_ascii=False),
                    context.prediction_summary,
                    1 if context.is_safe_to_reuse else 0,
                    context.created_at,
                ),
            )
            c.execute(
                """INSERT INTO context_state(user_id,thread_id,active_context_id,updated_at)
                   VALUES(?,?,?,?)
                   ON CONFLICT(user_id,thread_id) DO UPDATE SET
                       active_context_id=excluded.active_context_id,
                       updated_at=excluded.updated_at""",
                (user_id, thread_id, context.id, self._now()),
            )
            self._prune_recent_contexts(c, user_id, thread_id, keep=5)

    def set_active_context(self, user_id: str, thread_id: str, context_id: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO context_state(user_id,thread_id,active_context_id,updated_at)
                   VALUES(?,?,?,?)
                   ON CONFLICT(user_id,thread_id) DO UPDATE SET
                       active_context_id=excluded.active_context_id,
                       updated_at=excluded.updated_at""",
                (user_id, thread_id, context_id, self._now()),
            )

    def _prune_recent_contexts(self, conn, user_id: str, thread_id: str, keep: int = 5) -> None:
        rows = conn.execute(
            "SELECT id FROM diagnosis_contexts WHERE user_id=? AND thread_id=? ORDER BY created_at DESC, rowid DESC",
            (user_id, thread_id),
        ).fetchall()
        stale_ids = [r["id"] for r in rows[keep:]]
        if stale_ids:
            conn.executemany("DELETE FROM diagnosis_contexts WHERE id=?", [(x,) for x in stale_ids])

    # --- read ---
    def recent_turns(self, user_id, limit=8, thread_id=None) -> list[dict]:
        # thread_id가 주어지면 그 대화로만 한정한다(다른 thread 대화 누수 방지).
        # thread_id가 없을 때만 user 전체에서 조회한다.
        with self._conn() as c:
            if thread_id:
                rows = c.execute(
                    "SELECT role,content,created_at FROM turns WHERE user_id=? AND thread_id=? ORDER BY id DESC LIMIT ?",
                    (user_id, thread_id, limit)).fetchall()
            else:
                rows = c.execute(
                    "SELECT role,content,created_at FROM turns WHERE user_id=? ORDER BY id DESC LIMIT ?",
                    (user_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def latest_machine_values(self, user_id, thread_id=None) -> dict[str, dict]:
        """Legacy inspection helper. ContextManager는 이 값을 prediction input 보완에 사용하지 않는다."""
        with self._conn() as c:
            if thread_id:
                rows = c.execute(
                    "SELECT name,value,unit,created_at FROM machine_values WHERE user_id=? AND thread_id=? ORDER BY id DESC",
                    (user_id, thread_id)).fetchall()
            else:
                rows = c.execute(
                    "SELECT name,value,unit,created_at FROM machine_values WHERE user_id=? ORDER BY id DESC",
                    (user_id,)).fetchall()
        out: dict[str, dict] = {}
        for r in rows:
            if r["name"] not in out:
                out[r["name"]] = {"value": r["value"], "unit": r["unit"], "created_at": r["created_at"]}
        return out

    def latest_summary(self, user_id, kind, thread_id=None) -> Optional[str]:
        # thread_id가 주어지면 그 대화로만 한정한다(다른 thread 요약 누수 방지).
        with self._conn() as c:
            if thread_id:
                row = c.execute(
                    "SELECT content FROM summaries WHERE user_id=? AND thread_id=? AND kind=? ORDER BY id DESC LIMIT 1",
                    (user_id, thread_id, kind)).fetchone()
            else:
                row = c.execute(
                    "SELECT content FROM summaries WHERE user_id=? AND kind=? ORDER BY id DESC LIMIT 1",
                    (user_id, kind)).fetchone()
        return row["content"] if row else None

    def get_active_context(self, user_id: str, thread_id: str) -> DiagnosisContext | None:
        with self._conn() as c:
            state = c.execute(
                "SELECT active_context_id FROM context_state WHERE user_id=? AND thread_id=?",
                (user_id, thread_id),
            ).fetchone()
            if not state or not state["active_context_id"]:
                return None
            row = c.execute(
                "SELECT * FROM diagnosis_contexts WHERE id=? AND user_id=? AND thread_id=? AND is_safe_to_reuse=1",
                (state["active_context_id"], user_id, thread_id),
            ).fetchone()
        return self._context_from_row(row) if row else None

    def get_recent_contexts(self, user_id: str, thread_id: str, limit: int = 5) -> list[DiagnosisContext]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT * FROM diagnosis_contexts
                   WHERE user_id=? AND thread_id=? AND is_safe_to_reuse=1
                   ORDER BY created_at DESC, rowid DESC LIMIT ?""",
                (user_id, thread_id, limit),
            ).fetchall()
        return [self._context_from_row(r) for r in rows]


class RunStore:
    """실행 이력/관측 데이터 저장."""

    def __init__(self, db_path: str = LONGTERM_DB):
        self.db_path = db_path
        with closing(sqlite3.connect(self.db_path)) as c, c:
            cols = {row[1] for row in c.execute("PRAGMA table_info(runs)")}
            dropped_legacy = False
            if cols and "user_id" not in cols:
                c.execute("DROP TABLE IF EXISTS runs")
                dropped_legacy = True
            c.execute("""CREATE TABLE IF NOT EXISTS runs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT, user_id TEXT, thread_id TEXT, trace_json TEXT, created_at TEXT)""")
            if cols and not dropped_legacy and "thread_id" not in cols:
                c.execute("ALTER TABLE runs ADD COLUMN thread_id TEXT")

    def save(self, request_id, user_id, thread_id, trace: dict):
        with closing(sqlite3.connect(self.db_path)) as c, c:
            c.execute("INSERT INTO runs(request_id,user_id,thread_id,trace_json,created_at) VALUES(?,?,?,?,?)",
                      (request_id, user_id, thread_id, json.dumps(trace, ensure_ascii=False),
                       _dt.datetime.now().isoformat(timespec="seconds")))


conversation_store = ConversationStore()
run_store = RunStore()
print("장기 메모리(SQLite) 준비 완료:", LONGTERM_DB)
