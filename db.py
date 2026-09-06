"""
Persistence layer — Stage 1 foundation fix.

Replaces:
  - bookmarks.json (flat file, broke with >1 concurrent user)
  - st.session_state.results (wiped on every page refresh)

Uses SQLite (stdlib `sqlite3`, no external dependency) so this works
identically in local dev and on a small deployment without needing a
hosted database service. If/when the app outgrows SQLite's concurrent-
write limits, every function here maps 1:1 onto Postgres — swap the
connection layer, keep the call sites in streamlit_app.py unchanged.

Password hashing uses stdlib `hashlib.pbkdf2_hmac` (no bcrypt/passlib
dependency needed) with a random per-user salt and 260,000 iterations
(OWASP's current minimum recommendation for PBKDF2-SHA256, as of their
2023 cheat sheet).
"""

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

DB_PATH = os.environ.get("APP_DB_PATH", "app.db")

_PBKDF2_ITERATIONS = 260_000


# ════════════════════════════════════════════════════════════════════════════
# CONNECTION / SCHEMA
# ════════════════════════════════════════════════════════════════════════════

@contextmanager
def get_conn():
    """Yields a SQLite connection with foreign keys enabled and row access
    by column name. Always used as a context manager so connections are
    never leaked across Streamlit reruns."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Creates all tables if they don't exist yet. Safe to call on every
    app startup — CREATE TABLE IF NOT EXISTS is idempotent."""
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                is_teacher    INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS bookmarks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                surah       INTEGER NOT NULL,
                ayah        INTEGER NOT NULL,
                surah_name  TEXT NOT NULL,
                note        TEXT DEFAULT '',
                added_at    TEXT NOT NULL,
                UNIQUE(user_id, surah, ayah)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                start_surah  INTEGER NOT NULL,
                start_ayah   INTEGER NOT NULL,
                end_surah    INTEGER NOT NULL,
                end_ayah     INTEGER NOT NULL,
                similarity   REAL NOT NULL,
                passed       INTEGER NOT NULL,
                created_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_words (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                surah       INTEGER NOT NULL,
                ayah        INTEGER NOT NULL,
                word_text   TEXT NOT NULL,
                status      TEXT NOT NULL,
                word_order  INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS review_state (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                surah          INTEGER NOT NULL,
                ayah           INTEGER NOT NULL,
                ease_factor    REAL NOT NULL DEFAULT 2.5,
                interval_days  INTEGER NOT NULL DEFAULT 1,
                repetitions    INTEGER NOT NULL DEFAULT 0,
                next_due_date  TEXT NOT NULL,
                updated_at     TEXT NOT NULL,
                UNIQUE(user_id, surah, ayah)
            );

            CREATE TABLE IF NOT EXISTS classrooms (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                teacher_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name        TEXT NOT NULL,
                join_code   TEXT UNIQUE NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS classroom_members (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                classroom_id    INTEGER NOT NULL REFERENCES classrooms(id) ON DELETE CASCADE,
                student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                joined_at       TEXT NOT NULL,
                UNIQUE(classroom_id, student_user_id)
            );

            CREATE TABLE IF NOT EXISTS assignments (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                classroom_id  INTEGER NOT NULL REFERENCES classrooms(id) ON DELETE CASCADE,
                from_surah    INTEGER NOT NULL,
                to_surah      INTEGER NOT NULL,
                due_date      TEXT,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS assignment_submissions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                assignment_id    INTEGER NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
                student_user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                teacher_comment  TEXT DEFAULT '',
                teacher_reviewed INTEGER NOT NULL DEFAULT 0,
                created_at       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_bookmarks_user ON bookmarks(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_session_words_session ON session_words(session_id);
            CREATE INDEX IF NOT EXISTS idx_review_state_due ON review_state(user_id, next_due_date);
            CREATE INDEX IF NOT EXISTS idx_classroom_members_student ON classroom_members(student_user_id);
            CREATE INDEX IF NOT EXISTS idx_assignments_classroom ON assignments(classroom_id);
            CREATE INDEX IF NOT EXISTS idx_submissions_assignment ON assignment_submissions(assignment_id);
            -- Weak-ayah lookup (feeds the Stage 2 SRS queue directly off this index)
            CREATE INDEX IF NOT EXISTS idx_session_words_status ON session_words(surah, ayah, status);
            """
        )

        # Self-healing migration: CREATE TABLE IF NOT EXISTS above is a no-op
        # against a `users` table that already exists from an earlier run of
        # this app (e.g. an app.db file left over from before is_teacher
        # existed) — unlike Postgres, SQLite has no ADD COLUMN IF NOT EXISTS,
        # so this checks the actual current columns and patches the table if
        # it predates this column. Safe to run on every startup.
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "is_teacher" not in existing_columns:
            conn.execute("ALTER TABLE users ADD COLUMN is_teacher INTEGER NOT NULL DEFAULT 0")

        # Stage 3.2: "Exam Mode" vs "Practice Mode" is now a named, tracked
        # choice rather than an unlabeled always-random start. Existing rows
        # predate this column — they were all random-start sessions, so they
        # backfill as 'exam' to reflect what actually happened, not a guess.
        session_columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "mode" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN mode TEXT NOT NULL DEFAULT 'exam'")


# ════════════════════════════════════════════════════════════════════════════
# AUTH
# ════════════════════════════════════════════════════════════════════════════

def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Returns (hash_hex, salt_hex). Generates a new random salt if none given."""
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return digest.hex(), salt.hex()


def create_user(username: str, password: str) -> int | None:
    """Returns the new user's id, or None if the username is already taken.
    Raises ValueError for empty username/password (caller should validate
    in the UI too, but never trust the UI layer alone)."""
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password cannot be empty.")

    password_hash, password_salt = _hash_password(password)
    with get_conn() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, password_salt, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, password_hash, password_salt, datetime.now(timezone.utc).isoformat()),
            )
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None  # username already exists


def authenticate_user(username: str, password: str) -> int | None:
    """Returns the user's id if the password is correct, else None.
    Constant-time-ish: always hashes even on username-not-found to avoid
    trivially timing whether a username exists (not perfect, but better
    than short-circuiting immediately)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, password_hash, password_salt FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()

    if row is None:
        _hash_password(password, os.urandom(16))  # dummy hash, keeps timing similar
        return None

    computed_hash, _ = _hash_password(password, bytes.fromhex(row["password_salt"]))
    if computed_hash == row["password_hash"]:
        return row["id"]
    return None


# ════════════════════════════════════════════════════════════════════════════
# BOOKMARKS  (user-scoped — this is the fix for the concurrent-user bug)
# ════════════════════════════════════════════════════════════════════════════

def load_bookmarks(user_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT surah, ayah, surah_name, note, added_at FROM bookmarks "
            "WHERE user_id = ? ORDER BY surah, ayah",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_bookmark(user_id: int, surah: int, ayah: int, surah_name: str, note: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO bookmarks (user_id, surah, ayah, surah_name, note, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, surah, ayah, surah_name, note, datetime.now(timezone.utc).isoformat()),
        )


def remove_bookmark(user_id: int, surah: int, ayah: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM bookmarks WHERE user_id = ? AND surah = ? AND ayah = ?",
            (user_id, surah, ayah),
        )


def is_bookmarked(surah: int, ayah: int, bookmarks: list[dict]) -> bool:
    """Pure helper, unchanged signature from the old flat-file version —
    still just checks membership in an already-loaded list."""
    return any(b["surah"] == surah and b["ayah"] == ayah for b in bookmarks)


# ════════════════════════════════════════════════════════════════════════════
# SESSION / GRADING HISTORY  (this is the "no data loss on refresh" fix)
# ════════════════════════════════════════════════════════════════════════════

def save_session_result(user_id: int, result: dict, mode: str = "exam") -> int:
    """Persists one grade_continuous_recitation() result dict.

    `mode` is either "exam" (random starting ayah within the chosen range —
    mirrors how a real Hifz examiner tests, the app's flagship
    differentiator) or "practice" (sequential start from the beginning of
    the range). Stage 3.2: this used to be unconditionally random with no
    label anywhere in the UI or data — a genuine differentiator nobody
    could see or choose.

    Expects the same shape the grading function already produces:
      result = {
        "start_surah": int, "start_ayah": int,
        "end_surah": int, "end_ayah": int,
        "similarity": float, "passed": bool,
        "ayahs": [ {"surah": int, "ayah": int,
                     "words": [ {"text": str, "status": str}, ... ] }, ... ],
        ...
      }

    Returns the new session's row id.
    """
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions "
            "(user_id, start_surah, start_ayah, end_surah, end_ayah, similarity, passed, created_at, mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                result["start_surah"], result["start_ayah"],
                result["end_surah"], result["end_ayah"],
                result["similarity"], int(bool(result["passed"])),
                datetime.now(timezone.utc).isoformat(),
                mode,
            ),
        )
        session_id = cur.lastrowid

        word_rows = []
        order = 0
        for ayah in result["ayahs"]:
            for w in ayah["words"]:
                word_rows.append((session_id, ayah["surah"], ayah["ayah"], w["text"], w["status"], order))
                order += 1
        if word_rows:
            conn.executemany(
                "INSERT INTO session_words (session_id, surah, ayah, word_text, status, word_order) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                word_rows,
            )
        return session_id


def get_session_history(user_id: int, limit: int | None = 50) -> list[dict]:
    """Returns past sessions, newest first, WITHOUT per-word detail (cheap —
    use get_session_words() for a specific session if the UI needs the
    word-level breakdown, e.g. when a user expands one session)."""
    query = (
        "SELECT id, start_surah, start_ayah, end_surah, end_ayah, similarity, passed, created_at, mode "
        "FROM sessions WHERE user_id = ? ORDER BY created_at DESC"
    )
    params: tuple = (user_id,)
    if limit is not None:
        query += " LIMIT ?"
        params = (user_id, limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_session_words(session_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT surah, ayah, word_text, status FROM session_words "
            "WHERE session_id = ? ORDER BY word_order",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_overall_stats(user_id: int) -> dict:
    """Aggregate accuracy across ALL persisted sessions (survives refresh —
    this is the number that used to reset to zero every time the tab closed)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n_sessions, AVG(similarity) AS avg_similarity, "
            "SUM(passed) AS n_passed FROM sessions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    n_sessions = row["n_sessions"] or 0
    return {
        "n_sessions": n_sessions,
        "avg_similarity": (row["avg_similarity"] or 0.0),
        "n_passed": row["n_passed"] or 0,
        "pass_rate": (row["n_passed"] / n_sessions) if n_sessions else 0.0,
    }


def get_weak_ayahs(user_id: int, limit: int = 20) -> list[dict]:
    """Ayahs with the most 'wrong'/'missing' word-status hits across this
    user's history, most-problematic first. This is the exact query Stage 2's
    SRS queue will read from — the table shape already supports it, this
    function just isn't surfaced in the UI yet."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT sw.surah, sw.ayah, COUNT(*) AS miss_count
            FROM session_words sw
            JOIN sessions s ON s.id = sw.session_id
            WHERE s.user_id = ? AND sw.status IN ('wrong', 'missing')
            GROUP BY sw.surah, sw.ayah
            ORDER BY miss_count DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════════════════════
# STREAKS + TREND (Stage 2.2 — the one piece of Stage 2 that was still missing)
# ════════════════════════════════════════════════════════════════════════════

def get_streaks(user_id: int) -> dict:
    """
    Current streak = consecutive calendar days (up to and including today
    OR yesterday — a streak isn't "broken" until a full day is skipped)
    with at least one completed session. Longest streak = the best run
    ever recorded, so a lapsed streak doesn't erase past consistency.

    Deliberately derived from `sessions.created_at` rather than a new
    `daily_activity` table, per the roadmap's own "don't over-build" note
    for this stage — one distinct-dates query is enough for v1 and this
    ports to Postgres with zero SQL changes.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT date(created_at) AS d FROM sessions "
            "WHERE user_id = ? ORDER BY d DESC",
            (user_id,),
        ).fetchall()

    if not rows:
        return {"current_streak": 0, "longest_streak": 0, "active_today": False}

    dates = [date.fromisoformat(r["d"]) for r in rows]
    today = date.today()

    # Current streak: walk backward from today (or yesterday, if today has
    # no session yet — still "alive" until end of day) while consecutive.
    current_streak = 0
    active_today = dates[0] == today
    cursor = today if active_today else (today - timedelta(days=1))
    date_set = set(dates)
    if cursor in date_set or active_today:
        cursor = dates[0]
        while cursor in date_set:
            current_streak += 1
            cursor -= timedelta(days=1)

    # Longest streak ever: scan the full sorted-ascending distinct-date list
    # for the longest run of consecutive calendar days.
    asc = sorted(dates)
    longest_streak = 1
    run = 1
    for prev, cur in zip(asc, asc[1:]):
        if (cur - prev).days == 1:
            run += 1
        else:
            run = 1
        longest_streak = max(longest_streak, run)

    return {
        "current_streak": current_streak,
        "longest_streak": longest_streak,
        "active_today": active_today,
    }


def get_similarity_trend(user_id: int, limit_days: int = 30) -> list[dict]:
    """One row per calendar day with a session, average similarity that
    day — feeds st.line_chart directly. Most-recent `limit_days` days
    with activity, returned oldest-first (chart reads left-to-right)."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT date(created_at) AS d, AVG(similarity) AS avg_similarity
            FROM sessions
            WHERE user_id = ?
            GROUP BY date(created_at)
            ORDER BY d DESC
            LIMIT ?
            """,
            (user_id, limit_days),
        ).fetchall()
    return [{"date": r["d"], "similarity": r["avg_similarity"]} for r in reversed(rows)]



def _sm2_update(ease_factor: float, interval_days: int, repetitions: int, success: bool) -> tuple[float, int, int]:
    if success:
        repetitions += 1
        if repetitions == 1:
            interval_days = 1
        elif repetitions == 2:
            interval_days = 6
        else:
            interval_days = round(interval_days * ease_factor)
        ease_factor = max(1.3, ease_factor + 0.1)
    else:
        repetitions = 0
        interval_days = 1
        ease_factor = max(1.3, ease_factor - 0.2)
    return ease_factor, interval_days, repetitions


def upsert_review_state(user_id: int, surah: int, ayah: int, success: bool) -> None:
    from datetime import date, timedelta
    with get_conn() as conn:
        row = conn.execute(
            "SELECT ease_factor, interval_days, repetitions FROM review_state "
            "WHERE user_id = ? AND surah = ? AND ayah = ?",
            (user_id, surah, ayah),
        ).fetchone()
        if row:
            ease, interval, reps = row["ease_factor"], row["interval_days"], row["repetitions"]
        else:
            ease, interval, reps = 2.5, 1, 0

        ease, interval, reps = _sm2_update(ease, interval, reps, success)
        next_due = (date.today() + timedelta(days=interval)).isoformat()

        conn.execute(
            """
            INSERT INTO review_state (user_id, surah, ayah, ease_factor, interval_days, repetitions, next_due_date, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, surah, ayah) DO UPDATE SET
                ease_factor = excluded.ease_factor,
                interval_days = excluded.interval_days,
                repetitions = excluded.repetitions,
                next_due_date = excluded.next_due_date,
                updated_at = excluded.updated_at
            """,
            (user_id, surah, ayah, ease, interval, reps, next_due, datetime.now(timezone.utc).isoformat()),
        )


def seed_review_from_session(user_id: int, result: dict) -> None:
    for ayah in result["ayahs"]:
        statuses = [w["status"] for w in ayah["words"]]
        success = bool(statuses) and all(s == "correct" for s in statuses)
        upsert_review_state(user_id, ayah["surah"], ayah["ayah"], success)


def get_due_reviews(user_id: int, limit: int = 20) -> list[dict]:
    from datetime import date
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT surah, ayah, ease_factor, interval_days, repetitions, next_due_date
            FROM review_state
            WHERE user_id = ? AND next_due_date <= ?
            ORDER BY next_due_date ASC
            LIMIT ?
            """,
            (user_id, date.today().isoformat(), limit),
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["next_due_date"] = date.fromisoformat(d["next_due_date"])  # match db.py's DATE-typed return
        results.append(d)
    return results


def get_review_count(user_id: int) -> int:
    from datetime import date
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM review_state WHERE user_id = ? AND next_due_date <= ?",
            (user_id, date.today().isoformat()),
        ).fetchone()
    return row["n"]


# ════════════════════════════════════════════════════════════════════════════
# TEACHER DASHBOARD (Stage 2.3) — SQLite port, logic identical to db.py
# ════════════════════════════════════════════════════════════════════════════

def is_teacher(user_id: int) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT is_teacher FROM users WHERE id = ?", (user_id,)).fetchone()
    return bool(row and row["is_teacher"])


def set_teacher_status(user_id: int, value: bool) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET is_teacher = ? WHERE id = ?", (int(value), user_id))


def _generate_join_code() -> str:
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "".join(alphabet[b % len(alphabet)] for b in os.urandom(6))


def create_classroom(teacher_id: int, name: str) -> dict:
    with get_conn() as conn:
        for _ in range(5):
            code = _generate_join_code()
            try:
                cur = conn.execute(
                    "INSERT INTO classrooms (teacher_id, name, join_code, created_at) VALUES (?, ?, ?, ?)",
                    (teacher_id, name.strip(), code, datetime.now(timezone.utc).isoformat()),
                )
                return {"id": cur.lastrowid, "join_code": code}
            except sqlite3.IntegrityError:
                continue
    raise RuntimeError("Could not generate a unique join code after 5 attempts.")


def get_teacher_classrooms(teacher_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.name, c.join_code, c.created_at,
                   COUNT(cm.id) AS member_count
            FROM classrooms c
            LEFT JOIN classroom_members cm ON cm.classroom_id = c.id
            WHERE c.teacher_id = ?
            GROUP BY c.id
            ORDER BY c.created_at DESC
            """,
            (teacher_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def join_classroom(student_id: int, join_code: str) -> tuple[bool, str]:
    with get_conn() as conn:
        classroom = conn.execute(
            "SELECT id, name FROM classrooms WHERE join_code = ?", (join_code.strip().upper(),)
        ).fetchone()
        if not classroom:
            return False, "No classroom found with that code."
        try:
            conn.execute(
                "INSERT INTO classroom_members (classroom_id, student_user_id, joined_at) VALUES (?, ?, ?)",
                (classroom["id"], student_id, datetime.now(timezone.utc).isoformat()),
            )
        except sqlite3.IntegrityError:
            return False, f"You're already a member of {classroom['name']}."
    return True, f"Joined {classroom['name']}."


def get_student_classrooms(student_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.name FROM classrooms c
            JOIN classroom_members cm ON cm.classroom_id = c.id
            WHERE cm.student_user_id = ?
            ORDER BY c.name
            """,
            (student_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def create_assignment(classroom_id: int, from_surah: int, to_surah: int, due_date) -> int:
    due_str = due_date.isoformat() if due_date else None
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO assignments (classroom_id, from_surah, to_surah, due_date, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (classroom_id, from_surah, to_surah, due_str, datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_classroom_assignments(classroom_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, from_surah, to_surah, due_date, created_at FROM assignments "
            "WHERE classroom_id = ? ORDER BY created_at DESC",
            (classroom_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_student_assignments(student_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.from_surah, a.to_surah, a.due_date, c.name AS classroom_name,
                   EXISTS (
                       SELECT 1 FROM assignment_submissions s
                       WHERE s.assignment_id = a.id AND s.student_user_id = ?
                   ) AS already_submitted
            FROM assignments a
            JOIN classrooms c ON c.id = a.classroom_id
            JOIN classroom_members cm ON cm.classroom_id = c.id
            WHERE cm.student_user_id = ?
            ORDER BY (a.due_date IS NULL), a.due_date, a.created_at DESC
            """,
            (student_id, student_id),
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["already_submitted"] = bool(d["already_submitted"])
        results.append(d)
    return results


def record_submission(assignment_id: int, student_id: int, session_id: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO assignment_submissions (assignment_id, student_user_id, session_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (assignment_id, student_id, session_id, datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_assignment_submissions(assignment_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT sub.id AS submission_id, sub.session_id, sub.teacher_comment,
                   sub.teacher_reviewed, sub.created_at,
                   u.username,
                   s.start_surah, s.start_ayah, s.end_surah, s.end_ayah,
                   s.similarity, s.passed
            FROM assignment_submissions sub
            JOIN users u ON u.id = sub.student_user_id
            JOIN sessions s ON s.id = sub.session_id
            WHERE sub.assignment_id = ?
            ORDER BY sub.created_at DESC
            """,
            (assignment_id,),
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["teacher_reviewed"] = bool(d["teacher_reviewed"])
        d["passed"] = bool(d["passed"])
        results.append(d)
    return results


def update_submission_review(submission_id: int, teacher_comment: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE assignment_submissions SET teacher_comment = ?, teacher_reviewed = 1 WHERE id = ?",
            (teacher_comment, submission_id),
        )
