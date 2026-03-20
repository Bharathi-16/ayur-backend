"""
Database — SQLite with schema auto-creation
"""
import sqlite3
import os
import uuid
from datetime import datetime

DB_PATH = None

def init_db(app):
    global DB_PATH
    DB_PATH = app.config['DATABASE']
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT 'New Chat',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            token_count INTEGER DEFAULT 0,
            latency_ms INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] ✅ Database initialized")


def get_db():
    # Increase timeout to handle concurrent writes
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable Write-Ahead Logging (WAL) for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Session CRUD ──

def create_session(title="New Chat"):
    db = get_db()
    sid = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()
    db.execute("INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
               (sid, title, now, now))
    db.commit()
    db.close()
    return sid


def list_sessions():
    db = get_db()
    rows = db.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_session(sid):
    db = get_db()
    row = db.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
    db.close()
    return dict(row) if row else None


def update_session_title(sid, title):
    db = get_db()
    now = datetime.utcnow().isoformat()
    db.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, now, sid))
    db.commit()
    db.close()


def delete_session(sid):
    db = get_db()
    db.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
    db.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    db.commit()
    db.close()


# ── Message CRUD ──

def add_message(session_id, role, content, token_count=0, latency_ms=0):
    db = get_db()
    mid = str(uuid.uuid4())[:12]
    now = datetime.utcnow().isoformat()
    db.execute(
        "INSERT INTO messages (id, session_id, role, content, timestamp, token_count, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (mid, session_id, role, content, now, token_count, latency_ms)
    )
    db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    db.commit()
    db.close()
    return mid


def get_messages(session_id):
    db = get_db()
    rows = db.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,)
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def delete_last_assistant_message(session_id):
    """Delete last assistant message for regeneration"""
    db = get_db()
    row = db.execute(
        "SELECT id FROM messages WHERE session_id = ? AND role = 'assistant' ORDER BY timestamp DESC LIMIT 1",
        (session_id,)
    ).fetchone()
    if row:
        db.execute("DELETE FROM messages WHERE id = ?", (row['id'],))
        db.commit()
    db.close()


# ── Settings ──

def get_setting(key, default=None):
    db = get_db()
    row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    db.close()
    return row['value'] if row else default


def set_setting(key, value):
    db = get_db()
    db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    db.commit()
    db.close()
