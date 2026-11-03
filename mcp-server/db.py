"""
A tiny SQLite store for the MCP server — tasks and notes.

Kept deliberately boring: the point of the example is the MCP *server*, not
the persistence. Swap this for a real database and nothing in server.py
changes.
"""

import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "store.db"


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            priority TEXT DEFAULT 'medium',
            tags TEXT DEFAULT '',
            status TEXT DEFAULT 'open',
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            tags TEXT DEFAULT '',
            created_at TEXT
        );
        """)


def add_task(title, priority="medium", tags=""):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO tasks (title, priority, tags, created_at) VALUES (?,?,?,?)",
            (title, priority, tags, time.strftime("%Y-%m-%d")))
        return cur.lastrowid


def list_tasks(status="open", tag=None):
    q = "SELECT * FROM tasks WHERE status=?"
    args = [status]
    if tag:
        q += " AND tags LIKE ?"
        args.append(f"%{tag}%")
    q += " ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, id"
    with _conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def complete_task(task_id):
    with _conn() as c:
        cur = c.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
        return cur.rowcount > 0


def add_note(title, body, tags=""):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO notes (title, body, tags, created_at) VALUES (?,?,?,?)",
            (title, body, tags, time.strftime("%Y-%m-%d")))
        return cur.lastrowid


def get_note(note_id):
    with _conn() as c:
        r = c.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
        return dict(r) if r else None


def search(query):
    like = f"%{query}%"
    with _conn() as c:
        tasks = [dict(r) for r in c.execute(
            "SELECT * FROM tasks WHERE title LIKE ? OR tags LIKE ?",
            (like, like)).fetchall()]
        notes = [dict(r) for r in c.execute(
            "SELECT * FROM notes WHERE title LIKE ? OR body LIKE ? OR tags LIKE ?",
            (like, like, like)).fetchall()]
    return {"tasks": tasks, "notes": notes}
