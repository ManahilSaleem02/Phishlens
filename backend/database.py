"""Lightweight SQLite store for scan history and analytics (stdlib only)."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scans.db")
_lock = threading.Lock()


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                prediction TEXT NOT NULL,
                risk_score REAL NOT NULL,
                registered_domain TEXT,
                created_at REAL NOT NULL
            )
        """)


def record(url: str, prediction: str, risk_score: float, domain: str):
    with _lock, _conn() as con:
        con.execute(
            "INSERT INTO scans (url, prediction, risk_score, registered_domain, created_at) "
            "VALUES (?,?,?,?,?)",
            (url, prediction, risk_score, domain, time.time()))


def history(limit: int = 50):
    with _conn() as con:
        rows = con.execute(
            "SELECT url, prediction, risk_score, registered_domain, created_at "
            "FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def stats():
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) c FROM scans").fetchone()["c"]
        by_pred = con.execute(
            "SELECT prediction, COUNT(*) c FROM scans GROUP BY prediction").fetchall()
        avg_risk = con.execute("SELECT AVG(risk_score) a FROM scans").fetchone()["a"]
        # last 14 days buckets for the trend chart
        rows = con.execute("SELECT risk_score, created_at FROM scans "
                           "ORDER BY id DESC LIMIT 500").fetchall()
    counts = {r["prediction"]: r["c"] for r in by_pred}
    return {
        "total_scans": total,
        "phishing": counts.get("phishing", 0),
        "suspicious": counts.get("suspicious", 0),
        "legitimate": counts.get("legitimate", 0),
        "avg_risk_score": round(avg_risk or 0, 1),
        "recent_scores": [round(r["risk_score"], 1) for r in rows][::-1],
    }
