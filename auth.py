"""
VERITAS AI — Authentication Module
====================================
SQLite-based auth — no extra database needed.
Features:
  - Register with username + email + password
  - Secure password hashing (pbkdf2:sha256)
  - Session-based login / logout
  - Analysis history saved per user
  - User stats dashboard
  - login_required decorator
"""
#thie is the authentication which is done by sqlite3
import os
import sqlite3
import functools
from datetime import datetime

from flask import g, session, redirect, url_for, request
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'veritas.db')


# ── DB helpers ────────────────────────────────────────────────

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db


def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'user',
            created_at    TEXT    NOT NULL,
            last_login    TEXT
        );
        CREATE TABLE IF NOT EXISTS analyses (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL REFERENCES users(id),
            modality      TEXT    NOT NULL,
            label         TEXT    NOT NULL,
            confidence    REAL    NOT NULL,
            risk_level    TEXT    NOT NULL,
            input_name    TEXT,
            created_at    TEXT    NOT NULL
        );
    """)
    db.commit()
    db.close()
    print("[Auth] Database ready ->", DB_PATH)


# ── register ──────────────────────────────────────────────────

def register_user(username, email, password):
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(username) > 30:
        return False, "Username must be at most 30 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if '@' not in email or '.' not in email.split('@')[-1]:
        return False, "Please enter a valid email address."

    db     = get_db()
    now    = datetime.now().isoformat(sep=' ', timespec='seconds')
    hashed = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

    try:
        db.execute(
            "INSERT INTO users (username,email,password_hash,created_at) VALUES (?,?,?,?)",
            (username, email.lower(), hashed, now)
        )
        db.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError as e:
        if 'username' in str(e):
            return False, "Username already taken. Please choose another."
        if 'email' in str(e):
            return False, "An account with this email already exists."
        return False, "Registration failed. Please try again."


# ── login ─────────────────────────────────────────────────────

def login_user(username_or_email, password):
    db   = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE username=? OR email=?",
        (username_or_email, username_or_email.lower())
    ).fetchone()

    if user is None:
        return False, "No account found with that username or email.", None
    if not check_password_hash(user['password_hash'], password):
        return False, "Incorrect password. Please try again.", None

    now = datetime.now().isoformat(sep=' ', timespec='seconds')
    db.execute("UPDATE users SET last_login=? WHERE id=?", (now, user['id']))
    db.commit()
    return True, "Login successful!", dict(user)


# ── current user ──────────────────────────────────────────────

def current_user():
    if 'user_id' not in session:
        return None
    db   = get_db()
    user = db.execute(
        "SELECT id,username,email,role,created_at,last_login FROM users WHERE id=?",
        (session['user_id'],)
    ).fetchone()
    return dict(user) if user else None


# ── decorator ─────────────────────────────────────────────────

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated


# ── history & stats ───────────────────────────────────────────

def save_analysis(user_id, modality, label, confidence, risk_level, input_name=''):
    db  = get_db()
    now = datetime.now().isoformat(sep=' ', timespec='seconds')
    db.execute(
        "INSERT INTO analyses (user_id,modality,label,confidence,risk_level,input_name,created_at) VALUES (?,?,?,?,?,?,?)",
        (user_id, modality, label, round(float(confidence), 2), risk_level, input_name or '', now)
    )
    db.commit()


def get_user_history(user_id, limit=20):
    db   = get_db()
    rows = db.execute(
        "SELECT * FROM analyses WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    return [dict(r) for r in rows]


def get_user_stats(user_id):
    db    = get_db()
    total = db.execute(
        "SELECT COUNT(*) as n FROM analyses WHERE user_id=?", (user_id,)
    ).fetchone()['n']
    by_modality = db.execute(
        "SELECT modality, COUNT(*) as n FROM analyses WHERE user_id=? GROUP BY modality", (user_id,)
    ).fetchall()
    by_risk = db.execute(
        "SELECT risk_level, COUNT(*) as n FROM analyses WHERE user_id=? GROUP BY risk_level", (user_id,)
    ).fetchall()
    by_label = db.execute(
        "SELECT label, COUNT(*) as n FROM analyses WHERE user_id=? GROUP BY label ORDER BY n DESC LIMIT 5", (user_id,)
    ).fetchall()
    return {
        'total':       total,
        'by_modality': {r['modality']: r['n'] for r in by_modality},
        'by_risk':     {r['risk_level']: r['n'] for r in by_risk},
        'by_label':    [dict(r) for r in by_label],
    }
