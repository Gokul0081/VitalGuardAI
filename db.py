"""
db.py
Lightweight SQLite persistence layer, matching the project's ER diagram:
User (registered accounts), User_Session (guest sessions), Diabetes_Prediction,
CKD_Prediction, Admin, Model_Metrics.

Design note: accounts are OPTIONAL. Guests keep working exactly as before,
identified only by an anonymous session_id. A logged-in user's predictions
are additionally tagged with user_id, giving them persistent history across
visits/devices without forcing anyone to register.
"""
import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_session (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            security_question TEXT NOT NULL,
            security_answer_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS diabetes_prediction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id INTEGER,
            pregnancies REAL, glucose REAL, blood_pressure REAL, skin_thickness REAL,
            insulin REAL, bmi REAL, pedigree REAL, age REAL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES user_session(session_id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ckd_prediction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id INTEGER,
            age REAL, bp REAL, sg REAL, al REAL, su REAL, bgr REAL, bu REAL,
            sc REAL, sod REAL, pot REAL, hemo REAL, htn INTEGER, dm INTEGER,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES user_session(session_id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    cur.execute("SELECT COUNT(*) AS c FROM admin")
    if cur.fetchone()["c"] == 0:
        cur.execute(
            "INSERT INTO admin (username, password_hash) VALUES (?, ?)",
            ("admin", generate_password_hash("admin123")),
        )

    conn.commit()
    conn.close()


def ensure_session(session_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM user_session WHERE session_id = ?", (session_id,))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO user_session (session_id, created_at) VALUES (?, ?)",
            (session_id, datetime.utcnow().isoformat()),
        )
        conn.commit()
    conn.close()


# ---------------------------------------------------------------
# User accounts
# ---------------------------------------------------------------
def create_user(username, email, password_hash, security_question, security_answer_hash):
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO users (username, email, password_hash, security_question, security_answer_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (username, email, password_hash, security_question, security_answer_hash, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_username(username):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row


def get_user_by_email(email):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


def update_password(user_id, new_password_hash):
    conn = get_db()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_password_hash, user_id))
    conn.commit()
    conn.close()


def get_user_stats(user_id):
    conn = get_db()
    d_count = conn.execute("SELECT COUNT(*) AS c FROM diabetes_prediction WHERE user_id = ?", (user_id,)).fetchone()["c"]
    c_count = conn.execute("SELECT COUNT(*) AS c FROM ckd_prediction WHERE user_id = ?", (user_id,)).fetchone()["c"]
    conn.close()
    return {"diabetes_checks": d_count, "ckd_checks": c_count}


# ---------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------
def save_diabetes_prediction(session_id, form_values, result, confidence, user_id=None):
    conn = get_db()
    conn.execute("""
        INSERT INTO diabetes_prediction
        (session_id, user_id, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age, result, confidence, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        session_id, user_id, form_values["pregnancies"], form_values["glucose"], form_values["blood_pressure"],
        form_values["skin_thickness"], form_values["insulin"], form_values["bmi"],
        form_values["pedigree"], form_values["age"], result, confidence, datetime.utcnow().isoformat(),
    ))
    conn.commit()
    conn.close()


def save_ckd_prediction(session_id, form_values, result, confidence, user_id=None):
    conn = get_db()
    conn.execute("""
        INSERT INTO ckd_prediction
        (session_id, user_id, age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, htn, dm, result, confidence, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        session_id, user_id, form_values["age"], form_values["bp"], form_values["sg"], form_values["al"],
        form_values["su"], form_values["bgr"], form_values["bu"], form_values["sc"], form_values["sod"],
        form_values["pot"], form_values["hemo"], form_values["htn"], form_values["dm"],
        result, confidence, datetime.utcnow().isoformat(),
    ))
    conn.commit()
    conn.close()


def get_admin_by_username(username):
    conn = get_db()
    row = conn.execute("SELECT * FROM admin WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row


def get_session_history(session_id, user_id=None):
    """Guest history is scoped to this browser session. A logged-in user instead
    sees everything tied to their account, across any session/device."""
    conn = get_db()
    if user_id:
        diabetes_rows = conn.execute(
            "SELECT * FROM diabetes_prediction WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
        ckd_rows = conn.execute(
            "SELECT * FROM ckd_prediction WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    else:
        diabetes_rows = conn.execute(
            "SELECT * FROM diabetes_prediction WHERE session_id = ? ORDER BY created_at DESC", (session_id,)
        ).fetchall()
        ckd_rows = conn.execute(
            "SELECT * FROM ckd_prediction WHERE session_id = ? ORDER BY created_at DESC", (session_id,)
        ).fetchall()
    conn.close()
    return {"diabetes": diabetes_rows, "ckd": ckd_rows}


def get_dashboard_stats():
    conn = get_db()
    total_diabetes = conn.execute("SELECT COUNT(*) AS c FROM diabetes_prediction").fetchone()["c"]
    diabetic_count = conn.execute("SELECT COUNT(*) AS c FROM diabetes_prediction WHERE result = 'Diabetic'").fetchone()["c"]
    total_ckd = conn.execute("SELECT COUNT(*) AS c FROM ckd_prediction").fetchone()["c"]
    ckd_risk_count = conn.execute("SELECT COUNT(*) AS c FROM ckd_prediction WHERE result = 'High Risk'").fetchone()["c"]
    total_users = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    recent = conn.execute("""
        SELECT 'Diabetes' AS type, result, confidence, created_at FROM diabetes_prediction
        UNION ALL
        SELECT 'CKD' AS type, result, confidence, created_at FROM ckd_prediction
        ORDER BY created_at DESC LIMIT 15
    """).fetchall()
    conn.close()
    return {
        "total_diabetes": total_diabetes,
        "diabetic_count": diabetic_count,
        "total_ckd": total_ckd,
        "ckd_risk_count": ckd_risk_count,
        "total_users": total_users,
        "recent": recent,
    }
