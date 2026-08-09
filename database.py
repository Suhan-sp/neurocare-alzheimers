"""
SQLite Database Persistence Layer (database.py)
Manages User Accounts, Authentication Sessions, and Patient Evaluation Report History.
"""

import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from utils.logger import logger

DB_PATH = "neurocare.db"

def get_db_connection():
    """Establishes connection to local SQLite database with dictionary row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite database tables for users and patient reports."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Patient Evaluation Reports Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            patient_name TEXT NOT NULL,
            age INTEGER DEFAULT 70,
            gender TEXT DEFAULT 'M',
            diagnosis TEXT NOT NULL,
            final_probability_pct REAL NOT NULL,
            risk_level TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            cognitive_prob REAL NOT NULL,
            eeg_prob REAL NOT NULL,
            speech_prob REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()

    # 3. Seed Default Admin Account if missing
    cursor.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    if not cursor.fetchone():
        admin_pass = generate_password_hash("admin123")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("admin", "admin@neurocare.ai", admin_pass, "admin")
        )
        conn.commit()
        logger.info("Default Admin account created: username='admin', password='admin123'")

    conn.close()

def register_user(username, email, password, role='user'):
    """Registers a new user account."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        pass_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (username, email, pass_hash, role)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, user_id
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username or Email already exists."
    except Exception as e:
        conn.close()
        return False, str(e)

def authenticate_user(username_or_email, password):
    """Authenticates user or admin credentials."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?",
        (username_or_email, username_or_email)
    )
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None

def save_patient_report(user_id, patient_name, *args, **kwargs):
    """
    Saves a generated patient report into the database.
    Flexible signature supports:
    - save_patient_report(user_id, patient_name, age, gender, report_dict)
    - save_patient_report(user_id, patient_name, report_dict, cog_dict)
    - save_patient_report(user_id, patient_name, report_dict)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    age = 70
    gender = 'M'
    report_dict = {}

    # Unpack positional args dynamically
    if len(args) == 3:
        age = args[0]
        gender = args[1]
        report_dict = args[2] if isinstance(args[2], dict) else {}
    elif len(args) == 2:
        if isinstance(args[0], dict):
            report_dict = args[0]
            if isinstance(args[1], dict):
                age = args[1].get('Age', 70)
                gender = args[1].get('Gender', 'M')
        else:
            age = args[0]
            gender = args[1]
    elif len(args) == 1:
        if isinstance(args[0], dict):
            report_dict = args[0]

    # Override from kwargs if explicitly provided
    if kwargs.get('report_dict'):
        report_dict = kwargs['report_dict']
    if kwargs.get('age'):
        age = kwargs['age']
    if kwargs.get('gender'):
        gender = kwargs['gender']

    try:
        age = int(float(age))
    except Exception:
        age = 70
        
    gender = str(gender) if gender else 'M'
    ind = report_dict.get('individual_probabilities', {}) if isinstance(report_dict, dict) else {}
    
    cog_p = float(ind.get('cognitive') or 0.0)
    eeg_p = float(ind.get('eeg') or 0.0)
    sp_p = float(ind.get('speech') or 0.0)

    cursor.execute('''
        INSERT INTO reports (
            user_id, patient_name, age, gender, diagnosis,
            final_probability_pct, risk_level, confidence_score,
            cognitive_prob, eeg_prob, speech_prob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        patient_name,
        age,
        gender,
        report_dict.get('diagnosis', 'Healthy / Control Baseline') if isinstance(report_dict, dict) else 'Healthy / Control Baseline',
        float(report_dict.get('final_probability_pct', 0.0)) if isinstance(report_dict, dict) else 0.0,
        report_dict.get('risk_level', 'Low') if isinstance(report_dict, dict) else 'Low',
        float(report_dict.get('confidence_score', 90.0)) if isinstance(report_dict, dict) else 90.0,
        cog_p,
        eeg_p,
        sp_p
    ))
    conn.commit()
    report_id = cursor.lastrowid
    conn.close()
    return report_id

def get_user_reports(user_id):
    """Retrieves all past patient reports generated by a specific user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    )
    reports = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return reports

def get_all_reports_admin():
    """Retrieves all patient reports across system for Admin Dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT r.*, u.username as clinician_name 
        FROM reports r 
        LEFT JOIN users u ON r.user_id = u.id 
        ORDER BY r.created_at DESC
    ''')
    reports = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return reports
