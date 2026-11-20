"""
SQLite Database Persistence Layer (database.py)
Manages User Accounts, Authentication Sessions, and Patient Evaluation Report History.
"""

import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from utils.logger import logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "neurocare.db")

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
            role TEXT DEFAULT 'patient',
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

    # 3. Seed Default Admin Account (admin / admin@123) if not already existing
    admin_pass = generate_password_hash("admin@123")
    cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", ("admin",))
    existing_admin = cursor.fetchone()
    if not existing_admin:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("admin", "admin@neurocare.ai", admin_pass, "admin")
        )
        conn.commit()
        logger.info("Default Admin account created: username='admin', password='admin@123'")

    # 4. Seed Pre-configured Doctor Accounts if not already existing
    doc1_pass = generate_password_hash("doctor@1")
    cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", ("Doctor1",))
    existing_doc1 = cursor.fetchone()
    if not existing_doc1:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("Doctor1", "doctor1@neurocare.ai", doc1_pass, "doctor")
        )
        conn.commit()
        logger.info("Doctor1 account created: username='Doctor1', password='doctor@1'")

    doc2_pass = generate_password_hash("doctor@2")
    cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", ("Doctor2",))
    existing_doc2 = cursor.fetchone()
    if not existing_doc2:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("Doctor2", "doctor2@neurocare.ai", doc2_pass, "doctor")
        )
        conn.commit()
        logger.info("Doctor2 account created: username='Doctor2', password='doctor@2'")

    # 5. Seed Pre-configured Patient Accounts if not already existing
    patient_pass = generate_password_hash("patient@123")
    cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", ("patient",))
    existing_patient = cursor.fetchone()
    if not existing_patient:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("patient", "patient@neurocare.ai", patient_pass, "patient")
        )
        conn.commit()
        logger.info("Patient account created: username='patient', password='patient@123'")

    suhan_pass = generate_password_hash("suhan@123")
    cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", ("suhan",))
    existing_suhan = cursor.fetchone()
    if not existing_suhan:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("suhan", "suhan@neurocare.ai", suhan_pass, "patient")
        )
        conn.commit()
        logger.info("Patient account created: username='suhan', password='suhan@123'")

    conn.close()

def register_user(username, email, password, role='patient'):
    """Registers a new user account (default role: patient)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        clean_user = username.strip()
        clean_email = email.strip()
        clean_pwd = password.strip()

        # Check if username or email already exists (case-insensitively)
        cursor.execute(
            "SELECT id, username, email FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)",
            (clean_user, clean_email)
        )
        existing = cursor.fetchone()
        if existing:
            conn.close()
            if existing['username'].lower() == clean_user.lower():
                return False, f"Username '{clean_user}' is already registered. Please choose another or sign in."
            else:
                return False, f"Email '{clean_email}' is already registered. Please sign in or use another email."

        pass_hash = generate_password_hash(clean_pwd)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (clean_user, clean_email, pass_hash, role)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        logger.info(f"New user registered: '{clean_user}' (Role: {role})")
        return True, user_id
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username or Email already exists."
    except Exception as e:
        conn.close()
        return False, str(e)

def verify_user_credentials(username_or_email, password):
    """
    Validates user credentials against SQLite database with case-insensitive username/email matching.
    Returns (user_dict, error_message). If successful, error_message is None.
    """
    if not username_or_email or not str(username_or_email).strip():
        return None, "Please enter your username or email."
    if not password or not str(password).strip():
        return None, "Please enter your password."

    clean_identifier = str(username_or_email).strip()
    clean_password = str(password).strip()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)",
        (clean_identifier, clean_identifier)
    )
    user = cursor.fetchone()
    conn.close()

    if not user:
        return None, f"User account '{clean_identifier}' does not exist. Please check your username/email or register."

    if not check_password_hash(user['password_hash'], clean_password):
        return None, "Incorrect password. Please verify your password and try again."

    return dict(user), None

def authenticate_user(username_or_email, password):
    """Authenticates user, doctor, or admin credentials."""
    user, _ = verify_user_credentials(username_or_email, password)
    return user

def save_patient_report(user_id, patient_name, *args, **kwargs):
    """Saves a generated patient report into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    age = 70
    gender = 'M'
    report_dict = {}

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
        SELECT r.*, u.username as account_username, u.email as account_email, u.role as user_role 
        FROM reports r 
        LEFT JOIN users u ON r.user_id = u.id 
        ORDER BY r.created_at DESC
    ''')
    reports = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return reports

def get_patients_grouped_by_user():
    """
    Retrieves all Patient Account User IDs and all patients/reports under each user ID.
    Returns: List of User Dicts with attached 'reports' list.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE role IN ('patient', 'user') ORDER BY id ASC")
    user_rows = [dict(u) for u in cursor.fetchall()]

    for user in user_rows:
        cursor.execute("SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC", (user['id'],))
        user['reports'] = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return user_rows

def get_doctors_with_patients():
    """
    Retrieves all Doctor Accounts and all patients evaluated under each Doctor.
    Returns: List of Doctor Dicts with attached 'reports' list.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE role = 'doctor' ORDER BY username ASC")
    doctor_rows = [dict(d) for d in cursor.fetchall()]

    for doc in doctor_rows:
        cursor.execute("SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC", (doc['id'],))
        doc['reports'] = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return doctor_rows

def get_admin_stats():
    """Calculates high-level system metrics for Admin Dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as cnt FROM users")
    total_users = cursor.fetchone()['cnt']

    cursor.execute("SELECT COUNT(*) as cnt FROM users WHERE role = 'doctor'")
    total_doctors = cursor.fetchone()['cnt']

    cursor.execute("SELECT COUNT(*) as cnt FROM reports")
    total_reports = cursor.fetchone()['cnt']

    cursor.execute("SELECT COUNT(*) as cnt FROM reports WHERE risk_level = 'High'")
    high_risk = cursor.fetchone()['cnt']

    cursor.execute("SELECT COUNT(*) as cnt FROM reports WHERE risk_level = 'Low'")
    healthy = cursor.fetchone()['cnt']

    conn.close()
    return {
        'total_users': total_users,
        'total_doctors': total_doctors,
        'total_reports': total_reports,
        'high_risk': high_risk,
        'healthy': healthy
    }
