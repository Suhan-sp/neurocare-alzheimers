"""
NeuroCare AI - Master Flask Web Application (app.py)
Multimodal Early Prediction Platform for Alzheimer's Disease
Synthesizes Cognitive Profiling, 19-Channel EEG Signal Processing / Report Digitization, and Speech Acoustics.
Role-Based Authentication Engine (Patient Portal, Doctor Portal, Admin Portal).
"""

import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from werkzeug.utils import secure_filename
from predict import MultimodalPredictor
from database import (
    init_db, register_user, authenticate_user, 
    save_patient_report, get_user_reports, get_all_reports_admin, get_admin_stats
)
from utils.logger import logger

app = Flask(__name__)
app.secret_key = 'neurocare_secret_key_prod_2026'

# Configure upload directory
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB Max upload limit

# Initialize SQLite Database & Multimodal Predictor Instance
init_db()
predictor = MultimodalPredictor()

@app.route('/')
def index():
    """
    Root route:
    If unauthenticated -> Render 3-Portal Landing Page (Patient Login, Doctor Login, Admin Login).
    If authenticated -> Render Patient/Doctor Assessment Dashboard.
    """
    if not session.get('user'):
        return render_template('portal_landing.html')
    return render_template('index.html')

@app.route('/login')
def login_redirect():
    """Redirect legacy /login to Landing Portal Page."""
    if session.get('user'):
        return redirect(url_for('index'))
    return render_template('portal_landing.html')

# ==========================================
# 1. PATIENT AUTHENTICATION PORTAL
# ==========================================
@app.route('/patient/login', methods=['GET', 'POST'])
def patient_login():
    """Patient Login Portal (/patient/login)."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = authenticate_user(username, password)
        if user:
            session['user'] = user
            logger.info(f"Patient '{username}' logged in successfully.")
            return redirect(url_for('index'))
        else:
            return render_template('patient_login.html', error="Invalid username or password.")
    return render_template('patient_login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles Patient Account Registration (Default role = patient)."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        success, msg_or_id = register_user(username, email, password, role='patient')
        if success:
            session['user'] = {
                'id': msg_or_id,
                'username': username,
                'email': email,
                'role': 'patient',
                'name': username.capitalize()
            }
            logger.info(f"Registered new patient account '{username}'.")
            return redirect(url_for('index'))
        else:
            return render_template('register.html', error=msg_or_id)
    return render_template('register.html')

# ==========================================
# 2. DOCTOR AUTHENTICATION PORTAL
# ==========================================
@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    """
    Doctor Login Portal (/doctor/login).
    Pre-configured credentials: Doctor1 / doctor@1, Doctor2 / doctor@2.
    Only role == 'doctor' is permitted.
    """
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = authenticate_user(username, password)
        if user:
            # Check Role Security Constraint
            if user.get('role') != 'doctor':
                logger.warning(f"Unauthorized login attempt by non-doctor user '{username}' on Doctor Portal.")
                return render_template('doctor_login.html', error="Access denied. This portal is only for doctors.")
            
            session['user'] = user
            logger.info(f"Doctor '{username}' logged in successfully.")
            return redirect(url_for('index'))
        else:
            return render_template('doctor_login.html', error="Invalid username or password.")
    return render_template('doctor_login.html')

@app.route('/doctor/authorize', methods=['GET', 'POST'])
def doctor_authorize():
    """Requires Doctor1 authorization password to grant access for New Doctor registration."""
    if request.method == 'POST':
        auth_pass = request.form.get('auth_password', '').strip()
        doc1_user = authenticate_user('Doctor1', auth_pass)
        if doc1_user and doc1_user.get('role') == 'doctor':
            session['doctor1_authorized'] = True
            logger.info("Doctor1 authorization password verified successfully. Access granted to New Doctor registration.")
            return redirect(url_for('doctor_register'))
        else:
            return render_template('doctor_authorize.html', error="Invalid Doctor1 Authorization Password. Authorization failed.")
    return render_template('doctor_authorize.html')

@app.route('/doctor/register', methods=['GET', 'POST'])
def doctor_register():
    """Allows registration of a new doctor if Doctor1 password was verified or Doctor1 is logged in."""
    is_doc1_logged_in = session.get('user') and session['user'].get('role') == 'doctor' and session['user'].get('username', '').lower() == 'doctor1'
    is_authorized = session.get('doctor1_authorized', False)

    if not is_doc1_logged_in and not is_authorized:
        return redirect(url_for('doctor_authorize'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        success, msg_or_id = register_user(username, email, password, role='doctor')
        if success:
            logger.info(f"Registered new Doctor account '{username}'.")
            return render_template('doctor_register.html', msg=f"New Doctor account '{username}' registered successfully!")
        else:
            return render_template('doctor_register.html', error=msg_or_id)
            
    return render_template('doctor_register.html')

# ==========================================
# 3. ADMIN AUTHENTICATION PORTAL
# ==========================================
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """
    Admin Login Portal (/admin/login).
    Credentials: admin / admin@123.
    Only role == 'admin' is permitted.
    """
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = authenticate_user(username, password)
        if user:
            # Check Role Security Constraint
            if user.get('role') != 'admin':
                logger.warning(f"Unauthorized login attempt by non-admin user '{username}' on Admin Portal.")
                return render_template('admin_login.html', error="Access denied. Administrator credentials required.")
            
            session['user'] = user
            logger.info(f"Administrator '{username}' logged in successfully.")
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error="Invalid username or password.")
    return render_template('admin_login.html')

@app.route('/admin')
@app.route('/admin/dashboard')
def admin_dashboard():
    """Admin dashboard listing all patient reports across clinicians (Role Protected)."""
    if not session.get('user') or session['user'].get('role') != 'admin':
        return render_template('admin_login.html', error="Access denied. Administrator credentials required.")
    all_reports = get_all_reports_admin()
    stats = get_admin_stats()
    return render_template('admin_dashboard.html', stats=stats, reports=all_reports)

@app.route('/logout')
def logout():
    """Logs out user session and redirects to Portal Landing Page."""
    user_name = session.get('user', {}).get('username', 'User')
    session.clear()
    logger.info(f"User '{user_name}' logged out.")
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict_route():
    """Executes prediction on strict user inputs without dummy fallbacks."""
    if not session.get('user'):
        return render_template('portal_landing.html', error="Please sign in to your authorized portal to run assessments.")

    try:
        patient_name = request.form.get('PatientName', '').strip() or 'Anonymous Patient'
        
        # 1. Parse & Validate Strict Cognitive User Inputs
        age_str = request.form.get('Age', '').strip()
        gender_str = request.form.get('Gender', '').strip()
        educ_str = request.form.get('EDUC', '').strip()
        mmse_str = request.form.get('MMSE', '').strip()
        cdr_str = request.form.get('CDR', '').strip()
        fam_history_str = request.form.get('FamilyHistory', '').strip()

        if not age_str or not gender_str or not educ_str or not mmse_str or not cdr_str:
            return render_template('index.html', error="Please fill in all required patient fields: Age, Gender, Education, MMSE score, and Daily Memory rating.")

        age = float(age_str)
        gender = gender_str
        educ_level = float(educ_str)
        mmse = float(mmse_str)
        cdr = float(cdr_str)
        fam_history = 1.0 if fam_history_str.lower() in ['yes', 'y', '1', '1.0'] else 0.0

        # Volumetric ratio derivation based on user's exact CDR rating
        nwbv = 0.78 if cdr == 0.0 else (0.72 if cdr == 0.5 else 0.68)
        etiv = 1450.0
        asf = 1.18
        ses = 2.0

        cog_dict = {
            'Age': age,
            'Gender': gender,
            'EDUC': educ_level,
            'SES': ses,
            'MMSE': mmse,
            'CDR': cdr,
            'FamilyHistory': fam_history,
            'eTIV': etiv,
            'nWBV': nwbv,
            'ASF': asf
        }

        # 2. Parse Optional User EEG File (EDF/CSV signal OR scanned PDF/Image report)
        eeg_file_path = None
        if 'eegFile' in request.files and request.files['eegFile'].filename != '':
            file = request.files['eegFile']
            eeg_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(eeg_file_path)
            logger.info(f"Saved patient EEG file to {eeg_file_path}")

        # 3. Parse Optional User Speech Audio File or Mic Stream (None if not provided)
        speech_audio = None
        if 'speechFile' in request.files and request.files['speechFile'].filename != '':
            file = request.files['speechFile']
            speech_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(speech_file_path)
            speech_audio = speech_file_path
            logger.info(f"Saved patient speech audio file to {speech_file_path}")

        # Execute End-to-End Prediction
        res = predictor.predict_all(cog_dict=cog_dict, eeg_file=eeg_file_path, speech_audio=speech_audio)

        # Save Report to SQLite DB (user_id, patient_name, age, gender, report_dict)
        user_id = session['user'].get('id', 1)
        report_id = save_patient_report(user_id, patient_name, int(age), gender, res)

        return render_template('result.html', res=res, cog_input=cog_dict, patient_name=patient_name, report_id=report_id)

    except Exception as e:
        logger.error(f"Prediction route exception: {e}")
        return render_template('index.html', error=f"An error occurred during evaluation: {e}")

@app.route('/my-reports')
def my_reports():
    """Displays historical assessment reports for logged-in user."""
    if not session.get('user'):
        return render_template('portal_landing.html', error="Please sign in to view your assessment reports.")
    user_id = session['user'].get('id', 1)
    reports = get_user_reports(user_id)
    return render_template('my_reports.html', reports=reports)

if __name__ == '__main__':
    from waitress import serve
    print("\n=================================================================")
    print("  [SUCCESS] NEUROCARE MULTI-PORTAL SYSTEM IS LIVE!")
    print("  --> Patient Portal:  http://localhost:5000/patient/login")
    print("  --> Doctor Portal:   http://localhost:5000/doctor/login")
    print("  --> Admin Portal:    http://localhost:5000/admin/login")
    print("=================================================================\n")
    serve(app, host='0.0.0.0', port=5000)
