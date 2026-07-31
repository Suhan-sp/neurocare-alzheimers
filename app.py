"""
NeuroCare Patient Health Portal Server (app.py)
Serves user authentication, patient assessment forms,
admin dashboard, and database persistence for generated patient reports.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import io
import pandas as pd
from predict import MultimodalPredictor
from database import init_db, register_user, authenticate_user, save_patient_report, get_user_reports, get_all_reports_admin, get_admin_stats
from utils.logger import logger

app = Flask(__name__)
app.secret_key = 'neurocare_super_secret_key_2026'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # 50 MB Max upload limit

# Ensure upload directory & database exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
try:
    init_db()
except Exception as db_init_err:
    logger.warning(f"Database init warning: {db_init_err}")

# Initialize Master Predictor Engine
predictor = MultimodalPredictor(models_dir="saved_models")

def safe_float(val, default_val=0.0):
    """Safely converts input values to float, handling empty strings and None."""
    try:
        if val is None or str(val).strip() == '':
            return default_val
        return float(val)
    except (ValueError, TypeError):
        return default_val

# --- AUTHENTICATION ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User and Admin sign-in route."""
    next_url = request.args.get('next') or request.form.get('next') or url_for('home')
    message = request.args.get('msg')

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = authenticate_user(username, password)
        if user:
            session['user'] = user
            logger.info(f"User '{user['username']}' logged in successfully.")
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(next_url)
        return render_template('login.html', error="Invalid username or password.", next=next_url)
    return render_template('login.html', msg=message, next=next_url)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """New user registration route."""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        success, msg_or_id = register_user(username, email, password)
        if success:
            user = authenticate_user(username, password)
            session['user'] = user
            logger.info(f"New user registered: '{username}'")
            return redirect(url_for('home'))
        return render_template('register.html', error=msg_or_id)
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Sign-out route."""
    session.pop('user', None)
    return redirect(url_for('login'))

# --- PORTAL & DASHBOARD ROUTES (SIGN IN REQUIRED) ---

@app.route('/')
def home():
    """Renders patient assessment form (Requires Sign In)."""
    if not session.get('user'):
        return redirect(url_for('login', msg="Please sign in or register to take a patient cognitive assessment.", next=url_for('home')))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """Handles patient submission, executes AI models, and saves report to database."""
    if not session.get('user'):
        return redirect(url_for('login', msg="Please sign in to process patient assessments."))

    try:
        logger.info("Received patient assessment request...")

        # Parse Patient-Friendly Inputs
        patient_name = request.form.get('PatientName') or 'Patient Assessment'
        age = safe_float(request.form.get('Age'), default_val=72.0)
        gender = request.form.get('Gender') or 'F'
        educ_level = safe_float(request.form.get('EDUC') or request.form.get('EducationLevel'), default_val=14.0)
        mmse = safe_float(request.form.get('MMSE'), default_val=27.0)
        cdr = safe_float(request.form.get('CDR'), default_val=0.0)

        # Derivation of underlying brain volumetric estimates from clinical profile
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
            'eTIV': etiv,
            'nWBV': nwbv,
            'ASF': asf
        }

        # Parse Optional EEG File Upload
        eeg_file_path = None
        if 'eegFile' in request.files and request.files['eegFile'].filename != '':
            file = request.files['eegFile']
            eeg_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(eeg_file_path)
            logger.info(f"Saved patient EEG file to {eeg_file_path}")

        # Parse Optional Speech Audio File or Microphone Stream
        speech_audio = None
        if 'speechFile' in request.files and request.files['speechFile'].filename != '':
            file = request.files['speechFile']
            speech_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(speech_file_path)
            speech_audio = speech_file_path
            logger.info(f"Saved patient speech audio file to {speech_file_path}")

        # Execute Prediction Pipeline
        results = predictor.predict_all(cog_dict, eeg_file=eeg_file_path, speech_audio=speech_audio)
        results['patient_name'] = patient_name

        # Fail-Safe Database Persistence (Tied to logged-in user account)
        report_id = 1
        try:
            user_id = session.get('user', {}).get('id')
            report_id = save_patient_report(user_id, patient_name, age, gender, results)
            logger.info(f"Patient report #{report_id} saved to database for patient '{patient_name}'")
        except Exception as db_err:
            logger.warning(f"Database save non-blocking warning: {db_err}")

        return render_template('result.html', res=results, patient_name=patient_name, report_id=report_id, cog_input=cog_dict)

    except Exception as e:
        logger.error(f"Error during patient assessment execution: {e}")
        return render_template('index.html', error=f"Assessment Execution Error: {str(e)}")

@app.route('/my-reports')
def my_reports():
    """Displays personal patient report history for logged-in users."""
    if not session.get('user'):
        return redirect(url_for('login'))
    user_id = session['user']['id']
    reports = get_user_reports(user_id)
    return render_template('my_reports.html', reports=reports)

@app.route('/admin')
def admin_dashboard():
    """Admin Dashboard showing all system patient reports and user analytics."""
    if not session.get('user') or session['user'].get('role') != 'admin':
        return render_template('login.html', error="Admin access required. Please sign in as Admin.")
    stats = get_admin_stats()
    reports = get_all_reports_admin()
    return render_template('admin_dashboard.html', stats=stats, reports=reports)

if __name__ == '__main__':
    logger.info("Starting NeuroCare Patient Portal & Admin Server on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
