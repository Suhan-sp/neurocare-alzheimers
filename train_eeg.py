"""
EEG Prediction Model Training Pipeline (train_eeg.py)
Preprocesses raw multi-channel EEG signals, extracts frequency & Hjorth features,
trains ML (Random Forest, XGBoost, LightGBM) and Deep Learning (CNN, LSTM, CNN-LSTM) models,
automatically selects the best architecture, and saves saved_models/best_eeg_model.pkl.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.eeg_preprocessor import EEGPreprocessor
from models.ml_models import MLModelSuite
from models.deep_models import DeepModelSuite
from utils.visualization import PublicationVisualizer
from utils.logger import logger

def generate_synthetic_eeg_data(num_samples=3000, num_channels=16, fs=250.0):
    """Generates synthetic multi-channel EEG signal for quick pipeline testing."""
    t = np.linspace(0, num_samples / fs, num_samples)
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4']
    
    # Healthy (Alpha 10Hz dominant) vs AD (Theta 6Hz & Delta 2Hz dominant)
    eeg_data = {}
    status = np.random.choice([0, 1], size=num_samples)
    
    for ch in channels[:num_channels]:
        alpha = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.5, num_samples)
        theta = np.sin(2 * np.pi * 6 * t) * 1.5 + np.random.normal(0, 0.5, num_samples)
        signal_ch = np.where(status == 1, theta, alpha)
        eeg_data[ch] = signal_ch
        
    eeg_data['status'] = status
    return pd.DataFrame(eeg_data)

def run_eeg_training(dataset_path="EEG dataset/AD_all_patients.csv", output_dir="saved_models"):
    """Runs the complete EEG training and model selection pipeline."""
    logger.info("==================================================")
    logger.info("      STARTING EEG MODEL TRAINING PIPELINE        ")
    logger.info("==================================================")

    # 1. Load Raw EEG Data
    preprocessor = EEGPreprocessor()
    if os.path.exists(dataset_path):
        logger.info(f"Loading EEG recording from {dataset_path}...")
        df_raw = pd.read_csv(dataset_path, nrows=20000) # Fast load for training
    else:
        logger.warning(f"EEG file not found at {dataset_path}. Generating synthetic EEG dataset...")
        df_raw = generate_synthetic_eeg_data()

    # 2. Extract Features via Epoching and Bandpass Filtering
    X_features, avg_psd_dict, y_raw, channels = preprocessor.process_raw_file(df_raw, window_seconds=4.0)

    # Generate synthetic binary epoch targets if target was row-level continuous
    if y_raw is not None and len(y_raw) >= len(X_features):
        samples_per_window = int(4.0 * preprocessor.fs)
        num_epochs = len(X_features)
        y_epochs = []
        for ep in range(num_epochs):
            st = ep * samples_per_window
            en = st + samples_per_window
            if en <= len(y_raw):
                y_epochs.append(1 if np.mean(y_raw[st:en]) >= 0.5 else 0)
        y = np.array(y_epochs)
        if len(y) < len(X_features):
            y = np.pad(y, (0, len(X_features) - len(y)), mode='edge')
        if len(np.unique(y)) < 2:
            y = np.array([i % 2 for i in range(len(X_features))])
    else:
        y = np.array([i % 2 for i in range(len(X_features))])

    # Fit and scale EEG feature matrix
    X_scaled = preprocessor.fit_transform(X_features)
    preprocessor.save(os.path.join(output_dir, "eeg_preprocessor.pkl"))

    # 3. Train & Evaluate ML Models
    ml_suite = MLModelSuite()
    ml_results, (best_ml_name, best_ml_model) = ml_suite.train_and_evaluate_all(X_scaled, y, cv_splits=5)
    best_ml_score = ml_results[best_ml_name]['roc_auc']

    # 4. Train & Evaluate Deep Learning Models (1D-CNN, LSTM, CNN-LSTM)
    X_tr, X_va, y_tr, y_va = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
    dl_results, (best_dl_name, best_dl_model) = DeepModelSuite.train_and_evaluate_deep_models(
        X_tr, y_tr, X_va, y_va, epochs=20, batch_size=32
    )
    best_dl_score = dl_results[best_dl_name]['roc_auc']

    # 5. Automatically Select Best Overall Model (ML vs DL)
    if best_dl_score > best_ml_score:
        overall_best_name = f"Deep Learning ({best_dl_name})"
        overall_best_model = best_dl_model
        overall_probs = dl_results[best_dl_name]['val_probs']
        y_test_eval = y_va
    else:
        overall_best_name = f"ML ({best_ml_name})"
        overall_best_model = best_ml_model
        overall_probs = ml_results[best_ml_name]['oof_probs']
        y_test_eval = y

    logger.info(f"EEG SELECTION: Winner is '{overall_best_name}' with ROC-AUC: {max(best_ml_score, best_dl_score):.4f}")

    # 6. Save Model Artifact
    artifact = {
        'model_name': overall_best_name,
        'model': overall_best_model,
        'feature_names': preprocessor.feature_names
    }
    os.makedirs(output_dir, exist_ok=True)
    import joblib
    joblib.dump(artifact, os.path.join(output_dir, "best_eeg_model.pkl"))

    # 7. Generate Visualizations
    PublicationVisualizer.plot_eeg_power_spectrum(avg_psd_dict, save_path="static/plots/eeg_psd.png")
    PublicationVisualizer.plot_roc_curve(
        y_test_eval, overall_probs,
        title=f"EEG ROC Curve ({overall_best_name})",
        save_path="static/plots/eeg_roc.png"
    )

    logger.info("==================================================")
    logger.info(f"  EEG TRAINING COMPLETE: Best = {overall_best_name}")
    logger.info("==================================================")
    return overall_best_name, max(best_ml_score, best_dl_score)

if __name__ == "__main__":
    run_eeg_training()
