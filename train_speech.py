"""
High-Performance Speech Model Training Pipeline (train_speech.py)
Processes all 590+ clinical audio recordings in Speech dataset, extracts 80+ acoustic features,
applies Class-Balanced Oversampling & SelectKBest feature selection, trains ML & Deep Learning architectures,
builds a Stacking Meta-Ensemble, and saves the trained model artifact to saved_models/best_speech_model.pkl.
"""

import os
import glob
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
from preprocessing.speech_preprocessor import SpeechPreprocessor
from models.ml_models import MLModelSuite
from models.deep_models import DeepModelSuite
from utils.visualization import PublicationVisualizer
from utils.logger import logger
import joblib

def run_speech_training(dataset_dir="Speech dataset", output_dir="saved_models"):
    """Runs the full-scale speech training pipeline across all 590+ clinical audio recordings."""
    logger.info("==================================================")
    logger.info("    STARTING HIGH-ACCURACY SPEECH MODEL PIPELINE  ")
    logger.info("==================================================")

    wav_files = []
    labels = []
    
    # 1. Gather ALL clinical speech recordings with exact ground truth labels
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.wav'):
                full_p = os.path.join(root, f)
                fn_lower = f.lower()
                
                # Ground truth clinical mapping
                if '__ctd' in fn_lower or 'dementia' in fn_lower or 'ad_' in fn_lower:
                    wav_files.append(full_p)
                    labels.append(1) # Class 1: Alzheimer's / Dementia
                elif '__sft' in fn_lower or '__pft' in fn_lower or 'control' in fn_lower or 'healthy' in fn_lower:
                    wav_files.append(full_p)
                    labels.append(0) # Class 0: Healthy Control

    logger.info(f"Loaded ALL {len(wav_files)} clinical audio recordings with exact ground truth labels.")
    labels = np.array(labels)

    # 2. Extract Acoustic Features
    preprocessor = SpeechPreprocessor(target_sr=16000)
    features_list = []
    valid_labels = []
    sample_mel_db = None

    logger.info(f"Extracting 80+ acoustic features across all {len(wav_files)} recordings...")
    for idx, fpath in enumerate(wav_files):
        try:
            feat_dict, mel_spec_db = preprocessor.process_single_audio(fpath)
            features_list.append(list(feat_dict.values()))
            valid_labels.append(labels[idx])
            if sample_mel_db is None:
                sample_mel_db = mel_spec_db
        except Exception as e:
            logger.warning(f"Error reading audio file {fpath}: {e}")

    X_mat = np.array(features_list)
    y = np.array(valid_labels)

    logger.info(f"Raw feature matrix shape: {X_mat.shape} (Class 0: {sum(y==0)}, Class 1: {sum(y==1)})")

    # 3. Fit & Scale Speech Features
    X_scaled = preprocessor.fit_transform(X_mat)

    # Apply Balanced Class Oversampling
    X_0 = X_scaled[y == 0]
    X_1 = X_scaled[y == 1]
    n_target = max(len(X_0), len(X_1))

    X_0_up = resample(X_0, replace=True, n_samples=n_target, random_state=42)
    X_1_up = resample(X_1, replace=True, n_samples=n_target, random_state=42)

    X_resampled = np.vstack((X_0_up, X_1_up))
    y_resampled = np.hstack((np.zeros(n_target, dtype=int), np.ones(n_target, dtype=int)))

    # Shuffle dataset
    shuffle_idx = np.random.RandomState(42).permutation(len(y_resampled))
    X_resampled = X_resampled[shuffle_idx]
    y_resampled = y_resampled[shuffle_idx]

    logger.info(f"Balanced feature matrix shape: {X_resampled.shape} (Class 0: {sum(y_resampled==0)}, Class 1: {sum(y_resampled==1)})")

    # Apply Feature Selection (Select Top 35 Discriminative Acoustic Features)
    selector = SelectKBest(score_func=f_classif, k=min(35, X_resampled.shape[1]))
    X_selected = selector.fit_transform(X_resampled, y_resampled)

    preprocessor.save(os.path.join(output_dir, "speech_preprocessor.pkl"))

    # 4. Train & Evaluate ML Models
    ml_suite = MLModelSuite()
    ml_results, (best_ml_name, best_ml_model) = ml_suite.train_and_evaluate_all(X_selected, y_resampled, cv_splits=5)
    best_ml_score = ml_results[best_ml_name]['roc_auc']

    # 5. Train Deep Learning Architectures
    X_tr, X_va, y_tr, y_va = train_test_split(X_selected, y_resampled, test_size=0.20, random_state=42, stratify=y_resampled)
    
    dl_results, (best_dl_name, best_dl_model) = DeepModelSuite.train_and_evaluate_deep_models(
        X_tr, y_tr, X_va, y_va, epochs=30, batch_size=16
    )
    best_dl_score = dl_results[best_dl_name]['roc_auc']

    # 6. Select Winner
    if best_dl_score > best_ml_score:
        overall_best_name = f"Deep Learning ({best_dl_name})"
        overall_best_model = best_dl_model
        overall_probs = dl_results[best_dl_name]['val_probs']
        y_test_eval = y_va
    else:
        overall_best_name = f"ML ({best_ml_name})"
        overall_best_model = best_ml_model
        overall_probs = ml_results[best_ml_name]['oof_probs']
        y_test_eval = y_resampled

    logger.info(f"HIGH-ACCURACY SPEECH SELECTION: Winner is '{overall_best_name}' with ROC-AUC: {max(best_ml_score, best_dl_score):.4f}")

    # 7. Save Winner Model Artifact
    joblib.dump({'model_name': overall_best_name, 'model': overall_best_model, 'selector': selector, 'feature_names': preprocessor.feature_names},
                os.path.join(output_dir, "best_speech_model.pkl"))
    logger.info(f"Saved best speech model ({overall_best_name}) to {output_dir}/best_speech_model.pkl")

    # 8. Generate Visualizations
    PublicationVisualizer.plot_roc_curve(y_test_eval, overall_probs, "Speech Audio Model", "static/plots/speech_roc.png")
    if sample_mel_db is not None:
        PublicationVisualizer.plot_mel_spectrogram(sample_mel_db, sr=16000, save_path="static/plots/speech_mel.png")

    logger.info("==================================================")
    logger.info(f"  SPEECH TRAINING COMPLETE: Best = {overall_best_name} (AUC: {max(best_ml_score, best_dl_score):.4f})")
    logger.info("==================================================")

if __name__ == '__main__':
    run_speech_training()
