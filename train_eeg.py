"""
EEG Prediction Model Training Pipeline (train_eeg.py)
Processes 88 human subjects from OpenNeuro ds004504 dataset (eeg new dataset/):
- sub-001 to sub-036: Alzheimer's Disease (AD) -> Class 1
- sub-037 to sub-059: Frontotemporal Dementia (FTD) -> Class 1
- sub-060 to sub-088: Healthy Controls (CN) -> Class 0

Extracts 16-channel EEG PSD power bands (Delta, Theta, Alpha, Beta, Gamma), Hjorth parameters,
and spectral slowing ratios across 1,300+ clinical epochs with true ground-truth subject labels.
"""

import os
import glob
import pandas as pd
import numpy as np
import mne
import joblib
from sklearn.model_selection import train_test_split
from preprocessing.eeg_preprocessor import EEGPreprocessor
from models.ml_models import MLModelSuite
from models.deep_models import DeepModelSuite
from utils.visualization import PublicationVisualizer
from utils.logger import logger

def run_eeg_training(dataset_dir="eeg new dataset", output_dir="saved_models"):
    """Runs the complete 88-subject clinical EEG training and model selection pipeline."""
    logger.info("==================================================")
    logger.info("    STARTING CLINICAL EEG MODEL TRAINING PIPELINE ")
    logger.info("==================================================")

    preprocessor = EEGPreprocessor(fs=500.0, l_freq=0.5, h_freq=45.0)
    X_all_epochs = []
    y_all_epochs = []
    sample_psd_dict = None
    healthy_sample_df = None
    ad_sample_df = None

    logger.info(f"Loading 88 human subjects from '{dataset_dir}' with true ground-truth labels...")

    # Iterate through all 88 subjects in OpenNeuro ds004504 dataset
    for sub_num in range(1, 89):
        sub_id = f"sub-{sub_num:03d}"
        set_path = os.path.join(dataset_dir, sub_id, "eeg", f"{sub_id}_task-eyesclosed_eeg.set")
        
        if not os.path.exists(set_path):
            continue

        # True Subject Ground-Truth Mapping
        if sub_num <= 59:
            label = 1 # Class 1: Alzheimer's / Dementia (AD: sub-001 to 036, FTD: sub-037 to 059)
        else:
            label = 0 # Class 0: Healthy Control (CN: sub-060 to 088)

        try:
            # Crop to 30 seconds for sub-second feature extraction
            raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
            if raw.times[-1] > 30.0:
                raw.crop(tmin=0.0, tmax=30.0)
            raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
            
            df = raw.to_data_frame()
            if 'time' in df.columns:
                df = df.drop(columns=['time'])

            # Store sample CSVs for user testing in portal
            if sub_num == 60 and healthy_sample_df is None:
                healthy_sample_df = df.copy()
                healthy_sample_df['status'] = 0
            elif sub_num == 1 and ad_sample_df is None:
                ad_sample_df = df.copy()
                ad_sample_df['status'] = 1

            X_mat, psd_dict, _, _ = preprocessor.process_raw_file(df, window_seconds=2.0)
            
            if len(X_mat) > 0:
                X_all_epochs.append(X_mat)
                y_all_epochs.extend([label] * len(X_mat))
                if sample_psd_dict is None:
                    sample_psd_dict = psd_dict

        except Exception as e:
            logger.warning(f"Error processing {sub_id}: {e}")

    X_concat = np.vstack(X_all_epochs)
    y_concat = np.array(y_all_epochs)

    logger.info(f"Extracted EEG Dataset: {X_concat.shape[0]} total clinical epochs across {X_concat.shape[1]} features.")
    logger.info(f"Ground-Truth Class Balance -> Healthy (0): {sum(y_concat==0)} epochs, Alzheimer's (1): {sum(y_concat==1)} epochs")

    # Save single-patient test CSVs in EEG dataset folder
    os.makedirs("EEG dataset", exist_ok=True)
    if healthy_sample_df is not None:
        healthy_sample_df.to_csv("EEG dataset/single_patient_EEG.csv", index=False)
        healthy_sample_df.to_csv("EEG dataset/healthy_patient_EEG.csv", index=False)
        logger.info("Saved Healthy Control sample EEG to EEG dataset/single_patient_EEG.csv (status=0)")
    if ad_sample_df is not None:
        ad_sample_df.to_csv("EEG dataset/alzheimer_patient_EEG.csv", index=False)
        logger.info("Saved Alzheimer's Patient sample EEG to EEG dataset/alzheimer_patient_EEG.csv (status=1)")

    # Fit & Scale Features
    X_scaled = preprocessor.fit_transform(X_concat)
    preprocessor.save(os.path.join(output_dir, "eeg_preprocessor.pkl"))

    # 3. Train & Evaluate ML Model Suite
    ml_suite = MLModelSuite()
    ml_results, (best_ml_name, best_ml_model) = ml_suite.train_and_evaluate_all(X_scaled, y_concat, cv_splits=5)
    best_ml_score = ml_results[best_ml_name]['roc_auc']

    # 4. Train Deep Learning Architectures
    X_tr, X_va, y_tr, y_va = train_test_split(X_scaled, y_concat, test_size=0.20, random_state=42, stratify=y_concat)
    dl_results, (best_dl_name, best_dl_model) = DeepModelSuite.train_and_evaluate_deep_models(
        X_tr, y_tr, X_va, y_va, epochs=20, batch_size=32
    )
    best_dl_score = dl_results[best_dl_name]['roc_auc']

    # 5. Select Winner Model
    if best_dl_score > best_ml_score:
        overall_best_name = f"Deep Learning ({best_dl_name})"
        overall_best_model = best_dl_model
        overall_probs = dl_results[best_dl_name]['val_probs']
        y_test_eval = y_va
    else:
        overall_best_name = f"ML ({best_ml_name})"
        overall_best_model = best_ml_model
        overall_probs = ml_results[best_ml_name]['oof_probs']
        y_test_eval = y_concat

    logger.info(f"CLINICAL EEG SELECTION: Winner is '{overall_best_name}' with ROC-AUC: {max(best_ml_score, best_dl_score):.4f}")

    # 6. Save Model Artifact
    joblib.dump({'model_name': overall_best_name, 'model': overall_best_model, 'feature_names': preprocessor.feature_names},
                os.path.join(output_dir, "best_eeg_model.pkl"))
    logger.info(f"Saved best EEG model ({overall_best_name}) to {output_dir}/best_eeg_model.pkl")

    # 7. Generate Visualizations
    vis = PublicationVisualizer()
    vis.plot_roc_curve(y_test_eval, overall_probs, "EEG Signal Model", "static/plots/eeg_roc.png")
    if sample_psd_dict is not None:
        vis.plot_eeg_power_spectrum(sample_psd_dict, save_path="static/plots/eeg_psd.png")

    logger.info("==================================================")
    logger.info(f"  EEG TRAINING COMPLETE: Best = {overall_best_name} (AUC: {max(best_ml_score, best_dl_score):.4f})")
    logger.info("==================================================")

if __name__ == '__main__':
    run_eeg_training()
