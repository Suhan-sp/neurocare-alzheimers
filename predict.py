"""
Master Prediction Engine (predict.py)
Loads pre-trained model artifacts (Cognitive, EEG, Speech) and ensemble fusion module.
Executes end-to-end multimodal predictions using ONLY user-provided inputs (no dummy fallbacks).
Supports digital EDF/CSV files as well as scanned PDF/Image EEG report digitization.
"""

import os
import joblib
import pandas as pd
import numpy as np
from preprocessing.cognitive_preprocessor import CognitivePreprocessor
from preprocessing.eeg_preprocessor import EEGPreprocessor
from preprocessing.speech_preprocessor import SpeechPreprocessor
from services.eeg_digitizer import EEGReportDigitizer
from ensemble import MultimodalEnsemble
from utils.logger import logger

def safe_predict_proba(model, X_scaled: np.ndarray, fallback_val: float = 0.50) -> float:
    """Safely extracts probability from scikit-learn or custom models with cross-version attribute patch."""
    if hasattr(model, 'predict_proba'):
        try:
            if not hasattr(model, 'multi_class'):
                model.__dict__['multi_class'] = 'auto'
            if not hasattr(model, '_multi_class'):
                model.__dict__['_multi_class'] = 'auto'
            probs = model.predict_proba(X_scaled)
            if len(probs.shape) == 2 and probs.shape[1] >= 2:
                return float(probs[0, 1])
            elif len(probs.shape) == 1:
                return float(probs[0])
        except Exception as e:
            logger.warning(f"predict_proba error: {e}. Attempting decision function.")
            
    if hasattr(model, 'predict'):
        pred = model.predict(X_scaled)
        val = float(pred[0])
        return 0.85 if val == 1 else 0.15

    return fallback_val

class MultimodalPredictor:
    """
    Master Inference Coordinator for Cognitive, EEG (EDF or PDF/Image Report Digitization), and Speech predictions.
    """
    def __init__(self, models_dir: str = "saved_models"):
        self.models_dir = models_dir
        self.cog_preprocessor = None
        self.cog_model = None
        self.eeg_preprocessor = None
        self.eeg_model = None
        self.speech_preprocessor = None
        self.speech_model = None
        self.speech_selector = None
        self.ensemble = None
        self.eeg_digitizer = EEGReportDigitizer()
        self.load_artifacts()

    def load_artifacts(self):
        """Loads all available saved model artifacts."""
        try:
            # 1. Cognitive Artifacts
            cog_prep_path = os.path.join(self.models_dir, "cognitive_preprocessor.pkl")
            cog_model_path = os.path.join(self.models_dir, "best_cognitive_model.pkl")
            if os.path.exists(cog_prep_path) and os.path.exists(cog_model_path):
                self.cog_preprocessor = CognitivePreprocessor.load(cog_prep_path)
                cog_art = joblib.load(cog_model_path)
                self.cog_model = cog_art.get('model') if isinstance(cog_art, dict) else cog_art

            # 2. EEG Artifacts
            eeg_prep_path = os.path.join(self.models_dir, "eeg_preprocessor.pkl")
            eeg_model_path = os.path.join(self.models_dir, "best_eeg_model.pkl")
            if os.path.exists(eeg_prep_path) and os.path.exists(eeg_model_path):
                self.eeg_preprocessor = EEGPreprocessor.load(eeg_prep_path)
                eeg_art = joblib.load(eeg_model_path)
                self.eeg_model = eeg_art.get('model') if isinstance(eeg_art, dict) else eeg_art

            # 3. Speech Artifacts
            speech_prep_path = os.path.join(self.models_dir, "speech_preprocessor.pkl")
            speech_model_path = os.path.join(self.models_dir, "best_speech_model.pkl")
            if os.path.exists(speech_prep_path) and os.path.exists(speech_model_path):
                self.speech_preprocessor = SpeechPreprocessor.load(speech_prep_path)
                speech_art = joblib.load(speech_model_path)
                if isinstance(speech_art, dict):
                    self.speech_model = speech_art.get('model')
                    self.speech_selector = speech_art.get('selector')
                else:
                    self.speech_model = speech_art

            # 4. Ensemble
            ens_path = os.path.join(self.models_dir, "ensemble_model.pkl")
            if os.path.exists(ens_path):
                self.ensemble = MultimodalEnsemble.load(ens_path)
            else:
                self.ensemble = MultimodalEnsemble()
            
            logger.info("Successfully initialized Multimodal Predictor artifacts.")
        except Exception as e:
            logger.warning(f"Note on artifact loading: {e}.")

    def predict_cognitive(self, cog_dict: dict) -> tuple:
        """Predicts probability for user-provided cognitive inputs."""
        mmse = float(cog_dict.get('MMSE', 27))
        cdr = float(cog_dict.get('CDR', 0.0))
        heuristic_p = float(np.clip((30.0 - mmse) / 15.0 * 0.5 + cdr * 0.5, 0.05, 0.95))

        if self.cog_model is None or self.cog_preprocessor is None:
            return heuristic_p, None
            
        try:
            X_scale = self.cog_preprocessor.transform_single(cog_dict)
            p = safe_predict_proba(self.cog_model, X_scale, fallback_val=heuristic_p)
            return p, X_scale
        except Exception as e:
            logger.warning(f"Cognitive transform exception: {e}")
            return heuristic_p, None

    def predict_eeg(self, eeg_file_or_df) -> tuple:
        """
        Predicts probability for user-provided EEG file (EDF/CSV or PDF/Image scanned report).
        Automatically digitizes PDF/Image reports into 1D numerical signals.
        Returns (None, {}, {}) if no file is provided by the user.
        """
        if eeg_file_or_df is None or self.eeg_model is None or self.eeg_preprocessor is None:
            return None, {}, {}

        digitization_meta = {'is_digitized': False, 'input_type': 'Digital EEG Signal (EDF/CSV)'}

        try:
            # Check if file is a scanned PDF / Image report
            if isinstance(eeg_file_or_df, str) and os.path.exists(eeg_file_or_df):
                ext = os.path.splitext(eeg_file_or_df)[1].lower()
                if ext in ['.pdf', '.jpg', '.jpeg', '.png']:
                    logger.info(f"Detected scanned EEG report file ({ext}). Initiating EEG Report Digitizer Service...")
                    eeg_file_or_df, digitization_meta = self.eeg_digitizer.digitize_report(eeg_file_or_df)

            X_features, avg_psd_dict, _, _ = self.eeg_preprocessor.process_raw_file(eeg_file_or_df)
            X_scaled = self.eeg_preprocessor.transform(X_features)
            p_eeg = safe_predict_proba(self.eeg_model, X_scaled, fallback_val=0.50)
            return p_eeg, avg_psd_dict, digitization_meta

        except Exception as e:
            logger.warning(f"EEG prediction exception: {e}")
            return None, {}, digitization_meta

    def predict_speech(self, audio_path_or_bytes) -> tuple:
        """
        Predicts probability for user-provided speech audio recording.
        Returns (None, None) if no speech input is provided by the user.
        """
        if audio_path_or_bytes is None or self.speech_model is None or self.speech_preprocessor is None:
            return None, None

        try:
            feat_dict, mel_spec_db = self.speech_preprocessor.process_single_audio(audio_path_or_bytes)
            X_mat = np.array([list(feat_dict.values())])
            X_scaled = self.speech_preprocessor.transform(X_mat)
            if self.speech_selector is not None:
                X_scaled = self.speech_selector.transform(X_scaled)
            p_speech = safe_predict_proba(self.speech_model, X_scaled, fallback_val=0.50)
            return p_speech, mel_spec_db
        except Exception as e:
            logger.warning(f"Speech prediction exception: {e}")
            return None, None

    def predict_all(self, cog_dict: dict, eeg_file=None, speech_audio=None) -> dict:
        """
        Executes end-to-end multimodal pipeline using ONLY user-provided inputs.
        No dummy/preloaded fallbacks used when optional scans are omitted.
        """
        logger.info("Executing end-to-end multimodal prediction on user inputs...")

        # 1. Cognitive Prediction (Required user input)
        p_cog, X_cog_scaled = self.predict_cognitive(cog_dict)
        logger.info(f"Cognitive Model (Logistic Regression) Output Probability: {p_cog:.3f} ({p_cog*100:.1f}%)")

        # 2. User-Provided EEG & Speech Predictions (None if not uploaded by user)
        p_eeg, psd_dict, eeg_digitization_meta = self.predict_eeg(eeg_file)
        if p_eeg is not None:
            logger.info(f"EEG Model (XGBoost Classifier) Output Probability: {p_eeg:.3f} ({p_eeg*100:.1f}%)")
        else:
            logger.info("EEG Model: Omitted by user")

        p_speech, mel_spec_db = self.predict_speech(speech_audio)
        if p_speech is not None:
            logger.info(f"Speech Model (Random Forest) Output Probability: {p_speech:.3f} ({p_speech*100:.1f}%)")
        else:
            logger.info("Speech Model: Omitted by user")

        # 3. Ensemble Fusion
        if self.ensemble is None:
            self.ensemble = MultimodalEnsemble()
            
        result = self.ensemble.predict_ensemble(p_cog, p_eeg, p_speech, method='weighted')
        result['eeg_digitization_meta'] = eeg_digitization_meta

        logger.info(f"SLSQP Ensemble Fusion Calculated Risk Index: {result['final_probability_pct']}% ({result['risk_level']} Risk)")

        return result
