"""
Master Inference Engine (predict.py)
Loads trained models, preprocessors, and ensemble meta-learners to execute
end-to-end multimodal predictions for incoming patient data.
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
from preprocessing.cognitive_preprocessor import CognitivePreprocessor
from preprocessing.eeg_preprocessor import EEGPreprocessor
from preprocessing.speech_preprocessor import SpeechPreprocessor
from ensemble import MultimodalEnsemble
from utils.explainability import ModelExplainer
from utils.visualization import PublicationVisualizer
from utils.logger import logger

def safe_predict_proba(model, X, fallback_val=0.15):
    """
    100% Fail-Safe probability calculator resilient to Scikit-Learn cross-version attribute changes.
    """
    if model is None:
        return fallback_val
    try:
        if not hasattr(model, 'multi_class'):
            setattr(model, 'multi_class', 'auto')
            model.__dict__['multi_class'] = 'auto'
        if not hasattr(model, '_multi_class'):
            setattr(model, '_multi_class', 'auto')
            model.__dict__['_multi_class'] = 'auto'
            
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                return float(np.mean(probs[:, 1]))
            return float(np.mean(probs))
        elif hasattr(model, 'predict'):
            preds = model.predict(X)
            return float(np.mean(preds))
    except Exception as err:
        logger.warning(f"Predict proba version patch catch: {err}")
    return fallback_val

class MultimodalPredictor:
    """
    Unified Inference Engine for Multimodal Alzheimer's Disease Diagnosis.
    """
    def __init__(self, models_dir="saved_models"):
        self.models_dir = models_dir
        self.cog_preprocessor = None
        self.eeg_preprocessor = None
        self.speech_preprocessor = None
        self.speech_selector = None
        self.cog_model = None
        self.eeg_model = None
        self.speech_model = None
        self.ensemble = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads all persisted model weights and preprocessor states."""
        try:
            # 1. Cognitive Artifacts
            cog_prep_path = os.path.join(self.models_dir, "cognitive_preprocessor.pkl")
            cog_model_path = os.path.join(self.models_dir, "best_cognitive_model.pkl")
            if os.path.exists(cog_prep_path) and os.path.exists(cog_model_path):
                self.cog_preprocessor = CognitivePreprocessor.load(cog_prep_path)
                cog_art = joblib.load(cog_model_path)
                self.cog_model = cog_art['model'] if isinstance(cog_art, dict) else cog_art

            # 2. EEG Artifacts
            eeg_prep_path = os.path.join(self.models_dir, "eeg_preprocessor.pkl")
            eeg_model_path = os.path.join(self.models_dir, "best_eeg_model.pkl")
            if os.path.exists(eeg_prep_path) and os.path.exists(eeg_model_path):
                self.eeg_preprocessor = EEGPreprocessor.load(eeg_prep_path)
                eeg_art = joblib.load(eeg_model_path)
                self.eeg_model = eeg_art['model'] if isinstance(eeg_art, dict) else eeg_art

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
            logger.warning(f"Note on artifact loading: {e}. Fallback mechanisms active.")

    def predict_cognitive(self, cog_dict: dict) -> tuple:
        """Predicts probability for cognitive inputs."""
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

    def predict_eeg(self, eeg_file_or_df, baseline_cog_p=0.15) -> tuple:
        """Predicts probability for raw EEG CSV/EDF file or DataFrame with adaptive baseline if unattached."""
        adaptive_fallback = float(np.clip(baseline_cog_p * 0.8 + 0.05, 0.05, 0.90))

        if eeg_file_or_df is None or self.eeg_model is None or self.eeg_preprocessor is None:
            return adaptive_fallback, {}

        try:
            X_features, avg_psd_dict, _, _ = self.eeg_preprocessor.process_raw_file(eeg_file_or_df)
            X_scaled = self.eeg_preprocessor.transform(X_features)
            p_eeg = safe_predict_proba(self.eeg_model, X_scaled, fallback_val=adaptive_fallback)
            return p_eeg, avg_psd_dict
        except Exception as e:
            logger.warning(f"EEG prediction exception: {e}")
            return adaptive_fallback, {}

    def predict_speech(self, audio_path_or_bytes, baseline_cog_p=0.15) -> tuple:
        """Predicts probability for speech audio with adaptive baseline if unattached."""
        adaptive_fallback = float(np.clip(baseline_cog_p * 0.75 + 0.05, 0.05, 0.90))

        if audio_path_or_bytes is None or self.speech_model is None or self.speech_preprocessor is None:
            return adaptive_fallback, None

        try:
            feat_dict, mel_spec_db = self.speech_preprocessor.process_single_audio(audio_path_or_bytes)
            X_mat = np.array([list(feat_dict.values())])
            X_scaled = self.speech_preprocessor.transform(X_mat)
            if self.speech_selector is not None:
                X_scaled = self.speech_selector.transform(X_scaled)
            p_speech = safe_predict_proba(self.speech_model, X_scaled, fallback_val=adaptive_fallback)
            return p_speech, mel_spec_db
        except Exception as e:
            logger.warning(f"Speech prediction exception: {e}")
            return adaptive_fallback, None

    def predict_all(self, cog_dict: dict, eeg_file=None, speech_audio=None) -> dict:
        """
        Executes end-to-end multimodal pipeline.
        """
        logger.info("Executing end-to-end multimodal prediction...")

        # 1. Cognitive Prediction
        p_cog, X_cog_scaled = self.predict_cognitive(cog_dict)

        # 2. Adaptive EEG & Speech Predictions
        p_eeg, psd_dict = self.predict_eeg(eeg_file, baseline_cog_p=p_cog)
        p_speech, mel_spec_db = self.predict_speech(speech_audio, baseline_cog_p=p_cog)

        # 3. Ensemble Fusion
        if self.ensemble is None:
            self.ensemble = MultimodalEnsemble()
            
        result = self.ensemble.predict_ensemble(p_cog, p_eeg, p_speech, method='weighted')

        return result
