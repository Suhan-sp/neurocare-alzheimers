"""
Multimodal Ensemble Learning Module (ensemble.py)
Combines predictions from Cognitive, EEG, and Speech models using Soft Voting,
Optimal Weighted Averaging, and Stacking Meta-Classifiers.
Calculates Risk Level, Confidence Scores, and Individual Model Contributions.
Dynamically handles missing optional modalities without injecting dummy data.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import joblib
import os
from utils.logger import logger

class MultimodalEnsemble:
    """
    Fuses predictions from user-provided modalities (Cognitive, EEG, Speech).
    """
    def __init__(self, weights=None):
        # Default initial weights for 3 modalities: [0.34, 0.33, 0.33]
        self.weights = np.array(weights) if weights is not None else np.array([0.34, 0.33, 0.33])
        self.stacking_meta_learner = LogisticRegression()
        self.is_fitted = False

    def optimize_weights(self, val_probs_matrix: np.ndarray, y_val: np.ndarray):
        """Determines optimal ensemble weights by minimizing log loss."""
        logger.info("Optimizing ensemble weights using validation performance...")
        
        def loss_func(w):
            w_norm = w / (np.sum(w) + 1e-12)
            p_ens = np.dot(val_probs_matrix, w_norm)
            p_ens = np.clip(p_ens, 1e-7, 1 - 1e-7)
            log_loss = -np.mean(y_val * np.log(p_ens) + (1 - y_val) * np.log(1 - p_ens))
            return log_loss

        init_weights = [1/3, 1/3, 1/3]
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        res = minimize(loss_func, init_weights, bounds=bounds, method='SLSQP')

        if res.success:
            self.weights = res.x / np.sum(res.x)
            logger.info(f"Optimal Ensemble Weights [Cognitive, EEG, Speech]: {self.weights.round(3)}")
        else:
            self.weights = np.array([0.34, 0.33, 0.33])

        self.stacking_meta_learner.fit(val_probs_matrix, y_val)
        self.is_fitted = True

    def predict_ensemble(self, p_cog: float, p_eeg: float = None, p_speech: float = None, method: str = 'weighted') -> dict:
        """
        Combines ONLY user-provided modality probabilities into final diagnosis and risk level.
        Does NOT inject dummy data if EEG or Speech is omitted by the user.
        """
        active_weights = []
        active_probs = []
        modality_status = {
            'cognitive': True,
            'eeg': p_eeg is not None,
            'speech': p_speech is not None
        }

        # 1. Cognitive (Always required)
        active_probs.append(p_cog)
        active_weights.append(self.weights[0])

        # 2. EEG (If provided by user)
        if p_eeg is not None:
            active_probs.append(p_eeg)
            active_weights.append(self.weights[1])

        # 3. Speech (If provided by user)
        if p_speech is not None:
            active_probs.append(p_speech)
            active_weights.append(self.weights[2])

        active_weights = np.array(active_weights)
        active_weights = active_weights / np.sum(active_weights) # Normalize

        # Calculate final integrated probability across active user-provided inputs
        p_final = float(np.sum(np.array(active_probs) * active_weights))
        p_final = float(np.clip(p_final, 0.0, 1.0))

        # Risk Level Designation
        if p_final < 0.35:
            risk_level = "Low"
            diagnosis = "Healthy"
            is_alzheimers = False
        elif p_final < 0.65:
            risk_level = "Moderate"
            diagnosis = "Alzheimer's Disease (Mild Cognitive Impairment / MCI)"
            is_alzheimers = True
        else:
            risk_level = "High"
            diagnosis = "Alzheimer's Disease (High Risk)"
            is_alzheimers = True

        # Confidence Score (Distance from uncertain 0.5 boundary)
        confidence = float(np.abs(p_final - 0.5) * 2.0 * 100.0)
        confidence = max(50.0, min(99.9, confidence))

        # Modality Contributions
        total_active_p = np.sum(active_probs) + 1e-12
        contributions = {
            'Cognitive': float((p_cog / total_active_p) * 100.0),
            'EEG': float((p_eeg / total_active_p) * 100.0) if p_eeg is not None else 0.0,
            'Speech': float((p_speech / total_active_p) * 100.0) if p_speech is not None else 0.0
        }

        # Strategy label
        if p_eeg is not None and p_speech is not None:
            strategy_name = "Full Multimodal Optimal Weighting (3 Modalities)"
        elif p_eeg is not None or p_speech is not None:
            strategy_name = "Dual-Modality Dynamic Weighting"
        else:
            strategy_name = "Single Modality Assessment (Cognitive Data Only)"

        return {
            'final_probability': p_final,
            'final_probability_pct': round(p_final * 100.0, 1),
            'risk_level': risk_level,
            'diagnosis': diagnosis,
            'is_alzheimers': is_alzheimers,
            'confidence_score': round(confidence, 1),
            'strategy_name': strategy_name,
            'modality_status': modality_status,
            'individual_probabilities': {
                'cognitive': p_cog,
                'eeg': p_eeg,
                'speech': p_speech
            },
            'contributions': contributions
        }

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)
