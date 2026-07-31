"""
Multimodal Ensemble Learning Module (ensemble.py)
Combines predictions from Cognitive, EEG, and Speech models using Soft Voting,
Optimal Weighted Averaging, and Stacking Meta-Classifiers.
Calculates Risk Level, Confidence Scores, and Individual Model Contributions.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import joblib
import os
from utils.logger import logger

class MultimodalEnsemble:
    """
    Fuses predictions from three independent modalities (Cognitive, EEG, Speech).
    """
    def __init__(self, weights=None):
        # Default initial weights: Equal weighting (1/3, 1/3, 1/3)
        self.weights = np.array(weights) if weights is not None else np.array([0.34, 0.33, 0.33])
        self.stacking_meta_learner = LogisticRegression()
        self.is_fitted = False

    def optimize_weights(self, val_probs_matrix: np.ndarray, y_val: np.ndarray):
        """
        Determines optimal ensemble weights by minimizing cross-entropy loss on validation predictions.
        
        Args:
            val_probs_matrix (np.ndarray): Array of shape (N, 3) containing [P_cog, P_eeg, P_speech]
            y_val (np.ndarray): Binary ground truth array
        """
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
            logger.warning("Weight optimization did not converge. Falling back to equal weighting.")
            self.weights = np.array([0.34, 0.33, 0.33])

        # Train Stacking Meta-Learner
        self.stacking_meta_learner.fit(val_probs_matrix, y_val)
        self.is_fitted = True

    def predict_ensemble(self, p_cog: float, p_eeg: float, p_speech: float, method: str = 'weighted') -> dict:
        """
        Combines individual probabilities into final diagnosis, risk level, and confidence score.
        
        Args:
            p_cog (float): Cognitive model probability
            p_eeg (float): EEG model probability
            p_speech (float): Speech model probability
            method (str): Fusion method ('weighted', 'soft_voting', 'stacking')
            
        Returns:
            dict: Comprehensive prediction output.
        """
        probs = np.array([p_cog, p_eeg, p_speech])

        if method == 'soft_voting':
            p_final = float(np.mean(probs))
        elif method == 'stacking' and self.is_fitted:
            p_final = float(self.stacking_meta_learner.predict_proba(probs.reshape(1, -1))[0, 1])
        else: # Default: Weighted Average
            p_final = float(np.sum(probs * self.weights))

        p_final = float(np.clip(p_final, 0.0, 1.0))

        # Risk Level Designation
        if p_final < 0.35:
            risk_level = "Low"
            diagnosis = "Healthy"
        elif p_final < 0.70:
            risk_level = "Moderate"
            diagnosis = "Alzheimer's Disease (Early Stage / MCI)"
        else:
            risk_level = "High"
            diagnosis = "Alzheimer's Disease (Severe)"

        # Confidence Score (Distance from uncertain 0.5 decision boundary)
        confidence = float(np.abs(p_final - 0.5) * 2.0 * 100.0)
        confidence = max(50.0, min(99.9, confidence))

        # Individual Contributions (%)
        total_p = np.sum(probs) + 1e-12
        contributions = {
            'Cognitive': float((p_cog / total_p) * 100.0),
            'EEG': float((p_eeg / total_p) * 100.0),
            'Speech': float((p_speech / total_p) * 100.0)
        }

        return {
            'diagnosis': diagnosis,
            'is_alzheimers': bool(p_final >= 0.5),
            'final_probability': round(p_final, 4),
            'final_probability_pct': round(p_final * 100.0, 1),
            'risk_level': risk_level,
            'confidence_score': round(confidence, 1),
            'individual_probabilities': {
                'cognitive': round(p_cog, 4),
                'eeg': round(p_eeg, 4),
                'speech': round(p_speech, 4)
            },
            'ensemble_weights': {
                'cognitive': round(float(self.weights[0]), 3),
                'eeg': round(float(self.weights[1]), 3),
                'speech': round(float(self.weights[2]), 3)
            },
            'contributions': contributions
        }

    def save(self, filepath: str):
        """Saves ensemble object."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str):
        """Loads ensemble object."""
        if os.path.exists(filepath):
            return joblib.load(filepath)
        return MultimodalEnsemble()
