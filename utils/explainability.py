"""
Explainability Module
Computes SHAP (SHapley Additive exPlanations) values for tree/linear cognitive models
and extracts feature importances for clinical interpretability.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for web servers
import matplotlib.pyplot as plt
import shap
import os
from utils.logger import logger

class ModelExplainer:
    """
    Handles SHAP explainability and feature importance extraction.
    """
    @staticmethod
    def generate_shap_plots(model, X_sample: np.ndarray, feature_names: list, output_dir: str = "static/plots"):
        """
        Generates SHAP summary and waterfall plots for the cognitive model.
        
        Args:
            model: Trained classifier.
            X_sample (np.ndarray): Background feature matrix.
            feature_names (list): List of feature names.
            output_dir (str): Destination directory for plots.
            
        Returns:
            dict: Paths to saved plot images and calculated SHAP values.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        
        try:
            # Create SHAP Explainer (TreeExplainer for tree models, LinearExplainer/Explainer for others)
            model_type = type(model).__name__
            if any(t in model_type for t in ['Forest', 'XGB', 'Cat', 'LGBM', 'Tree']):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample).values

            # Handle multi-class / list output of shap_values
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # 1. SHAP Summary Plot
            plt.figure(figsize=(8, 5))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title("Cognitive SHAP Feature Summary", fontsize=12, pad=15)
            plt.tight_layout()
            summary_path = os.path.join(output_dir, "shap_summary.png")
            plt.savefig(summary_path, dpi=200, bbox_inches='tight')
            plt.close()
            paths['shap_summary'] = summary_path

            # 2. SHAP Bar Plot (Mean |SHAP| Value)
            plt.figure(figsize=(8, 4.5))
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            if len(mean_abs_shap.shape) > 1:
                mean_abs_shap = np.mean(mean_abs_shap, axis=0)
                
            sorted_idx = np.argsort(mean_abs_shap)
            plt.barh(np.array(feature_names)[sorted_idx], mean_abs_shap[sorted_idx], color='#3b82f6')
            plt.xlabel("Mean |SHAP Value| (Impact on Alzheimer's Diagnosis)")
            plt.title("SHAP Feature Importance Ranking")
            plt.tight_layout()
            bar_path = os.path.join(output_dir, "shap_waterfall.png")
            plt.savefig(bar_path, dpi=200, bbox_inches='tight')
            plt.close()
            paths['shap_waterfall'] = bar_path
            
            logger.info("Generated SHAP explainability plots successfully.")
            return paths, shap_values
        except Exception as e:
            logger.error(f"Error generating SHAP plots: {e}")
            return {}, None

    @staticmethod
    def get_feature_importances(model, feature_names: list) -> dict:
        """Extracts normalized feature importances from a model."""
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            imp = np.abs(model.coef_[0])
        else:
            return {f: 1.0 / len(feature_names) for f in feature_names}

        imp_norm = imp / (np.sum(imp) + 1e-12)
        return {f: float(val) for f, val in zip(feature_names, imp_norm)}
