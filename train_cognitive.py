"""
Cognitive Prediction Model Training Pipeline (train_cognitive.py)
Trains multiple machine learning models on Cognitive/Clinical patient metrics,
automatically selects the best model, evaluates performance, generates SHAP explainability,
and saves the trained model artifact to saved_models/best_cognitive_model.pkl.
"""

import os
import pandas as pd
import numpy as np
from preprocessing.cognitive_preprocessor import CognitivePreprocessor
from models.ml_models import MLModelSuite
from utils.visualization import PublicationVisualizer
from utils.explainability import ModelExplainer
from utils.logger import logger

def generate_synthetic_cognitive_data(n_samples=400):
    """Generates synthetic OASIS-like dataset if actual dataset is unavailable."""
    np.random.seed(42)
    age = np.random.randint(60, 96, size=n_samples)
    gender = np.random.choice(['M', 'F'], size=n_samples)
    educ = np.random.randint(8, 21, size=n_samples)
    ses = np.random.choice([1, 2, 3, 4, 5, np.nan], size=n_samples, p=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05])
    mmse = np.random.randint(15, 31, size=n_samples).astype(float)
    mmse[np.random.choice(n_samples, 20)] = np.nan
    cdr = np.random.choice([0.0, 0.5, 1.0, 2.0], size=n_samples, p=[0.5, 0.3, 0.15, 0.05])
    etiv = np.random.randint(1100, 2000, size=n_samples)
    nwbv = np.random.uniform(0.64, 0.85, size=n_samples)
    asf = np.random.uniform(0.85, 1.6, size=n_samples)

    # Risk heuristic for synthetic label
    risk = (90 - mmse.fillna(27)) * 0.1 + cdr * 2.0 + (age - 60) * 0.02 - nwbv * 3.0
    group = np.where(risk > 1.2, 'Demented', 'Nondemented')

    df = pd.DataFrame({
        'Group': group,
        'M/F': gender,
        'Age': age,
        'EDUC': educ,
        'SES': ses,
        'MMSE': mmse,
        'CDR': cdr,
        'eTIV': etiv,
        'nWBV': nwbv,
        'ASF': asf
    })
    return df

def run_cognitive_training(dataset_path="cognitive dataset/alzheimer.csv", output_dir="saved_models"):
    """Runs the complete cognitive training pipeline."""
    logger.info("==================================================")
    logger.info("  STARTING COGNITIVE MODEL TRAINING PIPELINE    ")
    logger.info("==================================================")

    # 1. Load Data
    if os.path.exists(dataset_path):
        logger.info(f"Loading Cognitive Dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        logger.warning(f"Dataset not found at {dataset_path}. Generating synthetic dataset...")
        df = generate_synthetic_cognitive_data()

    # 2. Preprocess Data
    preprocessor = CognitivePreprocessor()
    X_processed, y = preprocessor.fit_transform(df, target_col='Group')
    preprocessor.save(os.path.join(output_dir, "cognitive_preprocessor.pkl"))

    # 3. Train ML Model Suite
    suite = MLModelSuite()
    results, (best_name, best_model) = suite.train_and_evaluate_all(X_processed, y, cv_splits=5)

    # 4. Save Best Model
    suite.save_best_model(os.path.join(output_dir, "best_cognitive_model.pkl"))

    # 5. Visualizations & Explainability
    oof_probs = results[best_name]['oof_probs']
    oof_preds = results[best_name]['oof_preds']

    PublicationVisualizer.plot_confusion_matrix(
        y, oof_preds,
        title=f"Cognitive Confusion Matrix ({best_name})",
        save_path="static/plots/cognitive_cm.png"
    )
    PublicationVisualizer.plot_roc_curve(
        y, oof_probs,
        title=f"Cognitive ROC Curve ({best_name})",
        save_path="static/plots/cognitive_roc.png"
    )

    # SHAP Explainability
    ModelExplainer.generate_shap_plots(
        best_model, X_processed,
        feature_names=preprocessor.selected_feature_names,
        output_dir="static/plots"
    )

    logger.info("==================================================")
    logger.info(f"  COGNITIVE TRAINING COMPLETE: Best = {best_name}")
    logger.info("==================================================")
    return best_name, results[best_name]

if __name__ == "__main__":
    run_cognitive_training()
