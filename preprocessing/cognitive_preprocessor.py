"""
Cognitive Preprocessing Module
Handles data cleaning, missing value imputation, encoding, feature scaling,
and feature selection for clinical/cognitive data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib
import os
from utils.logger import logger

class CognitivePreprocessor:
    """
    Preprocessor pipeline for Cognitive / Clinical dataset (OASIS-style data).
    Target features: Age, Gender (M/F), Education (EDUC), SES, MMSE, CDR, eTIV, nWBV, ASF.
    """
    FEATURE_COLS = ['Age', 'Gender', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    
    def __init__(self, feature_selection_k=None):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.selector = None
        self.feature_selection_k = feature_selection_k
        self.selected_feature_names = self.FEATURE_COLS.copy()

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Group'):
        """
        Cleans, imputes, encodes, scales, and optionally selects features.
        
        Args:
            df (pd.DataFrame): Raw dataframe
            target_col (str): Target column name
            
        Returns:
            X_scaled (np.ndarray): Processed feature matrix
            y (np.ndarray): Target binary vector (1 = Alzheimer's/Demented, 0 = Healthy)
        """
        logger.info("Starting Cognitive preprocessing fit_transform...")
        df_clean = df.copy()

        # Handle Target if present
        y = None
        if target_col in df_clean.columns:
            # Map target categories: Demented and Converted to 1, Nondemented to 0
            y_raw = df_clean[target_col].astype(str).str.lower().str.strip()
            y = np.where((y_raw == 'demented') | (y_raw == 'converted') | (y_raw == 'ad'), 1, 0)
            df_clean = df_clean.drop(columns=[target_col])

        # Standardize gender column name if present as M/F
        if 'M/F' in df_clean.columns and 'Gender' not in df_clean.columns:
            df_clean = df_clean.rename(columns={'M/F': 'Gender'})

        # Encode Gender (M=1, F=0)
        if 'Gender' in df_clean.columns:
            if df_clean['Gender'].dtype == object:
                df_clean['Gender'] = df_clean['Gender'].map({'M': 1, 'F': 0, 'm': 1, 'f': 0}).fillna(0)

        # Select target features
        missing_cols = [col for col in self.FEATURE_COLS if col not in df_clean.columns]
        if missing_cols:
            logger.warning(f"Missing columns in input cognitive data: {missing_cols}. Filling with 0.")
            for col in missing_cols:
                df_clean[col] = 0.0

        X_df = df_clean[self.FEATURE_COLS]

        # Impute missing values (SES, MMSE often have missing values in OASIS)
        X_imputed = self.imputer.fit_transform(X_df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Feature selection if k specified
        if self.feature_selection_k and y is not None:
            self.selector = SelectKBest(score_func=mutual_info_classif, k=self.feature_selection_k)
            X_scaled = self.selector.fit_transform(X_scaled, y)
            mask = self.selector.get_support()
            self.selected_feature_names = [col for col, m in zip(self.FEATURE_COLS, mask) if m]
            logger.info(f"Selected Top {self.feature_selection_k} features: {self.selected_feature_names}")

        return X_scaled, y

    def transform_single(self, input_dict: dict) -> np.ndarray:
        """
        Transforms a single patient input dictionary for inference.
        
        Args:
            input_dict (dict): Dictionary with keys corresponding to FEATURE_COLS or equivalents.
            
        Returns:
            np.ndarray: Scaled array ready for model inference (shape: 1, num_features).
        """
        row = {}
        for col in self.FEATURE_COLS:
            val = input_dict.get(col, input_dict.get(col.lower(), 0.0))
            if col == 'Gender' and isinstance(val, str):
                val = 1.0 if val.upper().startswith('M') else 0.0
            try:
                row[col] = float(val) if val is not None and str(val) != '' else np.nan
            except ValueError:
                row[col] = 0.0

        df_single = pd.DataFrame([row])
        X_imp = self.imputer.transform(df_single[self.FEATURE_COLS])
        X_scale = self.scaler.transform(X_imp)

        if self.selector:
            X_scale = self.selector.transform(X_scale)

        return X_scale

    def save(self, filepath: str):
        """Saves preprocessor instance to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved Cognitive preprocessor to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Loads preprocessor instance from disk."""
        if os.path.exists(filepath):
            logger.info(f"Loading Cognitive preprocessor from {filepath}")
            return joblib.load(filepath)
        else:
            raise FileNotFoundError(f"Preprocessor file not found at {filepath}")
