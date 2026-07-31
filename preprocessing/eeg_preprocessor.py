"""
EEG Preprocessing Module
Implements MNE-based and SciPy signal filtering (Bandpass 0.5 - 45 Hz),
window epoching, and automated dataset feature matrix construction for EEG signals.
"""

import pandas as pd
import numpy as np
import scipy.signal as signal
import joblib
import os
from sklearn.preprocessing import StandardScaler
from feature_extraction.eeg_features import extract_multichannel_eeg_features
from utils.logger import logger

class EEGPreprocessor:
    """
    Preprocessing pipeline for EEG recordings (CSV / EDF format).
    Performs bandpass filtering, window epoching, and feature extraction.
    """
    DEFAULT_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4']
    
    def __init__(self, fs: float = 250.0, l_freq: float = 0.5, h_freq: float = 45.0):
        self.fs = fs
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.scaler = StandardScaler()
        self.feature_names = []

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Applies a Butterworth bandpass filter (0.5 Hz - 45 Hz).
        
        Args:
            data (np.ndarray): 2D signal array (samples, channels)
            
        Returns:
            np.ndarray: Filtered signal array.
        """
        nyquist = 0.5 * self.fs
        low = self.l_freq / nyquist
        high = min(self.h_freq / nyquist, 0.99)
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        filtered = np.zeros_like(data)
        if len(data.shape) == 1:
            filtered = signal.filtfilt(b, a, data)
        else:
            for ch in range(data.shape[1]):
                filtered[:, ch] = signal.filtfilt(b, a, data[:, ch])
                
        return filtered

    def process_raw_file(self, file_path_or_df, window_seconds: float = 4.0):
        """
        Processes a single raw CSV or EDF file into feature vectors and channel PSDs.
        
        Args:
            file_path_or_df: Path to CSV/EDF file OR pre-loaded DataFrame.
            window_seconds (float): Length of epoch window.
            
        Returns:
            np.ndarray: Feature matrix for the recording (num_epochs, num_features)
            dict: Average PSD dictionary per channel
        """
        if isinstance(file_path_or_df, str):
            if file_path_or_df.endswith('.edf'):
                import mne
                raw = mne.io.read_raw_edf(file_path_or_df, preload=True, verbose=False)
                raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, verbose=False)
                df = raw.to_data_frame()
                # Drop time column if present
                if 'time' in df.columns:
                    df = df.drop(columns=['time'])
            else:
                df = pd.read_csv(file_path_or_df)
        else:
            df = file_path_or_df.copy()

        # Extract channels and optional status target
        target_col = [c for c in df.columns if c.lower() in ['status', 'label', 'target', 'group']]
        y_val = None
        if target_col:
            y_val = df[target_col[0]].values
            df = df.drop(columns=target_col)

        channels = [c for c in df.columns if c in self.DEFAULT_CHANNELS]
        if not channels:
            channels = list(df.columns[:16]) # Fallback to first 16 columns

        signal_data = df[channels].values.astype(np.float64)

        # Apply Bandpass Filter
        signal_filtered = self.bandpass_filter(signal_data)

        # Epoching / Windowing
        samples_per_window = int(window_seconds * self.fs)
        total_samples = len(signal_filtered)
        
        if total_samples < samples_per_window:
            samples_per_window = total_samples

        num_epochs = max(1, total_samples // samples_per_window)
        feature_rows = []
        psd_accum = {}

        for ep in range(num_epochs):
            start = ep * samples_per_window
            end = start + samples_per_window
            if end > total_samples:
                break
            
            segment = signal_filtered[start:end, :]
            feat_dict, psd_dict = extract_multichannel_eeg_features(segment, channels, fs=self.fs)
            
            if not self.feature_names:
                self.feature_names = list(feat_dict.keys())
                
            feature_rows.append(list(feat_dict.values()))

            # Accumulate PSDs for visualization
            for ch, (freqs, psd) in psd_dict.items():
                if ch not in psd_accum:
                    psd_accum[ch] = (freqs, np.zeros_like(psd))
                psd_accum[ch] = (freqs, psd_accum[ch][1] + psd)

        # Average PSDs over epochs
        avg_psd_dict = {}
        for ch, (freqs, total_psd) in psd_accum.items():
            avg_psd_dict[ch] = (freqs, total_psd / num_epochs)

        X_mat = np.array(feature_rows)
        return X_mat, avg_psd_dict, y_val, channels

    def fit_transform(self, X_features: np.ndarray) -> np.ndarray:
        """Fits scale transformer and scales features."""
        return self.scaler.fit_transform(X_features)

    def transform(self, X_features: np.ndarray) -> np.ndarray:
        """Scales features using fitted scaler."""
        return self.scaler.transform(X_features)

    def save(self, filepath: str):
        """Saves EEG preprocessor state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved EEG preprocessor to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Loads EEG preprocessor state."""
        if os.path.exists(filepath):
            return joblib.load(filepath)
        else:
            raise FileNotFoundError(f"EEG Preprocessor not found at {filepath}")
