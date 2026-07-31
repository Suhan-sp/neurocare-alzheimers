"""
Speech Preprocessing Module
Implements audio loading, spectral noise reduction, silence trimming,
scaling, and batch processing for speech recordings.
"""

import numpy as np
import librosa
import joblib
import os
import io
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from feature_extraction.speech_features import extract_speech_features
from utils.logger import logger

class SpeechPreprocessor:
    """
    Preprocessing pipeline for Speech Audio (WAV files or microphone byte streams).
    """
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_and_clean_audio(self, file_path_or_bytes, top_db: float = 25.0) -> np.ndarray:
        """
        Loads audio from filepath or bytes, resamples, trims silence, and reduces noise.
        
        Args:
            file_path_or_bytes: Path string or bytes object.
            top_db (float): Silence threshold.
            
        Returns:
            np.ndarray: Denoised & trimmed audio signal array.
        """
        if isinstance(file_path_or_bytes, (bytes, bytearray, io.BytesIO)):
            if isinstance(file_path_or_bytes, (bytes, bytearray)):
                file_path_or_bytes = io.BytesIO(file_path_or_bytes)
            y, sr = sf.read(file_path_or_bytes)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1) # Convert stereo to mono
            if sr != self.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        else:
            y, sr = librosa.load(file_path_or_bytes, sr=self.target_sr, mono=True, duration=2.5)

        # 1. Silence Trimming
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        if len(y_trimmed) < self.target_sr * 0.3: # If trimmed too aggressively
            y_trimmed = y

        # 2. Simple Spectral Noise Reduction (Noise Floor Thresholding)
        stft = librosa.stft(y_trimmed)
        magnitude, phase = librosa.magphase(stft)
        noise_floor = np.mean(magnitude[:, :5], axis=1, keepdims=True)
        magnitude_clean = np.maximum(0, magnitude - 1.5 * noise_floor)
        stft_clean = magnitude_clean * phase
        y_clean = librosa.istft(stft_clean)

        return y_clean

    def process_single_audio(self, file_path_or_bytes):
        """
        Extracts features and mel spectrogram for a single audio input.
        
        Args:
            file_path_or_bytes: Path or audio bytes.
            
        Returns:
            dict: Audio feature dictionary.
            np.ndarray: Mel Spectrogram matrix (db).
        """
        y_clean = self.load_and_clean_audio(file_path_or_bytes)
        feat_dict, mel_spec_db = extract_speech_features(y_clean, sr=self.target_sr)
        
        if not self.feature_names:
            self.feature_names = list(feat_dict.keys())
            
        return feat_dict, mel_spec_db

    def fit_transform(self, X_features: np.ndarray) -> np.ndarray:
        """Fits scale transformer and scales features."""
        return self.scaler.fit_transform(X_features)

    def transform(self, X_features: np.ndarray) -> np.ndarray:
        """Scales features using fitted scaler."""
        return self.scaler.transform(X_features)

    def save(self, filepath: str):
        """Saves Speech preprocessor state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved Speech preprocessor to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Loads Speech preprocessor state."""
        if os.path.exists(filepath):
            return joblib.load(filepath)
        else:
            raise FileNotFoundError(f"Speech Preprocessor not found at {filepath}")
