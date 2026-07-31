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
        Loads audio from filepath or bytes, ressembles, trims silence, and reduces noise.
        """
        y = None
        sr = self.target_sr

        if isinstance(file_path_or_bytes, str) and os.path.exists(file_path_or_bytes):
            try:
                y, sr = librosa.load(file_path_or_bytes, sr=self.target_sr, mono=True, duration=2.5)
            except Exception:
                try:
                    y, sr = sf.read(file_path_or_bytes)
                    if len(y.shape) > 1:
                        y = np.mean(y, axis=1)
                    if sr != self.target_sr:
                        y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
                    sr = self.target_sr
                except Exception as e:
                    logger.error(f"Error loading audio file path: {e}")
                    y = np.zeros(self.target_sr * 2)

        elif isinstance(file_path_or_bytes, (bytes, bytearray, io.BytesIO)):
            if isinstance(file_path_or_bytes, (bytes, bytearray)):
                file_path_or_bytes = io.BytesIO(file_path_or_bytes)
            try:
                y, sr = sf.read(file_path_or_bytes)
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
                if sr != self.target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            except Exception as e:
                logger.error(f"Error reading audio bytes: {e}")
                y = np.zeros(self.target_sr * 2)
        else:
            y = np.zeros(self.target_sr * 2)

        # Ensure non-empty array
        if y is None or len(y) == 0:
            y = np.zeros(self.target_sr * 2)

        # 1. Silence Trimming
        try:
            y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
            if len(y_trimmed) < self.target_sr * 0.3:
                y_trimmed = y
        except Exception:
            y_trimmed = y

        # 2. Simple Spectral Noise Reduction
        try:
            stft = librosa.stft(y_trimmed)
            magnitude, phase = librosa.magphase(stft)
            noise_floor = np.mean(magnitude[:, :min(5, magnitude.shape[1])], axis=1, keepdims=True)
            magnitude_clean = np.maximum(0, magnitude - 1.5 * noise_floor)
            stft_clean = magnitude_clean * phase
            y_clean = librosa.istft(stft_clean)
        except Exception:
            y_clean = y_trimmed

        return y_clean

    def process_single_audio(self, file_path_or_bytes):
        """
        Extracts features and mel spectrogram for a single audio input.
        """
        y_clean = self.load_and_clean_audio(file_path_or_bytes)
        feat_dict, mel_spec_db = extract_speech_features(y_clean, sr=self.target_sr)
        
        if not self.feature_names:
            self.feature_names = list(feat_dict.keys())
            
        return feat_dict, mel_spec_db

    def fit_transform(self, X_features: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X_features)

    def transform(self, X_features: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X_features)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)
