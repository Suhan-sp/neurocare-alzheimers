"""
Speech Feature Extraction Module
Extracts MFCCs, Pitch, Energy, Zero Crossing Rate, Spectral Centroid, Chroma,
Mel Spectrogram statistics, Jitter, Shimmer, and Speaking Rate from audio signals using Librosa.
"""

import numpy as np
import librosa
from utils.logger import logger

def compute_jitter_shimmer(y: np.ndarray, sr: int):
    """
    Computes acoustic perturbation metrics: Jitter (pitch variability) and Shimmer (amplitude variability).
    
    Args:
        y (np.ndarray): Audio signal.
        sr (int): Sampling rate.
        
    Returns:
        tuple: (jitter, shimmer)
    """
    try:
        # Extract fundamental frequency F0
        y_short = y[:sr * 2] if len(y) > sr * 2 else y
        f0 = librosa.yin(y_short, fmin=80, fmax=400, sr=sr)
        f0_clean = f0[f0 > 0]
        
        if len(f0_clean) < 4:
            return 0.0, 0.0
            
        periods = 1.0 / f0_clean
        period_diffs = np.abs(np.diff(periods))
        jitter = float(np.mean(period_diffs) / (np.mean(periods) + 1e-12))
        
        # Amplitude of harmonic segments
        rms_frames = librosa.feature.rms(y=y_short)[0]
        rms_diffs = np.abs(np.diff(rms_frames))
        shimmer = float(np.mean(rms_diffs) / (np.mean(rms_frames) + 1e-12))
        
        return jitter, shimmer
    except Exception as e:
        logger.warning(f"Jitter/Shimmer calculation fallback: {e}")
        return 0.0, 0.0

def extract_speech_features(y: np.ndarray, sr: int = 16000) -> tuple:
    """
    Extracts comprehensive audio features from loaded audio time-series.
    
    Args:
        y (np.ndarray): Audio signal array.
        sr (int): Sampling rate (default 16,000 Hz).
        
    Returns:
        dict: Feature dictionary.
        np.ndarray: Mel Spectrogram matrix (for deep learning models and visualization).
    """
    features = {}
    
    # Ensure minimum signal length
    if len(y) < sr * 0.5:
        y = np.pad(y, (0, int(sr * 0.5) - len(y)))

    # 1. MFCC Features (13 coefficients + Delta + Delta-Delta)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        features[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))
        features[f'mfcc_delta_{i+1}_mean'] = float(np.mean(mfcc_delta[i]))
        features[f'mfcc_delta2_{i+1}_mean'] = float(np.mean(mfcc_delta2[i]))

    # 2. Pitch / Fundamental Frequency (F0)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    features['pitch_mean'] = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
    features['pitch_std'] = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

    # 3. RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    features['energy_mean'] = float(np.mean(rms))
    features['energy_std'] = float(np.std(rms))

    # 4. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))

    # 5. Spectral Centroid & Bandwidth
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(spec_cent))
    features['spectral_centroid_std'] = float(np.std(spec_cent))
    features['spectral_bandwidth_mean'] = float(np.mean(spec_bw))

    # 6. Chroma Features (12 pitch classes)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = float(np.mean(chroma))
    features['chroma_std'] = float(np.std(chroma))

    # 7. Mel Spectrogram Summary
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel_spec_mean'] = float(np.mean(mel_spec_db))
    features['mel_spec_std'] = float(np.std(mel_spec_db))

    # 8. Jitter & Shimmer
    jitter, shimmer = compute_jitter_shimmer(y, sr)
    features['jitter'] = jitter
    features['shimmer'] = shimmer

    # 9. Speaking Rate & Silence / Pause Ratio
    non_silent_intervals = librosa.effects.split(y, top_db=25)
    non_silent_duration = sum([(end - start) for start, end in non_silent_intervals]) / sr
    total_duration = len(y) / sr
    features['speaking_rate'] = float(non_silent_duration / (total_duration + 1e-12))
    features['pause_ratio'] = float(1.0 - features['speaking_rate'])

    return features, mel_spec_db
