"""
EEG Feature Extraction Module
Extracts Power Spectral Density (PSD), Band Powers (Delta, Theta, Alpha, Beta, Gamma),
Hjorth Parameters, Spectral Entropy, and Time-Domain Statistics from multi-channel EEG signals.
"""

import numpy as np
import scipy.signal as signal
from utils.logger import logger

def compute_hjorth_parameters(series: np.ndarray):
    """
    Computes Hjorth Parameters: Activity, Mobility, Complexity.
    
    Args:
        series (np.ndarray): 1D time-series signal.
        
    Returns:
        tuple: (activity, mobility, complexity)
    """
    # 1st and 2nd derivatives
    d1 = np.diff(series)
    d2 = np.diff(d1)
    
    var_zero = np.var(series)
    var_d1 = np.var(d1)
    var_d2 = np.var(d2)
    
    activity = var_zero
    
    # Avoid division by zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 1e-12 else 0.0
    
    if var_d1 > 1e-12:
        mob_d1 = np.sqrt(var_d2 / var_d1)
        complexity = mob_d1 / mobility if mobility > 1e-12 else 0.0
    else:
        complexity = 0.0
        
    return activity, mobility, complexity

def compute_spectral_entropy(psd: np.ndarray):
    """
    Computes Spectral Entropy from normalized PSD.
    
    Args:
        psd (np.ndarray): Power Spectral Density array.
        
    Returns:
        float: Spectral Entropy value.
    """
    psd_norm = psd / (np.sum(psd) + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

def extract_eeg_channel_features(data: np.ndarray, fs: float = 250.0) -> dict:
    """
    Extracts all frequency, time, and Hjorth features for a single channel.
    
    Args:
        data (np.ndarray): 1D channel signal.
        fs (float): Sampling frequency in Hz (default 250 Hz).
        
    Returns:
        dict: Feature dictionary for the channel.
    """
    features = {}
    
    # Basic Time Domain Statistics
    features['mean'] = float(np.mean(data))
    features['variance'] = float(np.var(data))
    features['std'] = float(np.std(data))
    
    # Hjorth Parameters
    act, mob, comp = compute_hjorth_parameters(data)
    features['hjorth_activity'] = float(act)
    features['hjorth_mobility'] = float(mob)
    features['hjorth_complexity'] = float(comp)
    
    # Power Spectral Density (PSD) using Welch's method
    nperseg = min(len(data), int(fs * 2))
    if nperseg < 16:
        nperseg = len(data)
    
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg)
    
    # Frequency Bands
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta': (12.0, 30.0),
        'gamma': (30.0, 45.0)
    }
    
    band_powers = {}
    try:
        from scipy.integrate import trapezoid as integrate_trapz
    except ImportError:
        integrate_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        
    total_power = float(integrate_trapz(psd, freqs)) + 1e-12
    
    for band_name, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        if np.any(idx):
            power = float(integrate_trapz(psd[idx], freqs[idx]))
        else:
            power = 0.0
        band_powers[band_name] = power
        features[f'band_{band_name}'] = power
        # Relative Band Power
        features[f'rel_band_{band_name}'] = power / total_power
        
    # Clinical Ratios (Key biomarkers for AD slowing)
    features['ratio_theta_alpha'] = band_powers['theta'] / (band_powers['alpha'] + 1e-12)
    features['ratio_theta_beta'] = band_powers['theta'] / (band_powers['beta'] + 1e-12)
    
    # Spectral Entropy
    features['spectral_entropy'] = float(compute_spectral_entropy(psd))
    
    return features, freqs, psd

def extract_multichannel_eeg_features(df_eeg: np.ndarray, channel_names: list, fs: float = 250.0):
    """
    Extracts concatenated feature vector for multi-channel EEG dataframe/matrix.
    
    Args:
        df_eeg (np.ndarray or pd.DataFrame): 2D array of shape (samples, channels)
        channel_names (list): List of channel names
        fs (float): Sampling frequency
        
    Returns:
        dict: Flattened feature dictionary.
        dict: Channel PSD dictionary for plotting.
    """
    feature_dict = {}
    psd_dict = {}
    
    if len(df_eeg.shape) == 1:
        df_eeg = df_eeg.reshape(-1, 1)
        
    num_channels = df_eeg.shape[1]
    
    for ch_idx in range(num_channels):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch_{ch_idx+1}"
        channel_signal = df_eeg[:, ch_idx]
        
        ch_feats, freqs, psd = extract_eeg_channel_features(channel_signal, fs=fs)
        psd_dict[ch_name] = (freqs, psd)
        
        for k, v in ch_feats.items():
            feature_dict[f"{ch_name}_{k}"] = v
            
    return feature_dict, psd_dict
