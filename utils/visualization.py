"""
Visualization Utility Module
Generates publication-quality figures: Confusion Matrices, ROC Curves, Precision-Recall Curves,
Learning Curves, EEG Power Spectral Density plots, and Speech Mel Spectrograms.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import librosa
import librosa.display
import os
from utils.logger import logger

# Set publication-style aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.family'] = 'sans-serif'

class PublicationVisualizer:
    """
    Generates academic/publication quality plots saved to static/plots directory.
    """
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path="static/plots/confusion_matrix.png"):
        """Plot and save publication-quality confusion matrix."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Healthy', "Alzheimer's"],
                    yticklabels=['Healthy', "Alzheimer's"], ax=ax, annot_kws={"size": 14})
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        return save_path

    @staticmethod
    def plot_roc_curve(y_true, y_probs, title="ROC Curve", save_path="static/plots/roc_curve.png"):
        """Plot and save ROC Curve."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(5.5, 4.5))
        plt.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(title, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        return save_path

    @staticmethod
    def plot_pr_curve(y_true, y_probs, title="Precision-Recall Curve", save_path="static/plots/pr_curve.png"):
        """Plot and save Precision-Recall Curve."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(5.5, 4.5))
        plt.plot(recall, precision, color='#059669', lw=2.5, label=f'PR AUC = {pr_auc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        return save_path

    @staticmethod
    def plot_eeg_power_spectrum(psd_dict: dict, save_path="static/plots/eeg_power_spectrum.png"):
        """Plots EEG Power Spectral Density across frequency bands (Delta, Theta, Alpha, Beta, Gamma)."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(8, 4.5))

        # Plot top 4 key channels or average
        for ch_name, (freqs, psd) in list(psd_dict.items())[:6]:
            plt.plot(freqs, 10 * np.log10(psd + 1e-12), label=ch_name, alpha=0.8, lw=1.5)

        # Highlight EEG frequency bands
        plt.axvspan(0.5, 4, color='red', alpha=0.1, label='Delta (0.5-4Hz)')
        plt.axvspan(4, 8, color='orange', alpha=0.1, label='Theta (4-8Hz)')
        plt.axvspan(8, 12, color='green', alpha=0.1, label='Alpha (8-12Hz)')
        plt.axvspan(12, 30, color='blue', alpha=0.1, label='Beta (12-30Hz)')
        plt.axvspan(30, 45, color='purple', alpha=0.1, label='Gamma (30-45Hz)')

        plt.xlim([0.5, 45])
        plt.xlabel('Frequency (Hz)', fontsize=10)
        plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=10)
        plt.title('EEG Power Spectral Density (PSD)', fontweight='bold')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        return save_path

    @staticmethod
    def plot_mel_spectrogram(mel_spec_db: np.ndarray, sr: int = 16000, save_path="static/plots/mel_spectrogram.png"):
        """Plots Mel Spectrogram of speech recording."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(7, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Audio Mel Spectrogram', fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        return save_path
