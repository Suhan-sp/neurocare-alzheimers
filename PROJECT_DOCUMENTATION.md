# NeuroCare: Complete Multimodal Alzheimer's Disease Prediction System Documentation

---

## Executive Summary & System Overview

**NeuroCare** is an advanced, non-invasive **Multimodal Alzheimer's Disease Prediction & Diagnostic System** engineered for clinical decision support, academic research, and healthcare delivery.

The system synthesizes data across three independent biomedical modalities:
1. **Cognitive Data**: Neuropsychological test scores, demographic metrics, and volumetric MRI brain ratios.
2. **EEG Signals**: 16/19-channel continuous electroencephalogram frequency decomposition and neural complexity analysis across 88 human subjects.
3. **Speech Audio**: Acoustic voice biomarkers, pitch variations, micro-tremors (Jitter/Shimmer), and speech pause ratios across 590+ clinical recordings.

---

## Detailed Component Specifications

### 1. 🧠 Modality 1: Cognitive Data Pipeline

- **Dataset**: OASIS Longitudinal MRI & Cognitive Dataset ($N = 373$ clinical patient records).
- **Input Features**: `Age`, `Gender`, `EDUC`, `SES`, `MMSE`, `CDR`, `eTIV`, `nWBV`, `ASF`.
- **Selected Model**: **Regularized Logistic Regression**
  - **Validation ROC-AUC**: `0.9677` (96.8%)
  - **Validation Accuracy**: `94.64%`

---

### 2. ⚡ Modality 2: 19-Channel Clinical EEG Signal Pipeline

- **Dataset**: OpenNeuro ds004504 19-Channel Rest EEG Dataset ($N = 88$ Human Subjects).
  - **Alzheimer's Disease (AD)**: `sub-001` to `sub-036` ($N = 36$ subjects) $\rightarrow$ Class 1
  - **Frontotemporal Dementia (FTD)**: `sub-037` to `sub-059` ($N = 23$ subjects) $\rightarrow$ Class 1
  - **Healthy Control (CN)**: `sub-060` to `sub-088` ($N = 29$ subjects) $\rightarrow$ Class 0
- **Total Clinical Epochs**: **1,320 Epochs** ($435$ Healthy vs $885$ Alzheimer's/Dementia).
- **Signal Processing & Feature Extraction**:
  - **Bandpass Filtering**: 4th-Order Butterworth Filter ($0.5\text{ Hz} - 45\text{ Hz}$).
  - **Epoching**: 2-second time-window epoching across 19 electrode channels.
  - **Welch Power Spectral Density (PSD)**: Extracted across 5 frequency bands ($\delta, \theta, \alpha, \beta, \gamma$).
  - **Hjorth Parameters**: Activity, Mobility, Complexity.
  - **Spectral Slowing Ratios**: $\theta / \alpha$ ratio.
- **Selected Winner Model**: **XGBoost Classifier**
  - **Validation ROC-AUC**: **`0.8203` (82.0%)**
  - **Validation Accuracy**: **`77.50%`**

---

### 3. 🎙️ Modality 3: Speech Audio Acoustic Pipeline

- **Dataset**: Clinical Speech Audio Dataset ($N = 591$ WAV files, resampled to 788 balanced instances).
  - **Cookie Theft Dementia (CTD)**: $197$ recordings $\rightarrow$ Class 1
  - **Fluency Controls (SFT/PFT)**: $394$ recordings $\rightarrow$ Class 0
- **Acoustic Biomarkers (80+ Features)**:
  - 20 MFCCs + Deltas + Delta-Deltas, Pitch ($F_0$), Jitter, Shimmer, Spectral Contrast (7 sub-bands), Spectral Rolloff, Spectral Flatness, Tonnetz (6 pitch classes), Pause/Speaking Ratios.
- **Selected Winner Model**: **Random Forest Classifier**
  - **Validation ROC-AUC**: **`0.9775` (97.8%)**
  - **Validation Accuracy**: **`92.13%`**

---

### 4. 🔀 Multimodal Ensemble Layer

- **Optimization Strategy**: Sequential Least Squares Programming (SLSQP).
- **Learned Weights**: $W_{\text{cognitive}} = 0.34$, $W_{\text{eeg}} = 0.33$, $W_{\text{speech}} = 0.33$.
- **Diagnostic Decision Thresholds**:
  - Probability $< 35\% \rightarrow$ **Healthy (Low Risk)**
  - Probability $35\% - 65\% \rightarrow$ **Mild / Moderate Risk**
  - Probability $> 65\% \rightarrow$ **Alzheimer's Disease (High Risk)**

---

## Summary of File Functions

| File Name | Description & Responsibility |
| :--- | :--- |
| [`app.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/app.py) | Main Flask server handling routes, authentication, form submissions, and report rendering. |
| [`database.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/database.py) | SQLite database interface managing users, sessions, passwords, and saved patient records. |
| [`predict.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/predict.py) | Master Inference Engine coordinating preprocessors, trained models, and fallbacks. |
| [`train_eeg.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/train_eeg.py) | Training script for 88-subject 19-Channel Clinical EEG signal models. |
| [`train_speech.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/train_speech.py) | Training script for 590+ clinical Speech Audio acoustic models. |
| [`EEG dataset/single_patient_EEG.csv`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/EEG%20dataset/single_patient_EEG.csv) | Healthy Control sample EEG recording (`status=0`) for live web portal testing. |
| [`EEG dataset/alzheimer_patient_EEG.csv`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/EEG%20dataset/alzheimer_patient_EEG.csv) | Alzheimer's Patient sample EEG recording (`status=1`) for live web portal testing. |
