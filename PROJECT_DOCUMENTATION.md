# NeuroCare: Complete Multimodal Alzheimer's Disease Prediction System Documentation

---

## Executive Summary & System Overview

**NeuroCare** is an advanced, non-invasive **Multimodal Alzheimer's Disease Prediction & Diagnostic System** engineered for clinical decision support, academic research, and healthcare delivery.

The system synthesizes data across three independent biomedical modalities:
1. **Cognitive Data**: Neuropsychological test scores, demographic metrics, and volumetric MRI brain ratios.
2. **EEG Signals**: 16-channel continuous electroencephalogram frequency decomposition and neural complexity analysis.
3. **Speech Audio**: Acoustic voice biomarkers, pitch variations, micro-tremors (Jitter/Shimmer), and speech pause ratios.

Because biomedical datasets for these modalities originate from different cohort studies and lack shared patient IDs, **the system trains three independent machine learning model pipelines** and fuses their probabilities during inference using a **Multimodal Ensemble Engine (SLSQP Optimal Weighted Average)**.

---

## System Architecture & Data Flow

```
                               ┌─────────────────────────┐
                               │  Patient Form & Uploads │
                               └────────────┬────────────┘
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         │                                  │                                  │
         ▼                                  ▼                                  ▼
┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
│ Modality 1: Cog │                │ Modality 2: EEG │                │Modality 3:Voice │
│  Age, MMSE, CDR │                │  16 Channels    │                │  MFCCs, Pitch,  │
│  EDUC, nWBV...  │                │ 0.5-45 Hz Filter│                │  Jitter, Pauses │
└────────┬────────┘                └────────┬────────┘                └────────┬────────┘
         │                                  │                                  │
         ▼                                  ▼                                  ▼
┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
│Cognitive Model  │                │    EEG Model    │                │  Speech Model   │
│Random Forest /  │                │    Logistic     │                │    Logistic     │
│Logistic (0.967) │                │Regression (1.0) │                │Regression (0.75)│
└────────┬────────┘                └────────┬────────┘                └────────┬────────┘
         │ (P_cog)                          │ (P_eeg)                          │ (P_speech)
         └──────────────────┬───────────────┴──────────────────┘
                            │
                            ▼
           ┌──────────────────────────────────┐
           │   Multimodal Ensemble Engine     │
           │  (SLSQP Optimal Weighted Fusion) │
           │  w1=0.34, w2=0.33, w3=0.33        │
           └────────────────┬─────────────────┘
                            │
                            ▼
           ┌──────────────────────────────────┐
           │   Clinical Diagnostic Report     │
           │ Risk Index %, Confidence, Safety │
           │ Precautions, Treatment Steps     │
           └────────────────┬─────────────────┘
                            │
                            ▼
           ┌──────────────────────────────────┐
           │ Persistent Database & Admin View │
           │  (SQLite neurocare.db, /admin)   │
           └──────────────────────────────────┘
```

---

## Detailed Component Specifications

### 1. 🧠 Modality 1: Cognitive Data Pipeline

- **Dataset**: OASIS Longitudinal MRI & Cognitive Dataset.
- **Input Features**:
  - `Age`: Patient age in years.
  - `Gender`: Male (`1`) / Female (`0`).
  - `EDUC`: Total years of formal education.
  - `SES`: Socioeconomic status (scale 1 - 5).
  - `MMSE`: Mini-Mental State Examination score (0 - 30).
  - `CDR`: Clinical Dementia Rating (0.0 = Normal, 0.5 = MCI, 1.0 = Mild, 2.0 = Severe).
  - `eTIV`: Estimated Total Intracranial Volume ($mm^3$).
  - `nWBV`: Normalized Whole Brain Volume ratio.
  - `ASF`: Atlas Scaling Factor.
- **Preprocessing Pipeline** ([`cognitive_preprocessor.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/preprocessing/cognitive_preprocessor.py)):
  - **Imputation**: Median imputation for missing metrics (`SimpleImputer`).
  - **Encoding**: Binary mapping for target labels (`Demented`/`Converted` $\rightarrow 1$, `Nondemented` $\rightarrow 0$).
  - **Scaling**: Standard Normal Scaling ($\mu = 0, \sigma = 1$).
- **Models Evaluated** ([`train_cognitive.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/train_cognitive.py)):
  - Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM.
- **Winning Selected Model**: **Regularized Logistic Regression**
  - **Validation ROC-AUC**: `0.9677` (96.8%)
  - **Validation Accuracy**: `94.6%`

---

### 2. ⚡ Modality 2: 16-Channel EEG Signal Pipeline

- **Dataset**: Multi-Channel Clinical EEG Alzheimer's Dataset.
- **Electrodes (16 Channels)**: `Fp1`, `Fp2`, `F7`, `F3`, `Fz`, `F4`, `F8`, `T3`, `C3`, `Cz`, `C4`, `T4`, `T5`, `P3`, `Pz`, `P4`.
- **Signal Processing Pipeline** ([`eeg_preprocessor.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/preprocessing/eeg_preprocessor.py), [`eeg_features.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/feature_extraction/eeg_features.py)):
  - **Bandpass Filtering**: 4th-Order Butterworth Filter ($0.5\text{ Hz} - 45\text{ Hz}$).
  - **Epoching**: Sliced continuous signal into 2-second time windows.
  - **Welch Power Spectral Density (PSD)**: Extracted across 5 frequency bands:
    - Delta ($\delta: 0.5 - 4\text{ Hz}$)
    - Theta ($\theta: 4 - 8\text{ Hz}$)
    - Alpha ($\alpha: 8 - 13\text{ Hz}$)
    - Beta ($\beta: 13 - 30\text{ Hz}$)
    - Gamma ($\gamma: 30 - 45\text{ Hz}$)
  - **Hjorth Parameters**: Activity (Variance), Mobility (Frequency spread), Complexity (Rhythm distortion).
  - **Spectral Ratios**: $\theta/\alpha$ ratio and $\theta/\beta$ slowing index.
- **Models Evaluated** ([`train_eeg.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/train_eeg.py)):
  - Machine Learning Suite (LR, RF, XGB, CatBoost, LightGBM) + Deep Learning (1D-CNN, LSTM, Hybrid CNN-LSTM).
- **Winning Selected Model**: **Regularized Logistic Regression**
  - **Validation ROC-AUC**: `1.0000` (100.0%)

---

### 3. 🎙️ Modality 3: Speech & Voice Acoustic Pipeline

- **Dataset**: Alzheimer's Clinical Speech & Audio Dataset (e.g. Pitt Corpus / Cookie Theft Bank).
- **Acoustic Processing Pipeline** ([`speech_preprocessor.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/preprocessing/speech_preprocessor.py), [`speech_features.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/feature_extraction/speech_features.py)):
  - **Resampling**: Standardized to 16,000 Hz (16 kHz) mono audio.
  - **Silence Trimming**: Applied `librosa.effects.trim(top_db=25)` to remove background pauses.
  - **Window Slicing**: Sliced representative 2.5-second audio window for sub-second real-time inference.
- **Extracted Features (68 Acoustic Biomarkers)**:
  - **MFCCs (52 Features)**: 13 static coefficients + 13 First Deltas ($\Delta$) + 13 Second Deltas ($\Delta^2$) + standard deviations.
  - **Pitch ($F_0$)**: Extracted using **YIN algorithm** (`librosa.yin`) to measure vocal monotone.
  - **Micro-Tremors**: Jitter (frequency instability) and Shimmer (loudness instability).
  - **Speech Pace**: Zero-Crossing Rate (ZCR), RMS Energy, Spectral Centroid, and Pause Ratio.
- **Models Evaluated** ([`train_speech.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/train_speech.py)):
  - ML Suite + Deep Learning Architectures (1D-CNN, LSTM, CNN-LSTM).
- **Winning Selected Model**: **Regularized Logistic Regression**
  - **Validation ROC-AUC**: `0.7500` (75.0%)

---

### 4. 🔀 Multimodal Ensemble Engine

- **Architecture** ([`ensemble.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/ensemble.py)):
  - Combines individual class probabilities ($P_{\text{cog}}, P_{\text{eeg}}, P_{\text{speech}}$) into a unified diagnostic score.
- **Optimization Strategy**: Sequential Least Squares Programming (SLSQP).
  - Subject to constraints: $\sum w_i = 1.0$ and $w_i \ge 0$.
- **Learned Contribution Weights**:
  - $W_{\text{cognitive}} = 0.34$ (34%)
  - $W_{\text{eeg}} = 0.33$ (33%)
  - $W_{\text{speech}} = 0.33$ (33%)
- **Diagnostic Decision Thresholds**:
  - Probability $< 35\% \rightarrow$ **Healthy (Low Risk)**
  - Probability $35\% - 65\% \rightarrow$ **Mild / Moderate Risk**
  - Probability $> 65\% \rightarrow$ **Alzheimer's Disease (High Risk)**

---

### 5. 💻 Web Application & System Infrastructure

- **Backend Framework**: Flask (`app.py`).
- **Database Storage**: SQLite (`neurocare.db`, [`database.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/database.py)) for persistent user accounts and patient reports.
- **User Authentication**: Password hashing (`werkzeug.security`), session management, and role-based access (`user` vs `admin`).
- **Admin Dashboard (`/admin`)**: System-wide analytics showing total registered users, generated reports, high risk counts, and complete patient history.
- **Printable Medical Report (`result.html`)**:
  - Official medical header with Patient Name, Assessment ID, Date, Evaluator.
  - Category Risk Breakdown Meters.
  - Immediate Patient Safety Precautions (Medication supervision, Fall prevention, Wandering alerts).
  - Clinical Treatment Protocol & Next Steps (Neurologist consult, MRI/PET scan, Cholinesterase Inhibitors, MIND diet, Sleep hygiene).
  - 1-Click Print to PDF using `@media print` CSS styling.

---

### 🌐 6. Production & Cloud Deployment Setup

- **Production WSGI Server**: [`prod_server.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/prod_server.py) powered by **Waitress** (6 worker threads).
- **Cloud Platform**: Render.com Web Service ([`render.yaml`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/render.yaml), [`Procfile`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/Procfile), [`.python-version`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/.python-version)).
- **Runtime Environment**: Python 3.10.12.
- **Live Deployment URL**: **`https://alzheimer-prediction-system.onrender.com`**

---

## Summary of File Functions

| File Name | Description & Responsibility |
| :--- | :--- |
| [`app.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/app.py) | Main Flask server handling routes, authentication, form submissions, and report rendering. |
| [`database.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/database.py) | SQLite database interface managing users, sessions, passwords, and saved patient records. |
| [`predict.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/predict.py) | Master Inference Engine coordinating preprocessors, trained models, and fallbacks. |
| [`ensemble.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/ensemble.py) | Multimodal Ensemble decision engine calculating SLSQP weighted averages. |
| [`prod_server.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/prod_server.py) | Multi-threaded Waitress WSGI production launcher for live web hosting. |
| [`train_cognitive.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/train_cognitive.py) | Training script for Cognitive Data ML models. |
| [`train_eeg.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/train_eeg.py) | Training script for 16-Channel EEG Signal models. |
| [`train_speech.py`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/train_speech.py) | Training script for Speech Audio acoustic models. |
| [`templates/index.html`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/templates/index.html) | Patient assessment portal UI with input fields, EEG upload, and voice recorder. |
| [`templates/result.html`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/templates/result.html) | Printable medical diagnosis report UI with precautions and treatment protocol. |
| [`templates/admin_dashboard.html`](file:///c:/Users/SUHAN%20S%20P/OneDrive/Desktop/31%20july/templates/admin_dashboard.html) | System Administrator portal for patient management and account tracking. |
