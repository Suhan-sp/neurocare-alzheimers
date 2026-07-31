# Multimodal Alzheimer's Disease Prediction System

A complete, production-quality **Multimodal Alzheimer's Disease Prediction & Explainability System** designed for academic research, final-year engineering projects, and research publication.

The system fuses three independent biomedical modalities:
1. **Cognitive & Clinical Data** (OASIS-style metrics: Age, Gender, Education, SES, MMSE, CDR, eTIV, nWBV, ASF)
2. **16-Channel EEG Signals** (Bandpass 0.5–45 Hz, Power Spectral Density for $\delta, \theta, \alpha, \beta, \gamma$ bands, Hjorth Parameters, Spectral Entropy)
3. **Speech Audio** (MFCCs, Pitch $F_0$, RMS Energy, ZCR, Spectral Centroid, Chroma, Mel Spectrogram, Jitter, Shimmer, Speaking Rate)

Because the datasets originate from independent sources without common patient IDs, three independent model suites are trained and optimized, and their predictions are dynamically combined during inference via a **Multimodal Ensemble Engine** (Soft Voting, Optimal Weighted Averaging, and Stacking Meta-Classifier).

---

## 🌟 Key Features

- **Automated Algorithm Selection**: Evaluates **Logistic Regression**, **Random Forest**, **XGBoost**, **CatBoost**, **LightGBM**, **1D-CNN**, **LSTM**, and **CNN-LSTM**, automatically selecting the best performer based on cross-validation ROC-AUC and F1 score.
- **Multimodal Fusion**: Combines independent modality probabilities into a unified diagnosis, Risk Level (Low, Moderate, High), Confidence Score %, and individual contribution breakdown.
- **Clinical Explainability**: Integrated **SHAP (SHapley Additive exPlanations)** summary and waterfall plots for cognitive feature attributions.
- **Biomedical Signal Processing**: MNE and SciPy bandpass filtering and Welch PSD estimation across standard EEG frequency bands ($\delta, \theta, \alpha, \beta, \gamma$).
- **Acoustic Audio Pipeline**: Librosa noise reduction, silence trimming, MFCC extraction, acoustic perturbation analysis (Jitter/Shimmer), and Mel Spectrogram generation.
- **Modern Web Application**: Flask web server with glassmorphic dark theme, live browser microphone recording, drag-and-drop file zones, and auto-fill clinical sample buttons for fast testing.

---

## 📁 System Architecture & Directory Structure

```
c:/Users/SUHAN S P/OneDrive/Desktop/31 july/
├── app.py                      # Flask Application server & API routes
├── train_cognitive.py          # Cognitive training & model selection script
├── train_eeg.py                # EEG feature extraction & model training script
├── train_speech.py             # Speech feature extraction & model training script
├── ensemble.py                 # Multimodal Ensemble Engine (Soft, Weighted, Stacking)
├── predict.py                  # Master Inference Engine
├── preprocessing/              # Cleaning, imputation, & signal filtering
│   ├── cognitive_preprocessor.py
│   ├── eeg_preprocessor.py
│   └── speech_preprocessor.py
├── feature_extraction/         # Biomedical signal & audio feature extractors
│   ├── eeg_features.py
│   └── speech_features.py
├── models/                     # Machine learning & deep learning wrappers
│   ├── ml_models.py
│   └── deep_models.py
├── saved_models/               # Persisted model weights and preprocessors
├── templates/                  # HTML5 Jinja templates
│   ├── base.html
│   ├── index.html
│   └── result.html
├── static/                     # CSS design system, JS handlers, & saved plots
│   ├── css/style.css
│   ├── js/main.js
│   └── plots/
├── utils/                      # Logging, explainability, & publication visualizers
│   ├── logger.py
│   ├── explainability.py
│   └── visualization.py
├── requirements.txt            # Package dependencies
└── README.md                   # System manual & research documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone & Environment Setup
```bash
git clone <repository_url>
cd "31 july"
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Model Training Pipelines

Run each modality's training pipeline independently:

### Train Cognitive Model
```bash
python train_cognitive.py
```
*Trains Logistic Regression, Random Forest, XGBoost, CatBoost, and LightGBM models on `cognitive dataset/alzheimer.csv`, performs 5-fold cross-validation, selects the top model, generates SHAP plots, and saves to `saved_models/best_cognitive_model.pkl`.*

### Train EEG Signal Model
```bash
python train_eeg.py
```
*Filters raw 16-channel EEG signals (`EEG dataset/AD_all_patients.csv`), computes Welch PSD and Hjorth parameters, trains ML and Deep Learning (CNN, LSTM, CNN-LSTM) architectures, and saves `saved_models/best_eeg_model.pkl`.*

### Train Speech Audio Model
```bash
python train_speech.py
```
*Extracts 13 MFCCs, pitch, RMS energy, ZCR, Mel Spectrogram stats, jitter, shimmer, and speaking rate from `.wav` files, trains ML and DL models, and saves `saved_models/best_speech_model.pkl`.*

---

## 🌐 Running the Flask Web Application

Launch the local dev server:
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### Portal Capabilities:
- **Interactive Form**: Input patient clinical metrics (Age, MMSE, CDR, eTIV, nWBV, ASF, etc.).
- **EEG File Upload**: Upload raw `.csv` or `.edf` EEG recordings.
- **Speech Recording**: Record live microphone audio directly in the browser or upload `.wav` audio files.
- **Pre-fill Sample Buttons**: Click **Pre-fill Healthy Patient** or **Pre-fill AD Patient** for instant full-system demonstration.

---

## 📊 Publication Visualizations & Explainability

The system automatically generates high-resolution publication-quality figures saved in `static/plots/`:
1. **Confusion Matrix**: Classification performance breakdown.
2. **ROC Curves**: Receiver Operating Characteristic with AUC score.
3. **SHAP Summary & Waterfall Plots**: Feature attribution ranking for clinical decision support.
4. **EEG Power Spectral Density (PSD)**: Frequency decomposition across $\delta, \theta, \alpha, \beta, \gamma$ bands.
5. **Speech Mel Spectrogram**: Audio energy distribution across mel scale frequencies.

---

## 📜 License & Citation

Designed for academic research, biomedical signal processing studies, and engineering capstone projects.
