# AgriVision AI — Production Crop Recommendation System with Explainable AI & Robustness Benchmarking

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-brightgreen.svg)](https://github.com/slundberg/shap)
[![Flask](https://img.shields.io/badge/Web%20Framework-Flask%203.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-23%20Passed-success.svg)](https://docs.pytest.org/)

An enterprise-grade, precision agriculture machine learning platform designed to assist farmers, agricultural extension officers, and agronomists in making data-driven crop selection decisions.

---

## Table of Contents
1. [Architecture & Machine Learning Pipeline](#1-architecture--machine-learning-pipeline)
2. [Key Features & Innovations](#2-key-features--innovations)
3. [Project Structure](#3-project-structure)
4. [Installation & Setup](#4-installation--setup)
5. [Running the Application](#5-running-the-application)
6. [Executing Tests & Robustness Evaluation](#6-executing-tests--robustness-evaluation)
7. [Robustness & Reliability Evaluation Findings](#7-robustness--reliability-evaluation-findings)
8. [REST API Documentation](#8-rest-api-documentation)
9. [Limitations & Agronomic Assumptions](#9-limitations--agronomic-assumptions)
10. [Academic Research Roadmap](#10-academic-research-roadmap)

---

## 1. Architecture & Machine Learning Pipeline

### Machine Learning Algorithm
- **Model**: `RandomForestClassifier` (100 ensemble estimators)
- **Target Classes**: 22 commercial crops (Rice, Maize, Jute, Cotton, Coconut, Papaya, Orange, Apple, Muskmelon, Watermelon, Grapes, Mango, Banana, Pomegranate, Lentil, Blackgram, Mungbean, Mothbeans, Pigeonpeas, Kidneybeans, Chickpea, Coffee).
- **Baseline Test Accuracy**: **99.77%** (Macro-F1: 0.9977).

### Preprocessing & Transformation Pipeline
Input features must undergo the exact dual-stage transformation verified by the trained model:
$$\mathbf{x}_{\text{raw}} \in \mathbb{R}^7 \xrightarrow{\text{MinMaxScaler}} \mathbf{x}_{\text{minmax}} \xrightarrow{\text{StandardScaler}} \mathbf{x}_{\text{standard}} \xrightarrow{\text{RandomForest}} \hat{\mathbf{y}}$$

```
[N, P, K, Temp, Humidity, pH, Rainfall] 
   └──> MinMaxScaler (`minmaxscaler.pkl`)
          └──> StandardScaler (`standscaler.pkl`)
                 └──> RandomForest (`model.pkl`) ──> Predict Proba / Classes
```

> **Note on Feature Order**: Features must strictly follow the canonical sequence:
> `['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']`.

---

## 2. Key Features & Innovations

### Feature 1: Agronomic Input Validation & Reliability Warnings
- **Server-Side Validation**: Never relies on client JavaScript alone. Rejects missing, malformed, non-numeric, NaN, infinite, or physically impossible values (e.g., negative rainfall, pH outside 0–14).
- **Reliability Warnings (Out-of-Distribution Detection)**: When an input is physically possible (e.g., Nitrogen = 180 kg/ha or Rainfall = 450 mm) but falls outside the empirical training envelope (N: 0–140, Rainfall: 20.2–298.6 mm), the system does not crash or replace inputs with arbitrary defaults. Instead, it processes the prediction and renders an **amber reliability warning** highlighting potential model uncertainty.

### Feature 2: Top-3 Ranked Crop Recommendations
- Ranks candidate crops using `model.predict_proba()`.
- Accurately labeled as **Ensemble Tree Vote Score** (% of individual decision trees in the ensemble voting for each crop).
- Prevents misleading claims of "calibrated absolute probability" while giving farmers realistic primary, runner-up, and alternative options.

### Feature 3: Explainable AI (SHAP)
- **Local Explanations**: Uses `shap.TreeExplainer` to compute exact feature attributions for the top recommended crop. Features pushing towards the recommendation are highlighted in emerald (+), while unfavorable features are shown in rose (-).
- **Global Feature Importance**: Exposes precomputed dataset-wide mean absolute SHAP values:
  1. Rainfall (0.0301)
  2. Humidity (0.0301)
  3. Potassium (0.0277)
  4. Phosphorus (0.0250)
  5. Nitrogen (0.0239)
  6. Temperature (0.0101)
  7. Soil pH (0.0054)
- **Graceful Fallback**: If SHAP computation encounters any runtime incompatibility, the system automatically falls back to Random Forest Gini feature importances so predictions never fail.

### Feature 4: Model Robustness & Reliability Benchmark Suite
- An independent evaluation module (`evaluate_robustness.py`) testing:
  - Gaussian measurement noise ($\sigma \in \{1\%, 5\%, 10\%, 20\%\}$).
  - Sensor calibration drift / bias ($\pm 5\%, \pm 10\%$).
  - Sensor dropouts with training median imputation.
  - Local perturbation stability / prediction consistency under infinitesimal jitter.

### Feature 5: Crop Comparison Dashboard
- Displays authentic, traceable agronomic parameters side-by-side for the recommended crops:
  - Scientific Name & Botanical Family
  - Optimal Growing Season (Kharif, Rabi, Zaid, Perennial)
  - Crop Duration / Maturity (days)
  - Soil Type & pH Range
  - Water & Rainfall Requirements
  - Temperature Tolerance
  - Fertilizer N-P-K demand profile
- **Traceable Sources**: FAO EcoCrop Database, ICAR Handbook of Agriculture.

### Feature 6: Estimated Economic Returns & Profitability Analysis
- Provides per-hectare cost of cultivation, benchmark yield (q/ha), market rate / MSP (₹/q), gross revenue, and estimated net profit.
- **Traceable Sources**: Commission for Agricultural Costs and Prices (CACP) Price Policy Reports 2024-25 and Agmarknet wholesale market averages.
- Includes clear economic disclaimers emphasizing regional price and weather variation.

### Feature 7: Live Weather Integration
- Integrates with Open-Meteo High-Resolution API (free, reliable, global geocoding, no mandatory API key) with optional OpenWeatherMap support.
- Allows farmers to search by district/city name and auto-fill current temperature, humidity, and rainfall into the telemetry form with one click.
- Prominent agronomic disclaimer separates instantaneous weather snapshots from seasonal climate requirements.

### Feature 8: Bilingual Interface (English / हिन्दी)
- One-click language switch in the header translating all labels, section headers, instructions, warnings, and error messages without page reloads.

### Feature 9: Modern UI/UX
- Responsive dark glassmorphism layout with Outfit and Plus Jakarta Sans typography.
- Synchronized dual-controls (interactive range sliders + numeric input boxes).
- Quick-fill presets (Rice, Maize, Banana, Coffee, Chickpea, Watermelon).
- Accessible contrast, animated loading states, and mobile responsiveness.

---

## 3. Project Structure

```
Crop-Recommendation-System-Using-Machine-Learning/
├── Crop_recommendation.csv         # Benchmark dataset (2,200 rows, 22 classes)
├── model.pkl                       # Serialized RandomForestClassifier
├── minmaxscaler.pkl                # Serialized MinMaxScaler (Stage 1)
├── standscaler.pkl                 # Serialized StandardScaler (Stage 2)
├── app.py                          # Flask application & REST API
├── evaluate_robustness.py          # Robustness & reliability benchmarking module
├── requirements.txt                # Pinned production dependencies
├── ROBUSTNESS_REPORT.md            # Quantitative benchmark results report
├── robustness_metrics.json         # Raw benchmark output data
├── services/                       # Modular business & ML logic
│   ├── __init__.py                 # Services export interface
│   ├── validator.py                # Agronomic bounds & OOD warning validator
│   ├── model_service.py            # Singleton ML inference & SHAP explainer
│   ├── crop_data.py                # Agronomic & economic database + i18n
│   └── weather_service.py          # Open-Meteo & OpenWeatherMap integration
├── templates/
│   └── index.html                  # Responsive dashboard UI
├── tests/                          # Automated PyTest suite
│   ├── test_validation.py          # Input boundary & OOD tests
│   ├── test_prediction.py          # Preprocessing & top-3 ranking tests
│   ├── test_shap.py                # SHAP attribution & fallback tests
│   └── test_app.py                 # Flask web & REST API integration tests
└── README.md                       # Comprehensive system documentation
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Pip package manager

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Crop-Recommendation-System-Using-Machine-Learning.git
   cd Crop-Recommendation-System-Using-Machine-Learning
   ```

2. **Create and activate a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Running the Application

### Start the Flask Server
```bash
python app.py
```
By default, the server starts at `http://localhost:5000`.

### Production Configuration
To configure production settings via environment variables:
```bash
# Set custom port (default 5000)
set PORT=8080
# Enable debug mode only for local dev (default is False)
set FLASK_DEBUG=0
# Optional: Set OpenWeatherMap API Key (Open-Meteo is used by default with no key required)
set OPENWEATHER_API_KEY=your_key_here

python app.py
```

---

## 6. Executing Tests & Robustness Evaluation

### Run Automated PyTest Suite
Execute all 23 unit and integration tests covering validation, scaling, prediction, SHAP, and REST APIs:
```bash
python -m pytest -v
```

### Run Robustness & Reliability Benchmark
Execute the standalone stress-testing harness to evaluate sensor noise, drift, and missing values:
```bash
python evaluate_robustness.py
```
This generates `ROBUSTNESS_REPORT.md` and `robustness_metrics.json`.

---

## 7. Robustness & Reliability Evaluation Findings

The benchmark suite evaluated **440 independent stratified test samples** (fixed seed = 42) across 5 degradation scenarios:

| Evaluation Scenario | Test Accuracy (%) | Macro-F1 | Prediction Consistency (%) | Degradation vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Test Set (Clean)** | **99.77%** | **0.9977** | **100.00%** | **Baseline (0.00%)** |
| Gaussian Noise ($\sigma = 1\%$) | **99.77%** | 0.9977 | 100.00% | 0.00% |
| Gaussian Noise ($\sigma = 5\%$) | **98.64%** | 0.9864 | 98.86% | -1.13% |
| Gaussian Noise ($\sigma = 10\%$) | **95.91%** | 0.9589 | 95.68% | -3.86% |
| Gaussian Noise ($\sigma = 20\%$) | **86.82%** | 0.8659 | 86.59% | -12.95% |
| Sensor Calibration Drift (+5%) | **97.27%** | 0.9723 | 97.50% | -2.50% |
| Sensor Calibration Drift (-5%) | **94.09%** | 0.9393 | 93.86% | -5.68% |
| Sensor Calibration Drift (-10%) | **79.77%** | 0.7898 | 79.55% | -20.00% |
| Missing Sensor: Temperature (Median Imputed) | **99.55%** | 0.9955 | 99.77% | -0.22% |
| Missing Sensor: Soil pH (Median Imputed) | **99.09%** | 0.9908 | 99.32% | -0.68% |
| Missing Sensor: Phosphorus (Median Imputed) | **94.09%** | 0.9402 | 93.86% | -5.68% |
| Missing Sensor: Nitrogen (Median Imputed) | **86.36%** | 0.8253 | 86.59% | -13.41% |
| Missing Sensor: Potassium (Median Imputed) | **86.36%** | 0.8272 | 86.59% | -13.41% |
| Missing Sensor: Humidity (Median Imputed) | **85.91%** | 0.8351 | 85.91% | -13.86% |
| **Missing Sensor: Rainfall (Median Imputed)** | **77.95%** | **0.7343** | **77.95%** | **-21.82%** |
| Input Jitter Stability ($\epsilon = \pm 1\%$) | **99.55%** | 0.9955 | 99.77% | -0.22% |
| Input Jitter Stability ($\epsilon = \pm 2\%$) | **98.86%** | 0.9886 | 98.64% | -0.91% |
| Input Jitter Stability ($\epsilon = \pm 5\%$) | **97.95%** | 0.9796 | 98.18% | -1.82% |

### Key Agronomic Insights from Evaluation
1. **Rainfall Criticality**: Missing rainfall telemetry causes a **21.82% accuracy drop**, confirming SHAP analysis that rainfall is the primary decision boundary distinguishing water-intensive crops (rice, jute) from arid pulses (mothbeans, chickpea).
2. **Sensor Bias Sensitivity**: Negative systematic calibration drift (-10%) produces higher error rates than positive drift (+10%), as moisture and nutrient deficits severely alter classification boundaries.
3. **Resilience to Mild Noise**: The Random Forest ensemble demonstrates strong robustness against standard IoT field sensor noise ($\le 5\%$), preserving over 98.6% accuracy.

---

## 8. REST API Documentation

### 1. Crop Recommendation Inference
- **Endpoint**: `POST /api/predict`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "Nitrogen": 90,
    "Phosporus": 42,
    "Potassium": 43,
    "Temperature": 20.8,
    "Humidity": 82.0,
    "Ph": 6.5,
    "Rainfall": 202.9
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "primary_crop": "Rice",
    "result_text": "Rice is the best crop to be cultivated right there",
    "top_recommendations": [
      {
        "rank": 1,
        "crop_name": "Rice",
        "score_percent": 100.0,
        "score_label": "100.0% Ensemble Tree Vote",
        "scientific_name": "Oryza sativa",
        "season": "Kharif (Monsoon)",
        "duration_days": "105 - 145 days"
      },
      {
        "rank": 2,
        "crop_name": "Chickpea",
        "score_percent": 0.0,
        "score_label": "0.0% Ensemble Tree Vote"
      },
      {
        "rank": 3,
        "crop_name": "Maize",
        "score_percent": 0.0,
        "score_label": "0.0% Ensemble Tree Vote"
      }
    ],
    "shap_explanation": {
      "success": true,
      "contributions": [
        {"feature_key": "rainfall", "display_name": "Rainfall", "shap_value": 0.2956, "direction": "Strong Positive (Favors Crop)"},
        {"feature_key": "K", "display_name": "Potassium (K)", "shap_value": 0.1655, "direction": "Strong Positive (Favors Crop)"}
      ]
    },
    "reliability_warnings": []
  }
  ```

### 2. Live Weather Query
- **Endpoint**: `GET /api/weather?city=Nagpur`
- **Response**:
  ```json
  {
    "success": true,
    "city": "Nagpur, Maharashtra, India",
    "temperature_c": 28.4,
    "humidity_percent": 62.0,
    "rainfall_mm": 0.0,
    "disclaimer": "Agronomic Notice: Live telemetry represents real-time atmospheric conditions..."
  }
  ```

### 3. Crop Knowledge Base
- **Endpoint**: `GET /api/crop-info/banana`
- **Endpoint**: `GET /api/global-shap`
- **Endpoint**: `GET /api/metadata`

---

## 9. Limitations & Agronomic Assumptions

1. **Environmental Scope**: The machine learning model is trained on 7 primary parameters ($N, P, K$, Temperature, Humidity, pH, Rainfall). It does not directly observe secondary micronutrients (Zinc, Boron, Iron), soil drainage capacity, water table depth, or solar irradiance.
2. **Out-of-Distribution Behavior**: Random Forest decision trees partition feature space using axis-aligned orthogonal cuts. For values outside the training distribution, tree models output predictions based on the nearest bounding leaves rather than extrapolating trends. Users must heed the built-in reliability warnings.
3. **No Commercial Guarantee**: Recommendations represent biological and statistical suitability, not economic guarantees. Local market demand, pest outbreaks, hail, and unseasonal weather events affect realized harvest success.

---

## 10. Academic Research Roadmap

For researchers seeking to publish or extend this work:

1. **Uncertainty Quantification (Conformal Prediction)**:
   - Implement conformal prediction sets to provide mathematically guaranteed coverage ($1 - \alpha$) rather than point estimates.
2. **Multimodal Deep Learning & Remote Sensing**:
   - Fuse tabular soil telemetry with multispectral satellite imagery (Sentinel-2 / Landsat NDVI & NDWI indices) and digital elevation models (DEM) for topographical slope awareness.
3. **Multi-Objective Optimization (Crop Suitability vs. Economic Risk)**:
   - Formulate crop recommendation as a Pareto-frontier optimization problem balancing agronomic yield probability, water footprint, and market price volatility.
4. **Dynamic Climate Change Downscaling**:
   - Couple recommendation models with IPCC CMIP6 climate projection models to project crop viability shifts over 2030–2050.
