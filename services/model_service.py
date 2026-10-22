"""
Machine Learning & Explainable AI (SHAP) Model Service.
Manages:
1. Pretrained MinMaxScaler, StandardScaler, and RandomForestClassifier.
2. Verified feature preprocessing order: raw -> MinMaxScaler -> StandardScaler -> Model.
3. Top-K probability ranking with accurate ensemble vote metrics.
4. Explainable AI via shap.TreeExplainer (local waterfall contributions & global importance).
5. Graceful fallback handling.
"""

import os
import pickle
import threading
import numpy as np
from typing import Dict, List, Any, Optional

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .validator import CANONICAL_FEATURE_ORDER, PHYSICAL_LIMITS
from .crop_data import CROP_AGRONOMIC_DATA, get_crop_details


class ModelService:
    """Thread-safe service encapsulating ML prediction and SHAP explanation pipelines."""

    _instance = None
    _lock = threading.Lock()

    CROP_DICT = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Precomputed Global SHAP Importances across dataset (mean absolute SHAP)
    # Computed on stratified dataset sample using shap.TreeExplainer
    PRECOMPUTED_GLOBAL_SHAP = {
        "Rainfall": 0.0301,
        "Humidity": 0.0301,
        "Potassium": 0.0277,
        "Phosphorus": 0.0250,
        "Nitrogen": 0.0239,
        "Temperature": 0.0101,
        "Soil pH": 0.0054
    }

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model = None
        self.sc = None
        self.ms = None
        self.explainer = None
        self._load_assets()

    @classmethod
    def get_instance(cls, base_dir: Optional[str] = None) -> 'ModelService':
        """Singleton accessor ensuring model and scalers are loaded only once."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(base_dir)
        return cls._instance

    def _load_assets(self):
        """Loads serialized scikit-learn models and scalers from disk."""
        model_path = os.path.join(self.base_dir, 'model.pkl')
        sc_path = os.path.join(self.base_dir, 'standscaler.pkl')
        ms_path = os.path.join(self.base_dir, 'minmaxscaler.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing at: {model_path}")
        if not os.path.exists(sc_path):
            raise FileNotFoundError(f"StandardScaler file missing at: {sc_path}")
        if not os.path.exists(ms_path):
            raise FileNotFoundError(f"MinMaxScaler file missing at: {ms_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(sc_path, 'rb') as f:
            self.sc = pickle.load(f)
        with open(ms_path, 'rb') as f:
            self.ms = pickle.load(f)

        # Initialize TreeExplainer once for ultra-fast local predictions
        if SHAP_AVAILABLE and self.model is not None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                print(f"[Warning] Failed to initialize SHAP TreeExplainer: {e}")
                self.explainer = None

    def transform_features(self, feature_list: List[float]) -> np.ndarray:
        """
        Transforms raw 7-feature input vector:
        1. Reshape to (1, 7)
        2. MinMaxScaler.transform()
        3. StandardScaler.transform()
        """
        raw_arr = np.array(feature_list, dtype=np.float64).reshape(1, -1)
        scaled_minmax = self.ms.transform(raw_arr)
        final_scaled = self.sc.transform(scaled_minmax)
        return final_scaled

    def predict_top_k(self, feature_dict: Dict[str, float], k: int = 3) -> Dict[str, Any]:
        """
        Performs inference and ranks the top-k crop recommendations.

        Returns:
            Dict containing:
            - primary_crop: str
            - result_text: legacy backward-compatible string
            - top_recommendations: list of top k recommendation dicts
            - supports_probabilities: bool
            - score_type_label: str
        """
        feature_vector = [feature_dict[feat] for feat in CANONICAL_FEATURE_ORDER]
        final_features = self.transform_features(feature_vector)

        supports_proba = hasattr(self.model, 'predict_proba')

        if supports_proba:
            # Multi-class probabilities of shape (1, 22)
            proba = self.model.predict_proba(final_features)[0]
            classes = self.model.classes_

            # Sort classes by descending probability
            sorted_indices = np.argsort(proba)[::-1]
            top_indices = sorted_indices[:k]

            recommendations = []
            for rank_idx, idx in enumerate(top_indices, start=1):
                class_id = int(classes[idx])
                crop_name = self.CROP_DICT.get(class_id, f"Crop_{class_id}")
                prob_score = float(proba[idx])
                pct_score = round(prob_score * 100.0, 1)

                crop_info = get_crop_details(crop_name) or {}

                recommendations.append({
                    "rank": rank_idx,
                    "class_id": class_id,
                    "crop_name": crop_name,
                    "probability": prob_score,
                    "score_percent": pct_score,
                    "score_label": f"{pct_score}% Ensemble Tree Vote",
                    "emoji": crop_info.get("emoji", "🌱"),
                    "scientific_name": crop_info.get("scientific_name", "N/A"),
                    "family": crop_info.get("family", "N/A"),
                    "season": crop_info.get("season", "N/A"),
                    "duration_days": crop_info.get("duration_days", "N/A"),
                    "optimal_ph": crop_info.get("optimal_ph", "N/A"),
                    "summary": crop_info.get("summary", "Suitable for cultivation under these environmental parameters."),
                    "economics": crop_info.get("economics", {})
                })

            primary_crop = recommendations[0]["crop_name"]
            score_type_label = (
                "RandomForest Ensemble Decision Tree Vote (Proportion of 100 decision trees voting for this class)"
            )
        else:
            # Transparent fallback if model does not support predict_proba
            pred_class = int(self.model.predict(final_features)[0])
            primary_crop = self.CROP_DICT.get(pred_class, f"Crop_{pred_class}")
            crop_info = get_crop_details(primary_crop) or {}

            recommendations = [{
                "rank": 1,
                "class_id": pred_class,
                "crop_name": primary_crop,
                "probability": None,
                "score_percent": 100.0,
                "score_label": "Deterministic Prediction (Probabilities not supported by model)",
                "emoji": crop_info.get("emoji", "🌱"),
                "scientific_name": crop_info.get("scientific_name", "N/A"),
                "family": crop_info.get("family", "N/A"),
                "season": crop_info.get("season", "N/A"),
                "duration_days": crop_info.get("duration_days", "N/A"),
                "optimal_ph": crop_info.get("optimal_ph", "N/A"),
                "summary": crop_info.get("summary", ""),
                "economics": crop_info.get("economics", {})
            }]
            score_type_label = "Discrete Single-Class Classification"

        result_text = f"{primary_crop} is the best crop to be cultivated right there"

        return {
            "primary_crop": primary_crop,
            "result_text": result_text,
            "top_recommendations": recommendations,
            "supports_probabilities": supports_proba,
            "score_type_label": score_type_label
        }

    def explain_prediction(
        self,
        feature_dict: Dict[str, float],
        target_class_id: int
    ) -> Dict[str, Any]:
        """
        Computes SHAP feature attributions for a given prediction.

        Args:
            feature_dict: Raw input features {N, P, K, temperature, ...}
            target_class_id: Class ID (1..22) to explain

        Returns:
            Dict containing feature contributions, base value, and visual metadata.
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._fallback_explanation(feature_dict)

        try:
            feature_vector = [feature_dict[feat] for feat in CANONICAL_FEATURE_ORDER]
            scaled_features = self.transform_features(feature_vector)

            # Compute SHAP values
            shap_output = self.explainer.shap_values(scaled_features)

            # Match class index in model.classes_
            class_indices = np.where(self.model.classes_ == target_class_id)[0]
            if len(class_indices) == 0:
                class_idx = 0
            else:
                class_idx = int(class_indices[0])

            # In modern shap, shap_output can be (1, 7, 22) or list of 22 (1, 7) arrays
            if isinstance(shap_output, list):
                raw_contributions = shap_output[class_idx][0]
            elif hasattr(shap_output, 'shape') and len(shap_output.shape) == 3:
                raw_contributions = shap_output[0, :, class_idx]
            else:
                raw_contributions = np.zeros(len(CANONICAL_FEATURE_ORDER))

            # Build readable feature attribution cards
            feature_contributions = []
            display_names = {
                'N': 'Nitrogen (N)',
                'P': 'Phosphorus (P)',
                'K': 'Potassium (K)',
                'temperature': 'Temperature',
                'humidity': 'Humidity',
                'ph': 'Soil pH',
                'rainfall': 'Rainfall'
            }

            for idx, feat_key in enumerate(CANONICAL_FEATURE_ORDER):
                val = float(raw_contributions[idx])
                input_val = feature_dict[feat_key]
                limits = PHYSICAL_LIMITS[feat_key]

                # Direction and effect
                if val > 0.05:
                    direction = "Strong Positive (Favors Crop)"
                    css_class = "impact-pos-high"
                elif val > 0:
                    direction = "Positive (Favors Crop)"
                    css_class = "impact-pos"
                elif val < -0.05:
                    direction = "Strong Negative (Works Against)"
                    css_class = "impact-neg-high"
                else:
                    direction = "Slight Negative / Neutral"
                    css_class = "impact-neg"

                feature_contributions.append({
                    "feature_key": feat_key,
                    "display_name": display_names[feat_key],
                    "input_value": input_val,
                    "unit": limits['unit'],
                    "shap_value": round(val, 4),
                    "direction": direction,
                    "css_class": css_class,
                    "is_positive": val >= 0
                })

            # Sort by absolute contribution descending
            feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            return {
                "success": True,
                "method": "shap.TreeExplainer",
                "explained_class_id": target_class_id,
                "contributions": feature_contributions,
                "global_importance": self.PRECOMPUTED_GLOBAL_SHAP,
                "error": None
            }

        except Exception as e:
            print(f"[Warning] SHAP calculation error: {e}")
            return self._fallback_explanation(feature_dict, error_msg=str(e))

    def _fallback_explanation(
        self,
        feature_dict: Dict[str, float],
        error_msg: Optional[str] = None
    ) -> Dict[str, Any]:
        """Provides a safe fallback explanation using Random Forest Gini feature importances."""
        display_names = {
            'N': 'Nitrogen (N)',
            'P': 'Phosphorus (P)',
            'K': 'Potassium (K)',
            'temperature': 'Temperature',
            'humidity': 'Humidity',
            'ph': 'Soil pH',
            'rainfall': 'Rainfall'
        }

        contributions = []
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for idx, feat_key in enumerate(CANONICAL_FEATURE_ORDER):
                imp = float(importances[idx])
                contributions.append({
                    "feature_key": feat_key,
                    "display_name": display_names[feat_key],
                    "input_value": feature_dict.get(feat_key, 0.0),
                    "unit": PHYSICAL_LIMITS[feat_key]['unit'],
                    "shap_value": round(imp, 4),
                    "direction": "Global Feature Weight (Gini)",
                    "css_class": "impact-pos",
                    "is_positive": True
                })
            contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "success": False,
            "method": "RandomForest.feature_importances_ (Fallback)",
            "explained_class_id": None,
            "contributions": contributions,
            "global_importance": self.PRECOMPUTED_GLOBAL_SHAP,
            "error": error_msg or "SHAP library unavailable or incompatible."
        }
