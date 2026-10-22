"""Unit tests for Explainable AI (SHAP) module."""
import pytest
from services.model_service import ModelService


class TestShapExplanation:

    @classmethod
    def setup_class(cls):
        cls.service = ModelService.get_instance()

    def test_local_shap_explanation_structure(self):
        """Tests that explain_prediction returns 7 feature attributions for the top crop."""
        sample_dict = {
            'N': 90.0, 'P': 42.0, 'K': 43.0,
            'temperature': 20.8, 'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9
        }
        res = self.service.explain_prediction(sample_dict, target_class_id=1)  # Class 1 = Rice

        assert res["success"] is True
        assert res["explained_class_id"] == 1
        assert "contributions" in res

        contributions = res["contributions"]
        assert len(contributions) == 7

        # Check required fields in each contribution card
        for c in contributions:
            assert "feature_key" in c
            assert "display_name" in c
            assert "input_value" in c
            assert "shap_value" in c
            assert "direction" in c
            assert "is_positive" in c

        # Rainfall should be positive and among top contributions for Rice with 202.9mm rain
        rainfall_c = next(c for c in contributions if c["feature_key"] == "rainfall")
        assert rainfall_c["shap_value"] > 0
        assert rainfall_c["is_positive"] is True

    def test_global_importance_available(self):
        """Tests that global importance contains all 7 features."""
        global_imp = ModelService.PRECOMPUTED_GLOBAL_SHAP
        assert len(global_imp) == 7
        assert "Rainfall" in global_imp
        assert "Humidity" in global_imp
        assert global_imp["Rainfall"] > 0

    def test_fallback_explanation_resilience(self):
        """Tests that if SHAP encounters an issue, fallback explanation gracefully returns."""
        sample_dict = {
            'N': 90.0, 'P': 42.0, 'K': 43.0,
            'temperature': 20.8, 'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9
        }
        fallback = self.service._fallback_explanation(sample_dict, error_msg="Simulated SHAP error")
        assert fallback["success"] is False
        assert len(fallback["contributions"]) == 7
        assert "Gini" in fallback["contributions"][0]["direction"]
