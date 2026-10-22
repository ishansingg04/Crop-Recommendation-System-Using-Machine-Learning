"""Unit tests for ML Prediction Pipeline and Top-3 Crop Ranking."""
import pytest
import numpy as np
from services.model_service import ModelService
from services.crop_data import CROP_AGRONOMIC_DATA


class TestModelPrediction:

    @classmethod
    def setup_class(cls):
        cls.service = ModelService.get_instance()

    def test_pipeline_transforms_shape(self):
        """Tests that 7-feature input vector transforms to (1, 7) scaled output."""
        sample = [90, 42, 43, 20.8, 82.0, 6.5, 202.9]
        transformed = self.service.transform_features(sample)
        assert transformed.shape == (1, 7)
        assert not np.isnan(transformed).any()

    def test_top_3_ranking_structure(self):
        """Tests that predict_top_k returns exactly 3 ranked crops sorted descending."""
        sample_dict = {
            'N': 90.0, 'P': 42.0, 'K': 43.0,
            'temperature': 20.8, 'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9
        }
        res = self.service.predict_top_k(sample_dict, k=3)

        assert "primary_crop" in res
        assert res["primary_crop"] == "Rice"
        assert res["result_text"] == "Rice is the best crop to be cultivated right there"

        recs = res["top_recommendations"]
        assert len(recs) == 3

        # Ranks must be 1, 2, 3
        assert [r["rank"] for r in recs] == [1, 2, 3]

        # Probabilities must be descending
        probs = [r["probability"] for r in recs]
        assert probs[0] >= probs[1] >= probs[2]

        # Score labels must mention Ensemble Tree Vote
        assert "Ensemble Tree Vote" in recs[0]["score_label"]

    def test_all_22_classes_mapped(self):
        """Tests that all 22 target classes exist in CROP_DICT and have agronomic data."""
        assert len(self.service.CROP_DICT) == 22
        for class_id in range(1, 23):
            crop_name = self.service.CROP_DICT[class_id]
            assert crop_name.lower() in CROP_AGRONOMIC_DATA

    def test_deterministic_known_presets(self):
        """Tests known sample presets from dataset produce expected primary crops."""
        # Preset for Coffee: N=100, P=25, K=30, temp=25.5, humidity=60.0, ph=6.8, rainfall=160.0
        coffee_sample = {
            'N': 100.0, 'P': 25.0, 'K': 30.0,
            'temperature': 25.5, 'humidity': 60.0, 'ph': 6.8, 'rainfall': 160.0
        }
        res = self.service.predict_top_k(coffee_sample)
        assert res["primary_crop"] == "Coffee"

        # Preset for Chickpea: N=40, P=72, K=77, temp=17.0, humidity=17.0, ph=7.5, rainfall=88.5
        chickpea_sample = {
            'N': 40.0, 'P': 72.0, 'K': 77.0,
            'temperature': 17.0, 'humidity': 17.0, 'ph': 7.5, 'rainfall': 88.5
        }
        res = self.service.predict_top_k(chickpea_sample)
        assert res["primary_crop"] == "Chickpea"
