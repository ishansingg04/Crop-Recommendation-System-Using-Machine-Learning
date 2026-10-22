"""Integration tests for Flask application routes and REST APIs."""
import pytest
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestFlaskEndpoints:

    def test_index_page_loads(self, client):
        """Tests that GET / renders the standby dashboard."""
        response = client.get('/')
        assert response.status_code == 200
        assert b"AgriVision AI" in response.data
        assert b"Compute Crop Recommendation" in response.data

    def test_form_prediction_success(self, client):
        """Tests standard POST /predict returns recommendations and SHAP explanations."""
        data = {
            'Nitrogen': '90',
            'Phosporus': '42',
            'Potassium': '43',
            'Temperature': '20.88',
            'Humidity': '82.0',
            'Ph': '6.5',
            'Rainfall': '202.94'
        }
        response = client.post('/predict', data=data)
        assert response.status_code == 200
        assert b"Rice" in response.data
        assert b"Primary Recommendation" in response.data
        assert b"Top Recommended Crops" in response.data
        assert b"Explainable AI (SHAP" in response.data

    def test_form_prediction_invalid_input(self, client):
        """Tests that POST /predict with invalid inputs returns 400 with friendly error."""
        data = {
            'Nitrogen': '-50',  # Negative value
            'Phosporus': '42',
            'Potassium': '43',
            'Temperature': '20.88',
            'Humidity': '82.0',
            'Ph': '6.5',
            'Rainfall': '202.94'
        }
        response = client.post('/predict', data=data)
        assert response.status_code == 400
        assert b"Validation Error" in response.data
        assert b"boundary exceeded" in response.data

    def test_api_predict_json(self, client):
        """Tests JSON REST endpoint /api/predict."""
        payload = {
            'Nitrogen': 75,
            'Phosporus': 48,
            'Potassium': 20,
            'Temperature': 22.5,
            'Humidity': 65.0,
            'Ph': 6.2,
            'Rainfall': 85.0
        }
        response = client.post('/api/predict', json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["primary_crop"] == "Maize"
        assert len(data["top_recommendations"]) == 3
        assert data["shap_explanation"]["success"] is True
        assert len(data["comparison"]) == 3

    def test_api_predict_invalid_returns_400(self, client):
        """Tests that /api/predict returns 400 JSON when a field is missing."""
        payload = {
            'Nitrogen': 75,
            'Phosporus': 48
            # Missing other 5 fields
        }
        response = client.post('/api/predict', json=payload)
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "ValidationError" in data["error_type"]

    def test_api_crop_info_endpoint(self, client):
        """Tests GET /api/crop-info/<crop_name>."""
        response = client.get('/api/crop-info/banana')
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["crop"]["name"] == "Banana"
        assert data["crop"]["scientific_name"] == "Musa acuminata"

    def test_api_metadata_endpoint(self, client):
        """Tests GET /api/metadata."""
        response = client.get('/api/metadata')
        assert response.status_code == 200
        data = response.get_json()
        assert len(data["supported_crops"]) == 22
        assert "Nitrogen" in data["physical_limits"]["N"]["name"]

    def test_api_global_shap_endpoint(self, client):
        """Tests GET /api/global-shap."""
        response = client.get('/api/global-shap')
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "Rainfall" in data["global_importance"]

    def test_api_weather_endpoint_success_or_graceful_error(self, client):
        """Tests GET /api/weather?city=Delhi."""
        response = client.get('/api/weather?city=Delhi')
        assert response.status_code in (200, 400)
        data = response.get_json()
        assert "success" in data
        if data["success"]:
            assert "temperature_c" in data
            assert "disclaimer" in data

    def test_api_weather_missing_city(self, client):
        """Tests GET /api/weather with empty city query."""
        response = client.get('/api/weather')
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
