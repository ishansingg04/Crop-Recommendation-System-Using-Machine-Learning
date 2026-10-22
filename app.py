"""
AgriVision AI - Production Crop Recommendation Engine
Flask Web Application & REST API
"""

import os
from flask import Flask, request, render_template, jsonify
from services import (
    AgronomicValidator,
    ValidationError,
    ModelService,
    WeatherService,
    CROP_AGRONOMIC_DATA,
    I18N_STRINGS,
    get_crop_details,
    CANONICAL_FEATURE_ORDER,
    PHYSICAL_LIMITS,
    TRAINING_BOUNDS
)

app = Flask(__name__)
# Ensure JSON responses don't sort keys alphabetically
app.config['JSON_SORT_KEYS'] = False

# Initialize singleton ML Model Service once at app startup
model_service = ModelService.get_instance()


@app.route('/', methods=['GET'])
def index():
    """Renders the main dashboard in standby state."""
    return render_template(
        "index.html",
        inputs=None,
        result=None,
        primary_crop=None,
        top_recommendations=None,
        shap_data=None,
        comparison_crops=None,
        warnings=None,
        error=None,
        i18n=I18N_STRINGS["en"]
    )


@app.route("/predict", methods=['POST'])
def predict():
    """
    Form submission endpoint for crop recommendation.
    Preserves backward compatibility while delivering Top 3 ranking,
    SHAP local attributions, OOD reliability warnings, and crop comparison data.
    """
    raw_data = request.form.to_dict()

    try:
        # Step 1: Validate inputs against strict physical & agronomic limits
        validated_inputs, warnings = AgronomicValidator.validate_inputs(raw_data)

        # Step 2: Inference & Top-3 Ranking
        prediction_result = model_service.predict_top_k(validated_inputs, k=3)
        top_recommendations = prediction_result["top_recommendations"]
        primary_crop = prediction_result["primary_crop"]
        legacy_result_text = prediction_result["result_text"]

        # Step 3: Compute Explainable AI (SHAP) attributions for primary crop
        primary_class_id = top_recommendations[0]["class_id"]
        shap_data = model_service.explain_prediction(validated_inputs, primary_class_id)

        # Step 4: Assemble comparison agronomic matrix for top recommendations
        comparison_crops = []
        for rec in top_recommendations:
            crop_name = rec["crop_name"]
            info = get_crop_details(crop_name)
            if info:
                comparison_crops.append(info)

        return render_template(
            "index.html",
            inputs=validated_inputs,
            result=legacy_result_text,
            primary_crop=primary_crop,
            top_recommendations=top_recommendations,
            shap_data=shap_data,
            comparison_crops=comparison_crops,
            warnings=warnings,
            error=None,
            i18n=I18N_STRINGS["en"]
        )

    except ValidationError as ve:
        # Return user-friendly error with original input preserved
        return render_template(
            "index.html",
            inputs=raw_data,
            result=None,
            primary_crop=None,
            top_recommendations=None,
            shap_data=None,
            comparison_crops=None,
            warnings=None,
            error=str(ve),
            i18n=I18N_STRINGS["en"]
        ), 400

    except Exception as e:
        # Non-leaking generic exception handler
        app.logger.error(f"Prediction processing error: {e}", exc_info=True)
        return render_template(
            "index.html",
            inputs=raw_data,
            result=None,
            primary_crop=None,
            top_recommendations=None,
            shap_data=None,
            comparison_crops=None,
            warnings=None,
            error="An internal error occurred during prediction. Please verify your inputs and try again.",
            i18n=I18N_STRINGS["en"]
        ), 500


# ── REST API ENDPOINTS ─────────────────────────────────────────────────────────

@app.route("/api/predict", methods=['POST'])
def api_predict():
    """
    JSON API endpoint for programmatic inference, mobile apps, or async UI calls.
    Accepts JSON body or multipart/form-data.
    """
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = request.form.to_dict()

    try:
        validated_inputs, warnings = AgronomicValidator.validate_inputs(data)
        prediction_result = model_service.predict_top_k(validated_inputs, k=3)
        top_recommendations = prediction_result["top_recommendations"]
        primary_class_id = top_recommendations[0]["class_id"]
        shap_data = model_service.explain_prediction(validated_inputs, primary_class_id)

        comparison_crops = []
        for rec in top_recommendations:
            info = get_crop_details(rec["crop_name"])
            if info:
                comparison_crops.append(info)

        return jsonify({
            "success": True,
            "inputs": validated_inputs,
            "primary_crop": prediction_result["primary_crop"],
            "result_text": prediction_result["result_text"],
            "top_recommendations": top_recommendations,
            "shap_explanation": shap_data,
            "comparison": comparison_crops,
            "reliability_warnings": warnings,
            "error": None
        })

    except ValidationError as ve:
        return jsonify({
            "success": False,
            "error": str(ve),
            "error_type": "ValidationError"
        }), 400

    except Exception as e:
        app.logger.error(f"API prediction error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal inference error. Please verify input schema and numerical values.",
            "error_type": "InternalServerError"
        }), 500


@app.route("/api/weather", methods=['GET'])
def api_weather():
    """
    Fetches real-time weather parameters for a given city or region.
    Query param: ?city=Nagpur
    """
    city = request.args.get("city", "").strip()
    if not city:
        return jsonify({
            "success": False,
            "error": "Query parameter 'city' is required. Example: /api/weather?city=Delhi"
        }), 400

    weather_data = WeatherService.get_weather_by_city(city)
    status_code = 200 if weather_data.get("success") else 400
    return jsonify(weather_data), status_code


@app.route("/api/crop-info/<crop_name>", methods=['GET'])
def api_crop_info(crop_name):
    """Returns agronomic facts and economic benchmarks for any supported crop."""
    info = get_crop_details(crop_name)
    if not info:
        return jsonify({
            "success": False,
            "error": f"Crop '{crop_name}' is not recognized in the 22 supported crop classes."
        }), 404
    return jsonify({"success": True, "crop": info})


@app.route("/api/global-shap", methods=['GET'])
def api_global_shap():
    """Returns dataset-wide global feature importance derived via SHAP TreeExplainer."""
    return jsonify({
        "success": True,
        "method": "shap.TreeExplainer (Mean Absolute SHAP on stratified dataset sample)",
        "global_importance": ModelService.PRECOMPUTED_GLOBAL_SHAP
    })


@app.route("/api/metadata", methods=['GET'])
def api_metadata():
    """Returns physical limits, training bounds, and supported crop list."""
    return jsonify({
        "features": CANONICAL_FEATURE_ORDER,
        "physical_limits": PHYSICAL_LIMITS,
        "training_bounds": TRAINING_BOUNDS,
        "supported_crops": list(CROP_AGRONOMIC_DATA.keys())
    })


# ── ERROR HANDLERS ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/'):
        return jsonify({"success": False, "error": "Endpoint not found."}), 404
    return render_template("index.html", error="Page not found (404).", i18n=I18N_STRINGS["en"]), 404


@app.errorhandler(500)
def internal_server_error(error):
    if request.path.startswith('/api/'):
        return jsonify({"success": False, "error": "Internal server error."}), 500
    return render_template("index.html", error="An internal server error occurred.", i18n=I18N_STRINGS["en"]), 500


if __name__ == "__main__":
    # Disable Flask debug mode by default for security, enable only if FLASK_DEBUG=1
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)