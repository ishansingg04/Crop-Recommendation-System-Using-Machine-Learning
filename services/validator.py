"""
Agronomic and Physical Input Validation Module.
Validates the seven environmental and soil features required for crop recommendation:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

Differentiates strictly between:
1. Physical / Agronomic impossibility (Invalid -> Rejected with ValueError)
2. Training distribution bounds (Valid but unfamiliar -> Generates Reliability Warning)
"""

import math
from typing import Dict, List, Tuple, Any


class ValidationError(ValueError):
    """Raised when an input violates physical, chemical, or agronomic boundaries."""
    pass


# Documented physical and chemical boundaries
# Sources:
# - Brady & Weil, "The Nature and Properties of Soils" (15th Ed.)
# - FAO / ICAR Agronomic Guidelines for Indian & Subtropical Agriculture
PHYSICAL_LIMITS = {
    'N': {
        'name': 'Nitrogen',
        'unit': 'kg/ha',
        'min': 0.0,
        'max': 300.0,
        'desc': 'Soil available nitrogen cannot be negative, and values above 300 kg/ha cause nitrogen toxicity in field crops.'
    },
    'P': {
        'name': 'Phosphorus',
        'unit': 'kg/ha',
        'min': 0.0,
        'max': 250.0,
        'desc': 'Soil available phosphorus cannot be negative, and values above 250 kg/ha exceed agronomic saturation.'
    },
    'K': {
        'name': 'Potassium',
        'unit': 'kg/ha',
        'min': 0.0,
        'max': 350.0,
        'desc': 'Soil available potassium cannot be negative, and values above 350 kg/ha represent extreme non-agricultural salinity.'
    },
    'temperature': {
        'name': 'Temperature',
        'unit': '°C',
        'min': -10.0,
        'max': 60.0,
        'desc': 'Terrestrial crop vegetation cannot sustain growth outside the range of -10°C to 60°C.'
    },
    'humidity': {
        'name': 'Humidity',
        'unit': '%',
        'min': 0.0,
        'max': 100.0,
        'desc': 'Relative humidity is physically bounded between 0% and 100%.'
    },
    'ph': {
        'name': 'Soil pH',
        'unit': 'pH scale',
        'min': 0.0,
        'max': 14.0,
        'desc': 'Chemical aqueous pH scale is bounded between 0.0 and 14.0.'
    },
    'rainfall': {
        'name': 'Rainfall',
        'unit': 'mm',
        'min': 0.0,
        'max': 3500.0,
        'desc': 'Precipitation cannot be negative, and agricultural rainfall rarely exceeds 3,500 mm in a single crop season.'
    }
}

# Empirical training dataset boundaries (Crop_recommendation.csv, 2200 samples)
TRAINING_BOUNDS = {
    'N': (0.0, 140.0),
    'P': (5.0, 145.0),
    'K': (5.0, 205.0),
    'temperature': (8.8, 43.7),
    'humidity': (14.2, 100.0),
    'ph': (3.5, 9.9),
    'rainfall': (20.2, 298.6)
}

# Accepted field aliases to support various form / API encodings
FIELD_ALIASES = {
    'N': ['n', 'nitrogen', 'Nitrogen'],
    'P': ['p', 'phosphorus', 'Phosphorus', 'phosporus', 'Phosporus'],
    'K': ['k', 'potassium', 'Potassium'],
    'temperature': ['temperature', 'Temperature', 'temp', 'Temp'],
    'humidity': ['humidity', 'Humidity', 'humid'],
    'ph': ['ph', 'pH', 'Ph', 'PH', 'soil_ph', 'Soil_pH'],
    'rainfall': ['rainfall', 'Rainfall', 'rain', 'Rain']
}

CANONICAL_FEATURE_ORDER = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


class AgronomicValidator:
    """Validator ensuring high data integrity, strict physical bounds, and OOD detection."""

    @classmethod
    def extract_value(cls, data: Dict[str, Any], canonical_key: str) -> Any:
        """Finds value from data dictionary using canonical name or aliases."""
        aliases = FIELD_ALIASES.get(canonical_key, [canonical_key])
        for alias in aliases:
            if alias in data:
                val = data[alias]
                if val is not None and str(val).strip() != "":
                    return val
        return None

    @classmethod
    def validate_inputs(cls, data: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
        """
        Validates all seven input features.
        
        Returns:
            Tuple of:
            - validated_values: dict mapping canonical feature names to parsed floats
            - warnings: list of out-of-distribution warning strings

        Raises:
            ValidationError: If any value is missing, non-numeric, NaN, infinite,
                             or violates physical/agronomic limits.
        """
        validated_values: Dict[str, float] = {}
        warnings: List[str] = []

        for feature in CANONICAL_FEATURE_ORDER:
            raw_val = cls.extract_value(data, feature)
            limits = PHYSICAL_LIMITS[feature]

            # 1. Missing or empty check
            if raw_val is None:
                raise ValidationError(
                    f"Missing required field: '{limits['name']}' ({limits['unit']}). Please provide a numeric value."
                )

            # 2. Numeric parsing check
            try:
                val = float(raw_val)
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Invalid input for '{limits['name']}': received '{raw_val}'. Expected a valid floating point number."
                )

            # 3. NaN or Infinite check
            if math.isnan(val) or math.isinf(val):
                raise ValidationError(
                    f"Invalid numeric value for '{limits['name']}': NaN or Infinite values are not permitted."
                )

            # 4. Physical / Agronomic limits check
            if val < limits['min'] or val > limits['max']:
                raise ValidationError(
                    f"Agronomic boundary exceeded for '{limits['name']}': {val} {limits['unit']}. "
                    f"Must be between {limits['min']} and {limits['max']} {limits['unit']}. {limits['desc']}"
                )

            validated_values[feature] = val

            # 5. Training Distribution Bounds check (Out-of-Distribution Warning)
            train_min, train_max = TRAINING_BOUNDS[feature]
            if val < train_min or val > train_max:
                warnings.append(
                    f"'{limits['name']}' value ({val} {limits['unit']}) is outside the model's training range "
                    f"({train_min} to {train_max} {limits['unit']}). The recommendation may have higher uncertainty."
                )

        return validated_values, warnings

    @classmethod
    def get_feature_vector(cls, validated_dict: Dict[str, float]) -> List[float]:
        """Returns the canonical ordered float list ready for scikit-learn transformers."""
        return [validated_dict[f] for f in CANONICAL_FEATURE_ORDER]
