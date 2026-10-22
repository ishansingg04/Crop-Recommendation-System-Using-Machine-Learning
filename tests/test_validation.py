"""Unit tests for Agronomic and Physical Input Validation."""
import pytest
from services.validator import AgronomicValidator, ValidationError


class TestAgronomicValidation:

    def test_valid_standard_input(self):
        """Tests that typical valid inputs pass validation with no warnings."""
        data = {
            'Nitrogen': '90',
            'Phosporus': '42',
            'Potassium': '43',
            'Temperature': '20.8',
            'Humidity': '82.0',
            'Ph': '6.5',
            'Rainfall': '202.9'
        }
        val, warnings = AgronomicValidator.validate_inputs(data)
        assert val['N'] == 90.0
        assert val['P'] == 42.0
        assert val['K'] == 43.0
        assert val['temperature'] == 20.8
        assert val['humidity'] == 82.0
        assert val['ph'] == 6.5
        assert val['rainfall'] == 202.9
        assert len(warnings) == 0

    def test_missing_feature(self):
        """Tests that missing inputs raise ValidationError."""
        data = {
            'Nitrogen': '90',
            'Potassium': '43',
            'Temperature': '20.8'
        }
        with pytest.raises(ValidationError, match="Missing required field"):
            AgronomicValidator.validate_inputs(data)

    def test_non_numeric_input(self):
        """Tests that non-numeric characters raise ValidationError."""
        data = {
            'Nitrogen': 'abc',
            'Phosporus': '42',
            'Potassium': '43',
            'Temperature': '20.8',
            'Humidity': '82.0',
            'Ph': '6.5',
            'Rainfall': '202.9'
        }
        with pytest.raises(ValidationError, match="Invalid input for 'Nitrogen'"):
            AgronomicValidator.validate_inputs(data)

    def test_nan_and_inf_rejected(self):
        """Tests that NaN and Infinite values are rejected."""
        data = {
            'Nitrogen': 'nan',
            'Phosporus': '42',
            'Potassium': '43',
            'Temperature': '20.8',
            'Humidity': '82.0',
            'Ph': '6.5',
            'Rainfall': '202.9'
        }
        with pytest.raises(ValidationError, match="NaN or Infinite"):
            AgronomicValidator.validate_inputs(data)

        data['Nitrogen'] = 'inf'
        with pytest.raises(ValidationError, match="NaN or Infinite"):
            AgronomicValidator.validate_inputs(data)

    def test_physical_boundary_violations(self):
        """Tests that physically impossible values are strictly rejected."""
        base = {
            'Nitrogen': '90', 'Phosporus': '42', 'Potassium': '43',
            'Temperature': '20.8', 'Humidity': '82.0', 'Ph': '6.5', 'Rainfall': '202.9'
        }

        # Negative values
        with pytest.raises(ValidationError, match="Agronomic boundary exceeded for 'Nitrogen'"):
            AgronomicValidator.validate_inputs({**base, 'Nitrogen': '-5'})

        with pytest.raises(ValidationError, match="Agronomic boundary exceeded for 'Humidity'"):
            AgronomicValidator.validate_inputs({**base, 'Humidity': '125'})

        with pytest.raises(ValidationError, match="Agronomic boundary exceeded for 'Soil pH'"):
            AgronomicValidator.validate_inputs({**base, 'Ph': '15'})

        with pytest.raises(ValidationError, match="Agronomic boundary exceeded for 'Rainfall'"):
            AgronomicValidator.validate_inputs({**base, 'Rainfall': '-1'})

    def test_out_of_distribution_reliability_warnings(self):
        """
        Tests that values which are physically possible but outside the training distribution
        are accepted, but generate explicit reliability warnings.
        """
        data = {
            'Nitrogen': '180',  # Physically possible (0-300), but outside training (0-140)
            'Phosporus': '42',
            'Potassium': '43',
            'Temperature': '20.8',
            'Humidity': '82.0',
            'Ph': '6.5',
            'Rainfall': '450'   # Physically possible (0-3500), but outside training (20.2-298.6)
        }
        val, warnings = AgronomicValidator.validate_inputs(data)
        assert val['N'] == 180.0
        assert val['rainfall'] == 450.0
        assert len(warnings) == 2
        assert any("Nitrogen" in w for w in warnings)
        assert any("Rainfall" in w for w in warnings)
