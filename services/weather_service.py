"""
Weather Integration Service.
Fetches real-time meteorological data (temperature, relative humidity, precipitation)
using Open-Meteo API (free, reliable, global geocoding, no mandatory API key)
with optional OpenWeatherMap fallback if OPENWEATHER_API_KEY environment variable is set.
"""

import os
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timezone


class WeatherService:
    """Service to safely query live weather data for user-selected geographic locations."""

    OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    REQUEST_TIMEOUT = 5  # seconds

    @classmethod
    def get_weather_by_city(cls, city_name: str) -> Dict[str, Any]:
        """
        Geocodes a city name and fetches current meteorological parameters.

        Returns:
            Dict containing:
            - success: bool
            - city: str
            - country: str
            - latitude: float
            - longitude: float
            - timestamp: str (ISO format)
            - temperature_c: float
            - humidity_percent: float
            - rainfall_mm: float (precipitation in past hour/current)
            - weather_description: str
            - disclaimer: str
            - error: Optional[str]
        """
        city_clean = city_name.strip()
        if not city_clean:
            return {"success": False, "error": "City or region name cannot be empty."}

        # Check if user provided OpenWeatherMap API key in env
        owm_api_key = os.environ.get("OPENWEATHER_API_KEY")
        if owm_api_key:
            return cls._fetch_openweathermap(city_clean, owm_api_key)

        # Default: Use Open-Meteo API
        return cls._fetch_open_meteo(city_clean)

    @classmethod
    def _fetch_open_meteo(cls, city: str) -> Dict[str, Any]:
        """Fetches weather from Open-Meteo with geocoding."""
        try:
            # Step 1: Geocoding
            geo_resp = requests.get(
                cls.OPEN_METEO_GEO_URL,
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                timeout=cls.REQUEST_TIMEOUT
            )
            if geo_resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Geocoding service returned HTTP {geo_resp.status_code}."
                }

            geo_data = geo_resp.json()
            results = geo_data.get("results")
            if not results or len(results) == 0:
                return {
                    "success": False,
                    "error": f"Location '{city}' could not be found. Please check the spelling or try a nearby major city."
                }

            loc = results[0]
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            resolved_name = loc.get("name", city)
            country = loc.get("country", "")
            admin1 = loc.get("admin1", "")
            location_label = f"{resolved_name}, {admin1}, {country}".replace(", ,", ",").strip(", ")

            # Step 2: Fetch current meteorological conditions
            weather_resp = requests.get(
                cls.OPEN_METEO_FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
                    "timezone": "auto"
                },
                timeout=cls.REQUEST_TIMEOUT
            )
            if weather_resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Weather data service returned HTTP {weather_resp.status_code}."
                }

            w_data = weather_resp.json()
            current = w_data.get("current", {})

            temp = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            rain = current.get("precipitation", 0.0)
            time_str = current.get("time", datetime.now(timezone.utc).isoformat())

            # Convert to sensible rainfall representation (if 0.0 current, provide safe indicator)
            return {
                "success": True,
                "city": location_label,
                "latitude": lat,
                "longitude": lon,
                "timestamp": time_str,
                "temperature_c": round(float(temp), 1) if temp is not None else None,
                "humidity_percent": round(float(humidity), 1) if humidity is not None else None,
                "rainfall_mm": round(float(rain), 1) if rain is not None else 0.0,
                "weather_code": current.get("weather_code"),
                "source": "Open-Meteo High-Resolution Numerical Weather Model",
                "disclaimer": (
                    "Agronomic Notice: Live telemetry represents real-time atmospheric conditions. "
                    "Agricultural crops require seasonal averages (e.g. cumulative rainfall over months). "
                    "You may auto-fill these values into the form as current reference points."
                )
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Weather lookup timed out. Please verify your internet connection."
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Network error during weather retrieval: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error while processing weather data: {str(e)}"
            }

    @classmethod
    def _fetch_openweathermap(cls, city: str, api_key: str) -> Dict[str, Any]:
        """Fetches weather from OpenWeatherMap API when API key is provided."""
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            resp = requests.get(
                url,
                params={"q": city, "appid": api_key, "units": "metric"},
                timeout=cls.REQUEST_TIMEOUT
            )
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"OpenWeatherMap API error: HTTP {resp.status_code} ({resp.text})"
                }
            d = resp.json()
            main = d.get("main", {})
            rain_dict = d.get("rain", {})
            rain_1h = rain_dict.get("1h", 0.0)

            return {
                "success": True,
                "city": f"{d.get('name')}, {d.get('sys', {}).get('country')}",
                "latitude": d.get("coord", {}).get("lat"),
                "longitude": d.get("coord", {}).get("lon"),
                "timestamp": datetime.fromtimestamp(d.get("dt", 0), timezone.utc).isoformat(),
                "temperature_c": round(float(main.get("temp")), 1),
                "humidity_percent": round(float(main.get("humidity")), 1),
                "rainfall_mm": round(float(rain_1h), 1),
                "source": "OpenWeatherMap API v2.5",
                "disclaimer": (
                    "Agronomic Notice: Live telemetry represents real-time atmospheric conditions. "
                    "Agricultural crops require seasonal averages. Use with caution for long-term planning."
                )
            }
        except Exception as e:
            return {"success": False, "error": f"OpenWeatherMap request failed: {str(e)}"}
