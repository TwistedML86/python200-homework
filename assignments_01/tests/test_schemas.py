import json
import pytest
from pydantic import ValidationError
from pathlib import Path

from weatherkit import WeatherResponse


def test_valid_response():
    # This path uses the test file's location.
    # A plain relative path depends on where I run pytest.
    file_path = Path(__file__).parent.parent / "weather_raw.json"

    with file_path.open(encoding="utf-8") as file:
        raw_data = json.load(file)

    response = WeatherResponse.model_validate(raw_data)

    assert len(response.hourly.time) == 168

def test_invalid_latitude():
    raw_data = {
        "latitude": 200.0,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 254.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [16.8],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw_data)

def test_mismatched_lengths():
    raw_data = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 254.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
                "2026-04-08T02:00",
            ],
            "temperature_2m": [16.8, 15.0],
            "precipitation": [0.0, 0.0, 0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw_data)

def test_null_temperature():
    raw_data = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 254.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [None],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw_data)
        