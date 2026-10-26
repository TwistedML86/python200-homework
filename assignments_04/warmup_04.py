import json
from dataclasses import dataclass
import pytest

#------------------------------------------
# Q1: Load model information a JSON file

class ModelInfo:
    def __init__(self, path):
        with open(path) as file:
            self.data = json.load(file)
    def feature_count(self) -> int:
        return len(self.data["features"])
example_data = {
    "features": ["temperature", "rain", "wind"]
}

with open("model_info_example.json", "w") as file:
    json.dump(example_data, file)

info = ModelInfo("model_info_example.json")
print("Feature count:", info.feature_count())

# Loading the file in __init__ keeps the data ready to use.
# Calling feature_count() again does not need to reopen the file.


# Q2: A dataclass for a day's forecast
@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        if self.will_rain:
            rain_message = "rain expected"
        else:
            rain_message = "no rain expected"

        return f"{self.date}: high {self.high_c}C, {rain_message}"
day1 = DayForecast("2026-10-23", 24.1, True)
day2 = DayForecast("2026-10-24", 22.5, False)

print(day1.summary())
print(day2.summary())

# @dataclass created the __init__ method for me,
# so I did not have to write how the three fields are assigned.

# Q3: Convert temperature text into a float
def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError("Invalid temperature: please enter a number.")
def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)

def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")
    
# Without match, the test would accept any ValueError,
# even if its message did not mention temperature.
