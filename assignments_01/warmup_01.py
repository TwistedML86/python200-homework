# --- Classes ---
# Q1
class Thermometer:
    
    def __init__(self, location, readings=None):
        self.location = location
        if readings is None:
            self.readings = []
        else: 
            self.readings = readings

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if len(self.readings) == 0:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if len(self.readings) == 0:
            return None
        return max(self.readings)
    def __repr__(self):
        return (
            f"Thermometer(location={self.location!r}, "
            f"n_readings={len(self.readings)}, "
            f"average={self.average()})"
        )
    
my_thermometer = Thermometer("Los Angeles")
my_thermometer.add(38.3)
my_thermometer.add(37.3)
my_thermometer.add(41.1)
my_thermometer.add(41)

print("Average:", my_thermometer.average())
print("Hottest:", my_thermometer.hottest())


# average() returns None if there are no readings.
# Without this check, it would divide by 0 and give a ZeroDivisionError.

# Q2
print(my_thermometer)
second_thermometer = Thermometer("Manila", [25.0,23.0])
print([my_thermometer, second_thermometer])

# Without __repr__, Python shows the class name and a memory address.
# That doesn't show the location or readings, so debugging is harder.

# Q3
class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold
    def breaches(self, thermometer):
        above_threshold = []
        for reading in thermometer.readings:
            if reading > self.threshold:
                above_threshold.append(reading)
        return above_threshold
alert_a = TemperatureAlert(38.0)
alert_b = TemperatureAlert(40.0)

print("Above 38:", alert_a.breaches(my_thermometer))
print("Above 40:", alert_b.breaches(my_thermometer))

# The threshold is stored in TemperatureAlert so the alert remembers it.
# I can check 20 thermometers without passing the threshold each time.

# - - - Dataclasses, Type Hints, and Docstrings - - - 

#Q1
from dataclasses import dataclass, FrozenInstanceError, field

@dataclass(frozen=True)
class Station:
    """Stores information about weather station"""

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float

station_a = Station("LA1", "Los Angeles", 34.0, -118.0, 90.0)
station_b = Station("LA1", "Los Angeles", 34.0, -118.0, 90.0)

print(station_a == station_b)

# Dataclasses compare the field values, so these stations are equal.
# The original class would give False because they are separate objects.

#Q2
try: 
    station_a.name = "Manila"
except FrozenInstanceError as error:
    print(error)

station_c = Station("Man", "Manila", 14.6, 121.0, 10.0)
stations = {station_a, station_b, station_c}
print("Stations in set:", len(stations))

# frozen=True also makes these stations hashable.
# This lets me use a set, where identical stations count once.

# Q3
# Using stations: list[Station] = [] gave this error:
# ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory
# Python refuses it because batches could share the same list.
# default_factory=list gives each batch its own empty list.
@dataclass
class StationBatch:
    """Stores a group of weather stations in a region."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Adds one station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Returns the highest station, or None if empty."""
        if len(self.stations) == 0:
            return None

        highest_station = self.stations[0]
        for station in self.stations:
            if station.elevation > highest_station.elevation:
                highest_station = station
        return highest_station
    
batch = StationBatch("Practice region")
print("Empty batch:", batch.highest())

batch.add(station_a)
batch.add(station_c)
print("Highest station:", batch.highest())

# --- Pydantic ---
# Q1
from pydantic import BaseModel, Field, ValidationError, model_validator



class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def check_sensor(self):
            if self.humidity == 0.0 and self.temperature_c < -40:
                raise ValueError("Possible failed sensor")
            return self

reading = Reading(
    station_id="LA1",
    timestamp="2025-10-12T12:00",
    temperature_c=21.5,
    humidity=40.0
)
print(reading)

# Q2
# Missing timestamp
try:
    Reading(station_id="LA1", temperature_c=21.5, humidity=40.0)
except ValidationError as error:
    print(error)

# Temperature above the allowed limit
try:
    Reading(
        station_id="LA1",
        timestamp="2025-10-12T12:00",
        temperature_c=150.0,
        humidity=40.0,

    )
except ValidationError as error:
    print(error)

# Humidity that cannot be converted to a number
try:
    Reading(
        station_id="LA1",
        timestamp="2025-10-12T12:00",
        temperature_c=21.5,
        humidity="very humid",
    )
except ValidationError as error:
    print(error)

converted_reading = Reading(
    station_id="LA1",
    timestamp="2025-10-12T12:00",
    temperature_c="21.5",
    humidity=40,
)
print(converted_reading)
print(type(converted_reading.temperature_c))
print(type(converted_reading.humidity))

# Pydantic can convert "21.5" to a float because it represents a number.
# "very humid" cannot be converted to a number, so it is rejected.

# Q3
# Test summary: 6 passed in 0.63s
try:
    Reading(
        station_id="LA",
        temperature_c="not a number",
        humidity=40.0,
    )
except ValidationError as error:
    for problem in error.errors():
        print(problem["loc"], problem["msg"])

# Three errors were reported
# Seeing all of them at once help me fix them without running the code repeatedly.

   
    
# Q4
print(Reading(
    station_id="LA1",
    timestamp="2025-10-12T12:00",
    temperature_c=21.5,
    humidity=40.0,
))

try:
    Reading(
        station_id="LA1",
        timestamp="2025-10-12T12:00",
        temperature_c=-45.0,
        humidity=0.0,
    )
except ValidationError as error:
    print(error)

# Field constraints check each field separately.
# This rule checks temperature and humidity together.


# --- pytest ---
# Q1
import pytest

def celsius_to_fahrenheit(celsius: float) -> float:
    """Converts Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

# Decimal calculations can have small rounding differences.
# pytest.approx checks that the result is close enough.

# Q2
def mean(values: list[float]) -> float:
    """Returns the mean, or raises an error for an empty list."""
    if len(values) == 0:
        raise ValueError("Cannot find the mean of an empty list")
    return sum(values) / len(values)

def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

# pytest.raises(ValueError) checks the error type.
# match="empty" also checks that the message explains the empty input.

# Q3
@pytest.mark.parametrize("values, expected", [
    ([10.0], 10.0),
    ([2.0, 4.0], 3.0),
    ([-6.0, -2.0], -4.0),
    ([-2.0, 0.0, 2.0], 0.0),

])
def test_mean_values(values, expected):
    assert mean(values) == pytest.approx(expected)
# Parametrize lets me check several cases without repeating the same test.

# Q4
# Failure output when I changed 9 / 5 to 9 / 4:
# E assert 257.0 == 212
# E + where 257.0 = celsius_to_fahrenheit(100)
# Pytest showed my result, 257.0, and the expected result, 212.
# This helps me see what went wrong instead of only saying "assertion failed".
# I changed the function back to 9 / 5.