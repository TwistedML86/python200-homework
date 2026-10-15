from dataclasses import dataclass

from .schemas import WeatherResponse


@dataclass
class HourlyReading:
    """Stores one hour's timestamp, temperature in Celsius, and precipitation in mm."""

    timestamp: str
    temperature_c: float
    precipitation_mm: float

def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Converts the hourly lists into individual readings.

    Args:
        response: Weather data already checked by WeatherResponse.

    Returns:
        A list of hourly readings in the original order.
    """
    readings = []

    for i in range(len(response.hourly.time)):
        reading = HourlyReading(
            timestamp=response.hourly.time[i],
            temperature_c=response.hourly.temperature_2m[i],
            precipitation_mm=response.hourly.precipitation[i],
        )
        readings.append(reading)

    return readings


# WeatherResponse checks outside data when it enters the program.
# HourlyReading is a dataclass because this data has already been checked.