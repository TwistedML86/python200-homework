from dataclasses import dataclass

from .records import HourlyReading


@dataclass
class DailySummary:
    """Stores one day's temperatures in Celsius and precipitation in mm."""

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Returns the difference between the highest and lowest temperature."""
        return self.temp_max - self.temp_min

class DailyAggregator:
    """Groups hourly readings into daily summaries."""

    def __init__(self, min_hours: int = 24) -> None:
        """Sets the minimum number of observations needed for a day."""
        self.min_hours = min_hours

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Returns summaries for days with enough observations."""
        days: dict[str, list[HourlyReading]] = {}

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in days:
                days[date] = []

            days[date].append(reading)

        summaries: list[DailySummary] = []

        for date in sorted(days):
            day_readings = days[date]

            if len(day_readings) < self.min_hours:
                continue

            temperatures = []
            precipitation_total = 0.0

            for reading in day_readings:
                temperatures.append(reading.temperature_c)
                precipitation_total += reading.precipitation_mm

            summary = DailySummary(
                date=date,
                temp_max=max(temperatures),
                temp_min=min(temperatures),
                precipitation_sum=precipitation_total,
                hours_observed=len(day_readings),
            )
            summaries.append(summary)

        return summaries
    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Returns the dates with too few observations."""
        counts: dict[str, int] = {}

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in counts:
                counts[date] = 0

            counts[date] += 1

        incomplete = []

        for date in sorted(counts):
            if counts[date] < self.min_hours:
                incomplete.append(date)

        return incomplete
    