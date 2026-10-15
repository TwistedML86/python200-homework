import json
from pathlib import Path

from weatherkit import WeatherResponse, to_readings, DailyAggregator

def main() -> None:
    """Loads and validates the weather file."""
    file_path = Path(__file__).parent / "weather_raw.json"

    with file_path.open(encoding="utf-8") as file:
        raw_data = json.load(file)

    response = WeatherResponse.model_validate(raw_data)

    print("Latitude:", response.latitude)
    print("Timezone:", response.timezone) 
    print("Hourly observations:", len(response.hourly.time))
    readings = to_readings(response)
    print("Readings created:", len(readings))
    print("First reading:", readings[0])
    print("Last reading:", readings[-1])
    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    print("Daily summaries:", len(summaries))

    print()
    print("Date        Max C   Min C   Rain mm   Range C")

    for summary in summaries:
        print(
            f"{summary.date}  "
            f"{summary.temp_max:6.1f}  "
            f"{summary.temp_min:6.1f}  "
            f"{summary.precipitation_sum:7.1f}  "
            f"{summary.temp_range():7.1f}"
        )

    incomplete = aggregator.incomplete_days(readings)
    if incomplete:
        print("Warning: days with too few readings:", incomplete)

# Without this guard, calling main() here would also run the report
# when someone imports this file to use a helper function.

if __name__ == "__main__":
    main()

# --- Task 7: Reflection ---
# Q1
# I would keep the 167 valid readings for a weather chart and show a gap.
# I would reject the file if a model needed every hourly temperature.
# To allow missing temperatures, I would change the HourlyBlock field
# to temperature_2m: list[float | None].
# I would also update the calculations to handle None values.

# Q2
# At noon, today has only about 12 hourly readings.
# With min_hours=24, today's summary gets dropped.
# The afternoon could be hotter, so the maximum is not for a full day.
# incomplete_days() lists today's date so we know why it was left out.

# Q3
# In Week 10, the pipeline can import DailyAggregator from weatherkit.
# I can reuse it without copying or rewriting the aggregation code.
