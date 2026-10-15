from weatherkit import HourlyBlock, WeatherResponse, HourlyReading, to_readings


def test_readings_count_and_order():
    response = WeatherResponse(
        latitude=35.2,
        longitude=-80.8,
        timezone="GMT",
        elevation=254.0,
        hourly=HourlyBlock(
            time=[
                "2026-04-08T00:00",
                "2026-04-08T01:00",
                "2026-04-08T02:00",
            ],
            temperature_2m=[16.8, 15.0, 14.2],
            precipitation=[0.0, 0.5, 1.0],
        ),
    )

    readings = to_readings(response)

    assert len(readings) == 3
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T02:00"

def test_reading_values_match():
    response = WeatherResponse(
        latitude=35.2,
        longitude=-80.8,
        timezone="GMT",
        elevation=254.0,
        hourly=HourlyBlock(
            time=["2026-04-08T00:00", "2026-04-08T01:00"],
            temperature_2m=[16.8, 15.0],
            precipitation=[0.0, 0.5],
        ),
    )

    readings = to_readings(response)

    for i in range(len(readings)):
        assert readings[i].timestamp == response.hourly.time[i]
        assert readings[i].temperature_c == response.hourly.temperature_2m[i]
        assert readings[i].precipitation_mm == response.hourly.precipitation[i]

def test_identical_readings_are_equal():
    reading_a = HourlyReading("2026-04-08T00:00", 16.8, 0.0)
    reading_b = HourlyReading("2026-04-08T00:00", 16.8, 0.0)

    assert reading_a == reading_b

