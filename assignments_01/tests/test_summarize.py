import pytest

from weatherkit import HourlyReading, DailyAggregator


@pytest.fixture
def sample_readings():
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.1),
        HourlyReading("2026-04-08T01:00", 16.0, 0.2),
        HourlyReading("2026-04-08T02:00", 8.0, 0.3),
        HourlyReading("2026-04-09T00:00", 12.0, 0.0),
        HourlyReading("2026-04-09T01:00", 20.0, 0.5),
    ]
def test_groups_by_date(sample_readings):
    aggregator = DailyAggregator(min_hours=2)
    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[0].date == "2026-04-08"
    assert summaries[1].date == "2026-04-09"
    assert summaries[0].hours_observed == 3
    assert summaries[1].hours_observed == 2

@pytest.mark.parametrize(
    "day_index, expected_max, expected_min",
    [
        (0, 16.0, 8.0),
        (1, 20.0, 12.0),
    ],
)
def test_max_and_min(sample_readings, day_index, expected_max, expected_min):
    aggregator = DailyAggregator(min_hours=2)
    summaries = aggregator.summarize(sample_readings)

    assert summaries[day_index].temp_max == expected_max
    assert summaries[day_index].temp_min == expected_min

def test_precipitation_sum(sample_readings):
    aggregator = DailyAggregator(min_hours=2)
    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(0.6)
    assert summaries[1].precipitation_sum == pytest.approx(0.5)

def test_incomplete_day_is_dropped(sample_readings):
    aggregator = DailyAggregator(min_hours=3)
    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 1
    assert summaries[0].date == "2026-04-08"
    assert aggregator.incomplete_days(sample_readings) == ["2026-04-09"]

def test_lower_min_hours_keeps_day(sample_readings):
    aggregator = DailyAggregator(min_hours=2)
    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[1].date == "2026-04-09"
    assert aggregator.incomplete_days(sample_readings) ==[]


# I changed temp_max to use min() instead of max().
# Both cases in test_max_and_min failed and caught the mistake.
# Then I changed it back to max().