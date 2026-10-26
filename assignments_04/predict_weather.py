from weather_model import WeatherClassifier


def main() -> None:
    """Use the saved model to predict running weather."""
    classifier = WeatherClassifier("models/weather_classifier.pkl")

    days = [
        {
            "temperature_2m_max": 20,
            "temperature_2m_min": 12,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 10,
        },
        {
            "temperature_2m_max": 35,
            "temperature_2m_min": 22,
            "precipitation_sum": 10,
            "wind_speed_10m_max": 45,
        },
        
        {
            "temperature_2m_max": 26,
            "temperature_2m_min": 15,
            "precipitation_sum": 2.9,
            "wind_speed_10m_max": 29,
        },
        {
            "temperature_2m_max": 24,
            "temperature_2m_min": 14,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 15,
        },
    ]

    predictions = classifier.predict(days)

    for number, (day, prediction) in enumerate(
        zip(days, predictions), start=1
    ):
        print(
            f"Day {number}: {day} -> {prediction.label}, "
            f"probability of good: {prediction.probability:.2f}"
        )


# The guard runs main() only when this file is executed directly.
# Without it, calling main() at the top level would also run predictions
# when another script imports this file.
if __name__ == "__main__":
    main()

# Task 6: Reflection

# 1. Keeping the features in a fixed order prevents mixing up inputs.
# For example, 25 could mean temperature or wind speed.
# If those columns were swapped, the model could read the wrong meaning
# and give an incorrect prediction.

# 2. The tests create their own small model so they do not depend on
# my saved model being available or staying the same.
# This makes the tests easier to repeat on another computer.

# 3. My component accepts a list of weather records and returns predictions,
# so a cloud pipeline could import it and process several days at once.
# Before production, I would add checks for inputs outside the training range.
# I noticed that a freezing, rainy, windy day was predicted as good
# with a probability rounded to 1.00. That showed me that a confident
# prediction can still be wrong.

