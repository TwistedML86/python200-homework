# Task 7: Load the saved model and predict
import json
import joblib
import pandas as pd

model = joblib.load("models/weather_classifier.pkl")

with open("models/weather_classifier_metadata.json") as file:
    metadata = json.load(file)

print("City:", metadata["city"])
print("Features:", metadata["feature_names"])
print("Saved model test accuracy:", metadata["test_accuracy"])

# Each row contains high, low, precipitation, and maximum wind.
days = pd.DataFrame(
    [
        [20, 12, 0, 10],    # Clearly good
        [35, 22, 10, 45],   # Clearly bad
        [26, 15, 2.9, 29],  # Near the rule limits
        [24, 14, 0, 15],    # Another mild day
    ],
    columns=metadata["feature_names"],
)

predictions = model.predict(days)
good_probabilities = model.predict_proba(days)[:, 1]

for i in range(len(days)):
    label = "good" if predictions[i] == 1 else "skip"

    print("Day", i + 1)
    print(days.iloc[i].to_dict())
    print("Prediction:", label)
    print("Probability of good:", round(good_probabilities[i], 2))
    print()

    # Task 8: Reflection
# 1. My borderline day was Day 3, near several of the weather limits.
# The model gave it about 0.06 probability of good, so it strongly favored skip.
# This surprised me because the day meets the original labeling rules.
# A probability of 0.52 would be uncertain. I would show the weather details
# and let the runner decide instead of making a strong recommendation.

# 2. Running this script before training would cause a FileNotFoundError
# because the saved model and metadata would not exist.
# A more helpful message could say: "Run train_classifier.py first."

# 3. For the cloud pipeline, I would replace these four hard-coded example
# days with the daily weather data supplied by the pipeline.

