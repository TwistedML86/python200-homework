# Task 1: Fetch the data
# City: Los Angeles

import requests
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import platform
import sklearn
import joblib
# ---------------------------------------------------------

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 34.05,
    "longitude": -118.24,
    "start_date": "2025-10-01",
    "end_date": "2026-09-30",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Dataset shape:", df.shape)
print(df.head())

# Task 2: Engineer the label
# Use CTD's default running-weather thresholds.
df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

print("Class balance (0 = skip, 1 = good):")
print(df["good_for_running"].value_counts())

print("Fraction of good days:", df["good_for_running"].mean())

# About 42.7% of the days are good under these rules.
# This seems reasonable because LA can have days above the 26 C limit.
# Both classes have plenty of examples, so the labels are not severely imbalanced.

# Task 3: Build and evaluate KNN
feature_cols = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[feature_cols]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_means = []

for k in k_values:
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=k)),
    ])

    scores = cross_val_score(
        candidate, X_train, y_train, cv=5, scoring="accuracy"
    )

    cv_means.append(scores.mean())
    print("k:", k, "Mean CV accuracy:", scores.mean())

best_k = k_values[cv_means.index(max(cv_means))]
print("Chosen k:", best_k)

# k=3 had the highest mean cross-validation accuracy.
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=best_k)),
])

knn_pipeline.fit(X_train, y_train)
knn_predictions = knn_pipeline.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)

print("KNN test accuracy:", knn_accuracy)
print(classification_report(y_test, knn_predictions))

# Task 4: Build and evaluate logistic regression
logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

logreg_pipeline.fit(X_train, y_train)
logreg_predictions = logreg_pipeline.predict(X_test)

logreg_accuracy = accuracy_score(y_test, logreg_predictions)
logreg_precision = precision_score(y_test, logreg_predictions)
logreg_recall = recall_score(y_test, logreg_predictions)
logreg_f1 = f1_score(y_test, logreg_predictions)

print("Logistic accuracy:", logreg_accuracy)
print("Logistic precision:", logreg_precision)
print("Logistic recall:", logreg_recall)
print("Logistic F1:", logreg_f1)
print(classification_report(y_test, logreg_predictions))

print("Logistic feature coefficients:")
coefficients = logreg_pipeline.named_steps["classifier"].coef_[0]

for name, coef in zip(feature_cols, coefficients):
    print(f"{name}: {coef:+.3f}")

# Precipitation and daily high temperature push most strongly toward skip.
# This makes sense because the rules reject rainy days and highs above 26 C.
# These coefficients describe changes with the other features held fixed.

# -----------------------------
# Task 5: Compare and choose
# Logistic regression had higher test accuracy: 97.26% versus 95.89% for KNN.
# Recall matters more to me because I do not want to miss good running days.
# A false negative misses a good day; a false positive recommends a skip day.
# Logistic regression missed one good day, while KNN missed two.
# I choose logistic regression because it had better recall and accuracy here.
# It is compact, predicts quickly, and its coefficients help explain its decisions.
# It also gives probabilities so I can see when a prediction is uncertain.


# Task 6: Save the model and metadata
joblib.dump(logreg_pipeline, "models/weather_classifier.pkl")

metadata = {
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__,
    "feature_names": feature_cols,
    "label_rule": (
        "Good when high is between 7 and 26 C inclusive, low >= 0 C, "
        "precipitation < 3 mm, and maximum wind < 30 km/h."
    ),
    "city": "Los Angeles",
    "test_accuracy": logreg_accuracy,
    "test_f1": logreg_f1,
}

with open("models/weather_classifier_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print("Saved weather classifier and metadata.")


