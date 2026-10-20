# Task 1: Fetch the data
# City: Los Angeles

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# -------------------------------------------

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


# Task 2: Look at the features
print(df.describe())

print("Missing values:")
print(df.isna().sum())

# There are no missing values, so I kept all 365 days.

correlations = df.corr(numeric_only=True)["temperature_2m_max"]
print("Correlations with daily high:")
print(correlations.sort_values())

# The daily low has the strongest correlation with the daily high, about 0.76.

plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"])
plt.title("Los Angeles Daily High vs Low")
plt.xlabel("Daily low temperature (C)")
plt.ylabel("Daily high temperature (C)")
plt.savefig("outputs/la_high_vs_low.png")
plt.close()

# Higher daily lows generally go with higher daily highs.
# One hot day has a high around 40.5 C and a low near 25 C.

# Task 3: Engineer a feature
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

print("Summer and non-summer days:")
print(df["is_summer"].value_counts())

# Summer might help, but the daily low may already capture some seasonal changes.
# I will compare the models to see whether the extra features help.

# Task 4: Baseline model
# Use the daily low to predict the daily high.
X = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline = LinearRegression()
baseline.fit(X_train, y_train)

y_pred = baseline.predict(X_test)

baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
baseline_r2 = baseline.score(X_test, y_test)

print("Baseline slope:", baseline.coef_[0])
print("Baseline RMSE:", baseline_rmse)
print("Baseline test R2:", baseline_r2)

# For each 1 C increase in the daily low, the predicted high rises about 0.97 C.
# The baseline has a test RMSE of about 3.20 C.
# It explains about 66% of the variation in the test daily highs.

# Task 5: Full model
# Use the daily low, rain, wind, and summer to predict the daily high.
feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer",
]

X_full = df[feature_cols].values

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

y_pred_full = model_full.predict(X_test_full)

full_rmse = np.sqrt(mean_squared_error(y_test_full, y_pred_full))
train_r2_full = model_full.score(X_train_full, y_train_full)
test_r2_full = model_full.score(X_test_full, y_test_full)

print("Full model training R2:", train_r2_full)
print("Full model test R2:", test_r2_full)
print("Full model test RMSE:", full_rmse)

print("Feature coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# The extra features helped: test RMSE dropped from about 3.20 to 2.96 C.
# Test R2 increased from about 0.66 to 0.71.
# Higher daily lows and summer raise the predicted high.
# More rain and stronger wind lower it, with the other features held fixed.
# Training and test R2 are fairly close, with no obvious sign of overfitting.

# Task 6: Compare predictions with actual highs
plt.scatter(y_pred_full, y_test_full)
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted high temperature (C)")
plt.ylabel("Actual high temperature (C)")

# Use the same limits for the perfect-prediction line.
lowest = min(y_pred_full.min(), y_test_full.min())
highest = max(y_pred_full.max(), y_test_full.max())

plt.plot([lowest, highest], [lowest, highest], "k--")
plt.savefig("outputs/predicted_vs_actual_high.png")
plt.savefig("outputs/la_predicted_vs_actual_high.png")
plt.close()

# Most points follow the diagonal, but some predictions are several degrees off.
# Points above the line are underpredictions; points below are overpredictions.
# Most of the hottest actual days are above the diagonal.
# The model tends to underestimate those highs.
# Some cooler days are below the line, meaning their highs were overestimated.


# ------------------------------
# Project summary
# I used 365 days of Los Angeles weather, from October 2025 through September 2026.
# The split used 292 days for training and 73 days for testing.
# The full model's test RMSE was about 2.96 C, compared with 3.20 C for the baseline.
# Its test R2 was about 0.71, explaining about 71% of the variation in test highs.
# A 1 C increase in the daily low raises the predicted high by about 0.99 C,
# with the other features held fixed.
# I was surprised that wind had a positive correlation with the daily high,
# but a negative coefficient when the other features were included.
# The features use different units, so coefficient size alone does not tell
# me which feature is most important.