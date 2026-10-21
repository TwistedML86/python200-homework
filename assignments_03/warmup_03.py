import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import(
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay, 
    classification_report,
)

import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y

)

# ----
# Q1
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled training means:", X_train_scaled.mean(axis=0))

# I fit the scaler only on training data to avoid leaking test information into training.
# Q2
encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(df[["season"]])

print("Encoded season shape:", season_encoded.shape)
print("Season columns:", encoder.get_feature_names_out())

# The season column became four columns because the dataset has four seasons.
# Each row has a 1 for its season and 0 for the other seasons.

# Q3
feature_cols = numeric + ["season"]
X_with_season = df[feature_cols]

X_train_season, X_test_season, y_train_season, y_test_season = train_test_split(
    X_with_season, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("numbers", StandardScaler(), numeric),
    ("seasons", OneHotEncoder(handle_unknown="ignore"), ["season"]),
])

season_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

season_pipeline.fit(X_train_season, y_train_season)

print("Q3 test accuracy:", season_pipeline.score(X_test_season, y_test_season))

# The pipeline automatically scales numeric features and encodes season
# before passing the prepared data to the classifier.
# It reuses the training preprocessing settings when predicting on test data.


# --- KNN ---
# Q1
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5)),
])

knn_pipeline.fit(X_train, y_train)
knn_predictions = knn_pipeline.predict(X_test)

print("KNN test accuracy:", accuracy_score(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions))

# Q2
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=k)),
    ])

    scores = cross_val_score(
        candidate, X_train, y_train, cv=5, scoring="accuracy"
    )

    print("k:", k, "Mean CV accuracy:", scores.mean())

# I would choose k=11 because it had the highest mean 5-fold CV accuracy,
# about 94.17%, among the values tested on the training data.

# Q3
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
raw_predictions = knn_raw.predict(X_test)

print("Unscaled KNN accuracy:", accuracy_score(y_test, raw_predictions))
print("Scaled KNN accuracy:", accuracy_score(y_test, knn_predictions))

# Scaling improved test accuracy from 92.5% to about 95.83%.
# KNN uses distances, so scaling stops features with larger numerical ranges
# from dominating the distance calculation.

# --- Logistic Regression ---
# Q1
logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

logreg_pipeline.fit(X_train, y_train)
logreg_predictions = logreg_pipeline.predict(X_test)

print("Logistic accuracy:", accuracy_score(y_test, logreg_predictions))
print("Logistic precision:", precision_score(y_test, logreg_predictions))
print("Logistic recall:", recall_score(y_test, logreg_predictions))
print("Logistic F1:", f1_score(y_test, logreg_predictions))

# Q2
coefficients = logreg_pipeline.named_steps["classifier"].coef_[0]

for name, coef in zip(numeric, coefficients):
    print(f"{name}: {coef:+.3f}")


# Wind and precipitation push most strongly toward skip.
# This makes sense to me because strong wind and heavy rain can make running harder.

# Q3
good_probabilities = logreg_pipeline.predict_proba(X_test)[:, 1]

print("Probability of good for the first five test days:")
print(good_probabilities[:5].round(2))

# A probability near 0.5 means the model is uncertain between good and skip.

# --- Evaluation ---
# Q1
cm = confusion_matrix(y_test, logreg_predictions)

print("Logistic confusion matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"],
)
display.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.close()

# The model made 9 false positives and 7 false negatives.
# I care more about false negatives because I would miss a good running day.
# I usually run in rain or sunshine, unless it is pouring.
# For my preference, recall for the good class matters more.

# --- joblib ---
# Q1
joblib.dump(logreg_pipeline, "models/warmup_model.pkl")

loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == logreg_pipeline.predict(X_test)).all()

print("Predictions match.")

# Saving the whole pipeline keeps the scaler and classifier together.
# Saving only the classifier would leave raw inputs unscaled,
# so predictions could be wrong.