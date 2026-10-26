import joblib
import pandas as pd
import pytest

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import WeatherClassifier, Prediction


@pytest.fixture
def classifier(tmp_path):
    """Create a small model for testing."""
    features = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    training_days = pd.DataFrame(
        [
            [20, 12, 0, 10],    # Good
            [22, 14, 0, 12],    # Good
            [-2, -8, 12, 50],   # Skip
            [0, -5, 15, 60],    # Skip
        ],
        columns=features,
    )
    labels = [1, 1, 0, 0]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=1)),
    ])
    pipeline.fit(training_days, labels)

    model_path = tmp_path / "test_model.pkl"
    joblib.dump(pipeline, model_path)

    return WeatherClassifier(model_path)

@pytest.mark.parametrize(
    "day, expected_label",
    [
        (
            {
                "temperature_2m_max": 20,
                "temperature_2m_min": 12,
                "precipitation_sum": 0,
                "wind_speed_10m_max": 10,
            },
            "good",
        ),
        (
            {
                "temperature_2m_max": -2,
                "temperature_2m_min": -8,
                "precipitation_sum": 12,
                "wind_speed_10m_max": 50,
            },
            "skip",
        ),
    ],
)
def test_predict_labels(classifier, day, expected_label):
    results = classifier.predict([day])

    assert results[0].label == expected_label

def test_probability_in_range(classifier):
    day = {
        "temperature_2m_max": 21,
        "temperature_2m_min": 13,
        "precipitation_sum": 0,
        "wind_speed_10m_max": 11,
    }

    results = classifier.predict([day])

    assert 0 <= results[0].probability <= 1

def test_three_days_return_predictions_in_order(classifier):
    good_day = {
        "temperature_2m_max": 20,
        "temperature_2m_min": 12,
        "precipitation_sum": 0,
        "wind_speed_10m_max": 10,
    }
    bad_day = {
        "temperature_2m_max": -2,
        "temperature_2m_min": -8,
        "precipitation_sum": 12,
        "wind_speed_10m_max": 50,
    }

    results = classifier.predict([good_day, bad_day, good_day])

    assert len(results) == 3
    assert all(isinstance(result, Prediction) for result in results)
    assert [result.label for result in results] == ["good", "skip", "good"]

def test_empty_input_returns_empty_list(classifier):
    assert classifier.predict([]) == []


def test_missing_feature_raises(classifier):
    day = {
        "temperature_2m_max": 20,
        "temperature_2m_min": 12,
        "precipitation_sum": 0,
    }

    with pytest.raises(ValueError, match="wind_speed_10m_max"):
        classifier.predict([day])


def test_missing_model_file_raises(tmp_path):
    missing_path = tmp_path / "missing_model.pkl"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        WeatherClassifier(missing_path)


# I temporarily made every prediction return "skip".
# The good-day case in test_predict_labels and
# test_three_days_return_predictions_in_order caught the mistake.
# I restored the correct label logic afterward.
