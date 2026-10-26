from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd


# Task 1: Store one prediction
@dataclass
class Prediction:
    """Store the predicted label and probability of a good running day."""

    label: str
    probability: float
    
# Task 2: Load the saved weather model
class WeatherClassifier:
    """Load a weather model once and reuse it for predictions."""

    def __init__(self, model_path: str | Path) -> None:
        """Load the saved pipeline.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Check the model path."
            )

        self.model = joblib.load(model_path)

    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """Predict running conditions for each day.

        Raises:
            ValueError: If a day is missing required features.
        """
        if not days:
            return []

        features = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]

        for day in days:
            missing = [name for name in features if name not in day]
            if missing:
                raise ValueError(f"Missing required features: {', '.join(missing)}")

        frame = pd.DataFrame(days)[features]
        labels = self.model.predict(frame)
        probabilities = self.model.predict_proba(frame)[:, 1]

        results = []
        for label, probability in zip(labels, probabilities):
            results.append(
                Prediction(
                    label="good" if label == 1 else "skip",
                    probability=float(probability),
                )
            )

        return results

