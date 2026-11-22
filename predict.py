"""Предсказание вида ириса по четырём измерениям цветка."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "iris_model.joblib"
FEATURE_NAMES = (
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
)


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict(model, features: list[float]) -> dict:
    row = pd.DataFrame([features], columns=FEATURE_NAMES)
    label = model.predict(row)[0]
    probabilities = {
        cls: float(prob)
        for cls, prob in zip(model.classes_, model.predict_proba(row)[0])
    }
    return {"class": label, "probabilities": probabilities}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Предсказание вида ириса по длине и ширине чашелистика и лепестка."
    )
    parser.add_argument(
        "features",
        nargs=4,
        type=float,
        metavar=FEATURE_NAMES,
        help="SepalLengthCm SepalWidthCm PetalLengthCm PetalWidthCm (см)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Путь к файлу модели (.joblib)",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    result = predict(model, args.features)

    print(result["class"])
    for name, prob in result["probabilities"].items():
        print(f"  {name}: {prob:.4f}")


if __name__ == "__main__":
    main()
