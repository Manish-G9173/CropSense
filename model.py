"""Dataset loading, cleaning, training, and evaluation for CropSense."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
DISPLAY_NAMES = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
}


class DatasetError(ValueError):
    """Raised when the supplied dataset cannot support model training."""


@dataclass
class CleanedDataset:
    frame: pd.DataFrame
    source: Path
    rows_loaded: int
    rows_dropped: int
    duplicate_rows_removed: int
    numeric_values_imputed: int


@dataclass
class ModelBundle:
    model: Pipeline
    dataset: CleanedDataset
    accuracy: float
    confusion: np.ndarray
    classes: np.ndarray
    feature_importance: pd.DataFrame
    test_size: int
    train_size: int


def find_dataset(project_dir: Path | None = None) -> Path | None:
    """Find the Kaggle CSV in the documented locations."""
    root = project_dir or Path(__file__).resolve().parent
    candidates = (
        root / "data" / "Crop_recommendation.csv",
        root / "data" / "crop_recommendation.csv",
        root / "Crop_recommendation.csv",
        root / "crop_recommendation.csv",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _normalize_headers(frame: pd.DataFrame) -> pd.DataFrame:
    normalized: dict[Any, str] = {}
    for original in frame.columns:
        key = str(original).strip().casefold().replace(" ", "").replace("_", "")
        aliases = {
            "n": "N", "nitrogen": "N",
            "p": "P", "phosphorus": "P",
            "k": "K", "potassium": "K",
            "temperature": "temperature", "temp": "temperature",
            "humidity": "humidity", "ph": "ph", "soilph": "ph",
            "rainfall": "rainfall", "label": "label", "crop": "label",
        }
        normalized[original] = aliases.get(key, str(original).strip())
    renamed = frame.rename(columns=normalized)
    if renamed.columns.duplicated().any():
        duplicates = renamed.columns[renamed.columns.duplicated()].tolist()
        raise DatasetError(f"The CSV has duplicate columns after normalization: {duplicates}.")
    return renamed


def load_and_clean_dataset(path: Path) -> CleanedDataset:
    """Load the dataset, coerce invalid numeric entries, and validate rows.

    Missing feature values are kept for median imputation inside the training
    pipeline. Invalid values outside broad physical ranges are marked missing
    and handled by that same training-only imputer. Rows without a target or
    with no usable feature values are removed.
    """
    if not path.is_file():
        raise DatasetError(f"Dataset not found: {path}")
    try:
        raw = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetError(f"Could not read the CSV: {exc}") from exc
    if raw.empty:
        raise DatasetError("The dataset CSV is empty.")

    rows_loaded = len(raw)
    frame = _normalize_headers(raw)
    missing = [column for column in [*FEATURES, TARGET] if column not in frame.columns]
    if missing:
        raise DatasetError(
            "The dataset is missing required columns: " + ", ".join(missing)
            + ". Expected N, P, K, temperature, humidity, ph, rainfall, and label."
        )

    frame = frame[[*FEATURES, TARGET]].copy()
    frame[TARGET] = frame[TARGET].astype("string").str.strip()
    frame[TARGET] = frame[TARGET].replace("", pd.NA)
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)

    # Broad validity bounds: values outside these ranges are not plausible
    # measurements for the named inputs and should not shape model training.
    bounds = {
        "N": (0, 300), "P": (0, 300), "K": (0, 300),
        "temperature": (-20, 60), "humidity": (0, 100),
        "ph": (0, 14), "rainfall": (0, 5000),
    }
    for column, (lower, upper) in bounds.items():
        invalid = frame[column].notna() & ~frame[column].between(lower, upper)
        frame.loc[invalid, column] = np.nan

    frame = frame.dropna(subset=[TARGET])
    frame = frame.dropna(subset=FEATURES, how="all")
    duplicate_rows_removed = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    numeric_values_imputed = int(frame[FEATURES].isna().sum().sum())
    rows_dropped = rows_loaded - len(frame)

    if len(frame) < 10:
        raise DatasetError(
            f"Only {len(frame)} usable rows remain after cleaning; at least 10 are required."
        )
    if frame[TARGET].nunique() < 2:
        raise DatasetError("At least two distinct crop labels are required for classification.")

    return CleanedDataset(
        frame=frame,
        source=path,
        rows_loaded=rows_loaded,
        rows_dropped=rows_dropped,
        duplicate_rows_removed=duplicate_rows_removed,
        numeric_values_imputed=numeric_values_imputed,
    )


def train_and_evaluate(path: Path, random_state: int = 42) -> ModelBundle:
    """Train a Random Forest with a reproducible held-out evaluation."""
    dataset = load_and_clean_dataset(path)
    X = dataset.frame[FEATURES]
    y = dataset.frame[TARGET].astype(str)
    empty_features = [feature for feature in FEATURES if X[feature].notna().sum() == 0]
    if empty_features:
        raise DatasetError(
            "No valid numeric values remain for: " + ", ".join(empty_features)
            + ". Check the CSV measurements before training."
        )
    counts = y.value_counts()
    test_count = int(np.ceil(len(y) * 0.2))
    train_count = len(y) - test_count
    can_stratify = (
        counts.min() >= 2
        and test_count >= len(counts)
        and train_count >= len(counts)
    )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y if can_stratify else None,
        )
    except ValueError as exc:
        raise DatasetError(f"Could not create a valid train/test split: {exc}") from exc
    if y_train.nunique() < 2:
        raise DatasetError("The training split contains fewer than two crop classes.")
    empty_train_features = [feature for feature in FEATURES if X_train[feature].notna().sum() == 0]
    if empty_train_features:
        raise DatasetError(
            "The training split has no valid measurements for: "
            + ", ".join(empty_train_features) + "."
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])
    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    except (ValueError, TypeError) as exc:
        raise DatasetError(f"Model training failed: {exc}") from exc

    classes = model.named_steps["classifier"].classes_
    matrix = confusion_matrix(y_test, predictions, labels=classes)
    importance = pd.DataFrame({
        "feature": [DISPLAY_NAMES[column] for column in FEATURES],
        "importance": model.named_steps["classifier"].feature_importances_,
    }).sort_values("importance", ascending=True)

    return ModelBundle(
        model=model,
        dataset=dataset,
        accuracy=float(accuracy_score(y_test, predictions)),
        confusion=matrix,
        classes=classes,
        feature_importance=importance,
        test_size=len(y_test),
        train_size=len(y_train),
    )


def recommend(model: Pipeline, values: dict[str, float], top_n: int = 3) -> list[tuple[str, float]]:
    """Map each probability to its corresponding fitted class, then rank."""
    row = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
    probabilities = model.predict_proba(row)[0]
    classes = model.named_steps["classifier"].classes_
    ranking = np.argsort(probabilities)[::-1][: min(top_n, len(classes))]
    return [(str(classes[index]), float(probabilities[index])) for index in ranking]


def compare_models_cv(
    path: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare candidate classifiers using leakage-safe stratified CV.

    First reserve a stratified 20% final test split. CV runs only on the
    remaining training rows. Imputation and scaling are inside each estimator
    pipeline so they are fit independently on each fold. The returned table
    includes CV mean/std and a single untouched holdout score per classifier.
    """
    if n_splits < 2:
        raise DatasetError("Cross-validation requires at least 2 folds.")
    dataset = load_and_clean_dataset(path)
    X = dataset.frame[FEATURES]
    y = dataset.frame[TARGET].astype(str)
    counts = y.value_counts()
    test_count = int(np.ceil(len(y) * 0.2))
    train_count = len(y) - test_count
    can_stratify = (
        counts.min() >= 2
        and test_count >= len(counts)
        and train_count >= len(counts)
    )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y if can_stratify else None,
        )
    except ValueError as exc:
        raise DatasetError(f"Could not create a model-comparison split: {exc}") from exc

    smallest_class = int(y_train.value_counts().min())
    if smallest_class < n_splits:
        raise DatasetError(
            f"Cannot run {n_splits}-fold stratified CV on the training split: the "
            f"smallest crop class has only {smallest_class} rows. Reduce the fold count."
        )

    def make_pipeline(classifier, scale: bool = False) -> Pipeline:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", classifier))
        return Pipeline(steps)

    candidates = {
        "K-Nearest Neighbors": make_pipeline(
            KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1), scale=True
        ),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
        ),
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
            scale=True,
        ),
        "Gaussian Naive Bayes": make_pipeline(GaussianNB()),
        "SVM (RBF)": make_pipeline(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"), scale=True
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
        ),
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []
    for name, estimator in candidates.items():
        scores = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        accuracy = scores["test_accuracy"]
        macro_f1 = scores["test_macro_f1"]
        final_model = clone(estimator).fit(X_train, y_train)
        holdout_predictions = final_model.predict(X_test)
        rows.append({
            "model": name,
            "accuracy_mean": float(accuracy.mean()),
            "accuracy_std": float(accuracy.std(ddof=1)),
            "accuracy_folds": tuple(float(score) for score in accuracy),
            "macro_f1_mean": float(macro_f1.mean()),
            "macro_f1_std": float(macro_f1.std(ddof=1)),
            "macro_f1_folds": tuple(float(score) for score in macro_f1),
            "holdout_accuracy": float(accuracy_score(y_test, holdout_predictions)),
            "holdout_macro_f1": float(f1_score(y_test, holdout_predictions, average="macro")),
            "folds": n_splits,
        })
    return pd.DataFrame(rows).sort_values(
        ["accuracy_mean", "macro_f1_mean"], ascending=False
    ).reset_index(drop=True)
