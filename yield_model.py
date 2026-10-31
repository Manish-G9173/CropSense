"""Leakage-aware yield regression and time-based model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor


CATEGORICAL_FEATURES = ["Crop", "Season", "State"]
NUMERIC_FEATURES = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
FEATURES = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
TARGET = "Yield"
LEAKAGE_COLUMNS = ["Production"]


class YieldDatasetError(ValueError):
    """Raised when the yield dataset cannot support a valid regression run."""


@dataclass
class CleanedYieldDataset:
    frame: pd.DataFrame
    source: Path
    rows_loaded: int
    rows_dropped: int
    duplicate_rows_removed: int
    feature_values_imputed: int


@dataclass
class YieldModelBundle:
    dataset: CleanedYieldDataset
    models: dict[str, Pipeline]
    comparison: pd.DataFrame
    best_model_name: str
    train_size: int
    test_size: int
    train_years: tuple[int, ...]
    test_years: tuple[int, ...]

    @property
    def best_model(self) -> Pipeline:
        return self.models[self.best_model_name]


def find_yield_dataset(project_dir: Path | None = None) -> Path | None:
    """Find the local CSV or the ZIP supplied by the user."""
    root = project_dir or Path(__file__).resolve().parent
    candidates = (
        root / "data" / "crop_yield.csv",
        root / "data" / "crop_yield.csv.zip",
        root / "crop_yield.csv",
        root / "crop_yield.csv.zip",
        Path.home() / "Downloads" / "crop_yield.csv.zip",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def load_and_clean_yield_dataset(path: Path) -> CleanedYieldDataset:
    """Load the yield table, preserve valid zero yields, and clean bad rows."""
    if not path.is_file():
        raise YieldDatasetError(f"Yield dataset not found: {path}")
    try:
        raw = pd.read_csv(path, compression="zip" if path.suffix.casefold() == ".zip" else "infer")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise YieldDatasetError(f"Could not read the yield CSV: {exc}") from exc
    if raw.empty:
        raise YieldDatasetError("The yield CSV is empty.")

    raw.columns = [str(column).strip() for column in raw.columns]
    if raw.columns.duplicated().any():
        raise YieldDatasetError("The CSV contains duplicate column names.")
    missing = [column for column in [*FEATURES, *LEAKAGE_COLUMNS, TARGET] if column not in raw.columns]
    if missing:
        raise YieldDatasetError(
            "Missing required columns: " + ", ".join(missing)
            + ". Expected Crop, Crop_Year, Season, State, Area, Production, "
            "Annual_Rainfall, Fertilizer, Pesticide, and Yield."
        )

    rows_loaded = len(raw)
    frame = raw[[*FEATURES, *LEAKAGE_COLUMNS, TARGET]].copy()
    numeric_columns = [*NUMERIC_FEATURES, *LEAKAGE_COLUMNS, TARGET]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string").str.strip().replace("", pd.NA)

    # Values with no physical meaning are imputed in the model pipeline. Keep
    # zero yield records because a zero harvest can be a real observed outcome.
    for column in ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]:
        frame.loc[frame[column].notna() & (frame[column] < 0), column] = np.nan
    frame.loc[frame["Crop_Year"].notna() & ~frame["Crop_Year"].between(1900, 2100), "Crop_Year"] = np.nan
    frame = frame.dropna(subset=[TARGET])
    frame = frame.loc[frame[TARGET] >= 0].copy()
    frame = frame.dropna(subset=CATEGORICAL_FEATURES, how="all")
    frame = frame.dropna(subset=["Crop_Year"])
    frame = frame.dropna(subset=[*FEATURES], how="all")

    duplicate_rows_removed = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    feature_values_imputed = int(frame[FEATURES].isna().sum().sum())
    rows_dropped = rows_loaded - len(frame)
    if len(frame) < 100:
        raise YieldDatasetError(f"Only {len(frame)} usable rows remain; at least 100 are required.")
    if frame[TARGET].nunique() < 2:
        raise YieldDatasetError("At least two distinct yield values are required.")
    if frame["Crop_Year"].notna().sum() == 0:
        raise YieldDatasetError("Crop_Year has no usable values for a time-based holdout.")

    # Production is deliberately excluded from the returned feature matrix:
    # it is the harvested output, so using it to predict Yield leaks the answer.
    return CleanedYieldDataset(
        frame=frame,
        source=path,
        rows_loaded=rows_loaded,
        rows_dropped=rows_dropped,
        duplicate_rows_removed=duplicate_rows_removed,
        feature_values_imputed=feature_values_imputed,
    )


def _build_pipeline(regressor: Any, scale_numeric: bool) -> Pipeline:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def _candidate_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "Linear Regression": _build_pipeline(LinearRegression(), scale_numeric=True),
        "Ridge Regression": _build_pipeline(Ridge(alpha=1.0), scale_numeric=True),
        "K-Nearest Neighbors": _build_pipeline(
            KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=4), scale_numeric=True
        ),
        "Decision Tree Regressor": _build_pipeline(
            DecisionTreeRegressor(min_samples_leaf=3, random_state=random_state), scale_numeric=False
        ),
        "Random Forest Regressor": _build_pipeline(
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                max_features=1.0,
                n_jobs=4,
                random_state=random_state,
            ),
            scale_numeric=False,
        ),
        "Gradient Boosting Regressor": _build_pipeline(
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                loss="huber",
                random_state=random_state,
            ),
            scale_numeric=False,
        ),
        "Linear SVR": _build_pipeline(
            LinearSVR(C=0.1, epsilon=0.1, dual=True, max_iter=12000, random_state=random_state),
            scale_numeric=True,
        ),
    }


def _mae_nonnegative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, np.maximum(y_pred, 0)))


def _rmse_nonnegative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, np.maximum(y_pred, 0))))


def _year_holdout(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    years = np.sort(frame["Crop_Year"].dropna().unique().astype(int))
    if len(years) < 6:
        raise YieldDatasetError("At least six distinct crop years are required for a time-based evaluation.")
    n_test_years = max(1, int(np.ceil(len(years) * 0.2)))
    test_years = tuple(int(year) for year in years[-n_test_years:])
    train_years = tuple(int(year) for year in years[:-n_test_years])
    train_idx = np.flatnonzero(frame["Crop_Year"].isin(train_years).to_numpy())
    test_idx = np.flatnonzero(frame["Crop_Year"].isin(test_years).to_numpy())
    if not len(train_idx) or not len(test_idx):
        raise YieldDatasetError("Could not create a non-empty time-based train/test split.")
    return train_idx, test_idx, train_years, test_years


def _forward_cv_splits(X_train: pd.DataFrame, n_splits: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    del random_state  # Time-series folds retain their natural chronological order.
    years = np.sort(X_train["Crop_Year"].dropna().unique().astype(int))
    if len(years) < 6:
        raise YieldDatasetError("Not enough distinct training years for forward-chaining CV.")
    folds = min(n_splits, max(2, (len(years) - 4) // 2))
    test_size = max(1, min(2, (len(years) - 3) // folds))
    splitter = TimeSeriesSplit(n_splits=folds, test_size=test_size)
    split_list: list[tuple[np.ndarray, np.ndarray]] = []
    year_array = np.asarray(years)
    for train_year_idx, validation_year_idx in splitter.split(year_array):
        fold_train_years = year_array[train_year_idx]
        fold_validation_years = year_array[validation_year_idx]
        train_rows = np.flatnonzero(X_train["Crop_Year"].isin(fold_train_years).to_numpy())
        validation_rows = np.flatnonzero(X_train["Crop_Year"].isin(fold_validation_years).to_numpy())
        split_list.append((train_rows, validation_rows))
    return split_list


def train_and_compare_yield_models(
    path: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> YieldModelBundle:
    """Compare regression pipelines with forward-chaining CV and a future holdout."""
    if n_splits < 2:
        raise YieldDatasetError("Cross-validation requires at least two folds.")
    dataset = load_and_clean_yield_dataset(path)
    frame = dataset.frame
    X = frame[FEATURES].copy()
    y = frame[TARGET].astype(float)
    train_idx, test_idx, train_years, test_years = _year_holdout(frame)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    cv_splits = _forward_cv_splits(X_train, n_splits, random_state)
    candidates = _candidate_models(random_state)

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, pipeline in candidates.items():
        try:
            cv_scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv_splits,
                scoring={
                    "mae": make_scorer(_mae_nonnegative, greater_is_better=False),
                    "rmse": make_scorer(_rmse_nonnegative, greater_is_better=False),
                },
                n_jobs=1,
                error_score="raise",
            )
            pipeline.fit(X_train, y_train)
            predictions = np.maximum(pipeline.predict(X_test), 0)
        except (ValueError, TypeError, RuntimeError) as exc:
            raise YieldDatasetError(f"{name} training/evaluation failed: {exc}") from exc

        rows.append({
            "Model": name,
            "CV MAE": float(-np.mean(cv_scores["test_mae"])),
            "CV MAE SD": float(np.std(-cv_scores["test_mae"], ddof=1)),
            "CV RMSE": float(-np.mean(cv_scores["test_rmse"])),
            "CV RMSE SD": float(np.std(-cv_scores["test_rmse"], ddof=1)),
            "Test MAE": float(mean_absolute_error(y_test, predictions)),
            "Test RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        })
        # The holdout score above is computed first. Then refit every candidate
        # on all observed years so the deployed predictor uses the full history.
        try:
            pipeline.fit(X, y)
        except (ValueError, TypeError, RuntimeError) as exc:
            raise YieldDatasetError(f"{name} final fit failed: {exc}") from exc
        fitted_models[name] = pipeline

    comparison = pd.DataFrame(rows).sort_values("CV RMSE", ascending=True).reset_index(drop=True)
    best_model_name = str(comparison.iloc[0]["Model"])
    return YieldModelBundle(
        dataset=dataset,
        models=fitted_models,
        comparison=comparison,
        best_model_name=best_model_name,
        train_size=len(train_idx),
        test_size=len(test_idx),
        train_years=train_years,
        test_years=test_years,
    )


def predict_yield(model: Pipeline, values: dict[str, Any]) -> float:
    """Predict yield in the dataset's target units from one user field profile."""
    row = pd.DataFrame([{feature: values[feature] for feature in FEATURES}], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    return max(0.0, prediction)
