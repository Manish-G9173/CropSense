"""Train and compare regressors for the crop-yield dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from yield_model import (
    YieldDatasetError,
    find_yield_dataset,
    train_and_compare_yield_models,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to crop_yield.csv or crop_yield.csv.zip (defaults to ./data/crop_yield.csv).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of forward-chaining cross-validation folds (default: 5).",
    )
    args = parser.parse_args()
    dataset = args.dataset or find_yield_dataset(Path(__file__).resolve().parent)
    if dataset is None:
        parser.error("yield dataset not found. Put crop_yield.csv or its ZIP in ./data.")
    try:
        bundle = train_and_compare_yield_models(dataset, n_splits=args.cv_folds)
    except YieldDatasetError as exc:
        parser.error(str(exc))

    print(f"Dataset: {bundle.dataset.source.resolve()}")
    print(f"Rows loaded/usable/dropped: {bundle.dataset.rows_loaded:,}/{len(bundle.dataset.frame):,}/{bundle.dataset.rows_dropped:,}")
    print(f"Feature values imputed inside model pipelines: {bundle.dataset.feature_values_imputed:,}")
    print(f"Train years: {min(bundle.train_years)}–{max(bundle.train_years)} ({bundle.train_size:,} rows)")
    print(f"Future holdout years: {min(bundle.test_years)}–{max(bundle.test_years)} ({bundle.test_size:,} rows)")
    print("Production is excluded from features to prevent target leakage.")
    print(f"Best model by mean forward-CV RMSE: {bundle.best_model_name}")
    columns = ["Model", "CV MAE", "CV MAE SD", "CV RMSE", "CV RMSE SD", "Test MAE", "Test RMSE"]
    print(bundle.comparison[columns].to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
