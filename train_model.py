"""Train/evaluate CropSense and optionally compare classifiers with CV."""

from __future__ import annotations

import argparse
from pathlib import Path

from model import DatasetError, compare_models_cv, find_dataset, train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to Crop_recommendation.csv (defaults to ./data/Crop_recommendation.csv).",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Compare KNN, Decision Tree, Logistic Regression, Gaussian Naive Bayes, SVM, and Random Forest.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified folds for model comparison (default: 5).",
    )
    args = parser.parse_args()
    dataset = args.dataset or find_dataset(Path(__file__).resolve().parent)
    if dataset is None:
        parser.error(
            "dataset not found. Put Crop_recommendation.csv in ./data or pass --dataset PATH. "
            "See README.md for the Kaggle download steps."
        )
    try:
        bundle = train_and_evaluate(dataset)
    except DatasetError as exc:
        parser.error(str(exc))

    print(f"Dataset: {bundle.dataset.source.resolve()}")
    print(f"Usable rows: {len(bundle.dataset.frame):,}")
    print(f"Rows dropped: {bundle.dataset.rows_dropped:,}")
    print(f"Duplicate rows removed: {bundle.dataset.duplicate_rows_removed:,}")
    print(f"Missing/invalid feature values imputed: {bundle.dataset.numeric_values_imputed:,}")
    print(f"Train/test rows: {bundle.train_size:,}/{bundle.test_size:,}")
    print(f"Held-out accuracy: {bundle.accuracy:.4f}")
    print("Confusion matrix (rows=true, columns=predicted; labels follow sklearn class order):")
    print(bundle.confusion)
    print("Classes:", ", ".join(map(str, bundle.classes)))

    if args.cross_validate:
        try:
            results = compare_models_cv(dataset, n_splits=args.cv_folds)
        except DatasetError as exc:
            parser.error(str(exc))
        print(f"\nStratified {args.cv_folds}-fold cross-validation comparison")
        print("CV runs on the 80% training split; the stratified 20% test split stays outside CV.")
        print("Imputation/scaling are fit per fold. Ranking uses mean CV accuracy, then mean macro-F1.")
        print(f"{'Model':<24} {'CV accuracy mean +/- SD':<26} {'CV macro-F1 mean +/- SD':<26} {'Test acc':<10} {'Test F1':<10} Fold accuracy")
        for row in results.itertuples(index=False):
            accuracy = f"{row.accuracy_mean:.4f} +/- {row.accuracy_std:.4f}"
            macro_f1 = f"{row.macro_f1_mean:.4f} +/- {row.macro_f1_std:.4f}"
            fold_accuracy = ", ".join(f"{score:.4f}" for score in row.accuracy_folds)
            print(f"{row.model:<24} {accuracy:<26} {macro_f1:<26} {row.holdout_accuracy:<10.4f} {row.holdout_macro_f1:<10.4f} {fold_accuracy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
