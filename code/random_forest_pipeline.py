from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a random forest genre classifier on FMA small metadata features."
    )
    parser.add_argument(
        "--subset",
        choices=["small", "medium"],
        default="small",
        help="Which official FMA subset to train on.",
    )
    parser.add_argument(
        "--max-train-per-genre",
        type=int,
        default=None,
        help="Optional cap per genre for the official training split.",
    )
    parser.add_argument(
        "--max-eval-per-genre",
        type=int,
        default=None,
        help="Optional cap per genre for the official validation and test splits.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        nargs="+",
        default=[100, 200, 400],
        help="One or more forest sizes to evaluate; the best validation model is saved.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth. Leave unset to grow fully expanded trees.",
    )
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="Feature subsampling strategy for each split, e.g. sqrt, log2, or a float.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples required at each leaf.",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Minimum samples required to split an internal node.",
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=None,
        help="Limit the number of leaf nodes in each tree.",
    )
    parser.add_argument(
        "--ccp-alpha",
        type=float,
        default=0.0,
        help="Cost-complexity pruning strength applied to each tree.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("code/best_random_forest.pkl"),
        help="Where to save the best validation checkpoint.",
    )
    return parser.parse_args()


def resolve_project_paths() -> tuple[Path, Path, Path]:
    project_candidates = [Path.cwd(), Path.cwd().parent]
    for candidate in project_candidates:
        metadata_dir = candidate / "fma_metadata"
        tracks_path = metadata_dir / "tracks.csv"
        features_path = metadata_dir / "features.csv"
        if tracks_path.exists() and features_path.exists():
            return candidate.resolve(), tracks_path.resolve(), features_path.resolve()

    raise FileNotFoundError(
        "Could not find fma_metadata/tracks.csv and fma_metadata/features.csv "
        "from the current directory or its parent."
    )


def flatten_columns(columns: pd.Index) -> list[str]:
    return ["__".join(str(part) for part in column) for column in columns]


def load_fma_features(tracks_path: Path, features_path: Path, subset: str) -> pd.DataFrame:
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    features = pd.read_csv(features_path, index_col=0, header=[0, 1, 2])
    features.columns = flatten_columns(features.columns)

    frame = pd.DataFrame(
        {
            "track_id": tracks.index.astype(int),
            "subset": tracks[("set", "subset")],
            "split": tracks[("set", "split")],
            "genre": tracks[("track", "genre_top")],
        }
    ).set_index("track_id")

    frame = frame.join(features, how="inner")
    frame = frame[(frame["subset"] == subset) & frame["genre"].notna()].copy()
    return frame


def cap_per_genre(frame: pd.DataFrame, max_per_genre: int | None) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=False)
    if max_per_genre is None:
        return frame.sample(frac=1, random_state=SEED).reset_index(drop=False)

    sampled_groups = [
        group.sample(min(len(group), max_per_genre), random_state=SEED)
        for _, group in frame.groupby("genre", sort=False)
    ]
    return pd.concat(sampled_groups).sample(frac=1, random_state=SEED).reset_index(drop=False)


def split_features_and_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_columns = [
        column for column in frame.columns if column not in {"track_id", "subset", "split", "genre", "label"}
    ]
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)
    return x, y


def parse_max_features(value: str) -> str | float | int | None:
    if value.lower() == "none":
        return None
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer():
        return int(numeric)
    return numeric


def evaluate(model: RandomForestClassifier, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    predictions = model.predict(x)
    accuracy = accuracy_score(y, predictions)
    return accuracy, predictions


def main() -> None:
    args = parse_args()
    project_dir, tracks_path, features_path = resolve_project_paths()
    output_path = args.output if args.output.is_absolute() else (project_dir / args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_fma_features(tracks_path, features_path, args.subset)
    print(f"Usable FMA {args.subset} rows: {len(metadata):,}")
    print(metadata["split"].value_counts().to_string())
    print(metadata["genre"].value_counts().sort_index().to_string())

    train_df = cap_per_genre(metadata[metadata["split"] == "training"], args.max_train_per_genre)
    val_df = cap_per_genre(metadata[metadata["split"] == "validation"], args.max_eval_per_genre)
    test_df = cap_per_genre(metadata[metadata["split"] == "test"], args.max_eval_per_genre)

    genres = sorted(train_df["genre"].unique())
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    idx_to_genre = {idx: genre for genre, idx in genre_to_idx.items()}

    for frame in (train_df, val_df, test_df):
        frame["label"] = frame["genre"].map(genre_to_idx).astype(int)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Genre mapping: {genre_to_idx}")

    x_train, y_train = split_features_and_labels(train_df)
    x_val, y_val = split_features_and_labels(val_df)
    x_test, y_test = split_features_and_labels(test_df)

    best_val_acc = -1.0
    best_model: RandomForestClassifier | None = None
    best_n_estimators: int | None = None
    history: list[dict[str, float | int]] = []

    max_features = parse_max_features(str(args.max_features))

    for n_estimators in args.n_estimators:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=args.max_depth,
            max_features=max_features,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split,
            max_leaf_nodes=args.max_leaf_nodes,
            ccp_alpha=args.ccp_alpha,
            n_jobs=-1,
            random_state=SEED,
        )
        model.fit(x_train, y_train)

        train_acc, _ = evaluate(model, x_train, y_train)
        val_acc, _ = evaluate(model, x_val, y_val)
        history.append(
            {
                "n_estimators": n_estimators,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )
        print(
            f"n_estimators {n_estimators:4d} | "
            f"train acc {train_acc:.3f} | "
            f"val acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_n_estimators = n_estimators
            with output_path.open("wb") as handle:
                pickle.dump(
                    {
                        "model": best_model,
                        "genre_to_idx": genre_to_idx,
                        "idx_to_genre": idx_to_genre,
                        "feature_columns": [
                            column
                            for column in train_df.columns
                            if column not in {"track_id", "subset", "split", "genre", "label"}
                        ],
                        "config": {
                            "subset": args.subset,
                            "max_train_per_genre": args.max_train_per_genre,
                            "max_eval_per_genre": args.max_eval_per_genre,
                            "n_estimators": best_n_estimators,
                            "max_depth": args.max_depth,
                            "max_features": max_features,
                            "min_samples_split": args.min_samples_split,
                            "min_samples_leaf": args.min_samples_leaf,
                            "max_leaf_nodes": args.max_leaf_nodes,
                            "ccp_alpha": args.ccp_alpha,
                            "seed": SEED,
                        },
                        "history": history,
                    },
                    handle,
                )

    if best_model is None or best_n_estimators is None:
        raise RuntimeError("No model was trained. Provide at least one --n-estimators value.")

    test_acc, test_predictions = evaluate(best_model, x_test, y_test)
    print(f"Best validation accuracy: {best_val_acc:.3f} at n_estimators={best_n_estimators}")
    print(f"Test accuracy: {test_acc:.3f}")

    print("Classification report:")
    print(classification_report(y_test, test_predictions, target_names=genres, digits=3))

    confusion = confusion_matrix(y_test, test_predictions)
    confusion_df = pd.DataFrame(confusion, index=genres, columns=genres)
    print("Confusion matrix:")
    print(confusion_df.to_string())
    print(f"Saved best model to: {output_path}")


if __name__ == "__main__":
    main()
