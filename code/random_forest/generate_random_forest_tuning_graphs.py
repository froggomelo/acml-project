#!/usr/bin/env python3
"""Generate Random Forest tuning graphs from a tuning-results CSV.

The script uses only columns that are already present in the tuning CSV:
train_acc, val_acc, fit_seconds, grid_index, completed_at, and hyperparameters.

Example:
    python code/generate_random_forest_tuning_graphs.py \
        --csv code/random_forest/random_forest_tuning_results_small.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RANDOM_FOREST_DIR = SCRIPT_DIR
DEFAULT_CSV_PATH = RANDOM_FOREST_DIR / "random_forest_tuning_results_small.csv"

REQUIRED_COLUMNS = {
    "n_estimators",
    "max_depth",
    "max_features",
    "min_samples_split",
    "min_samples_leaf",
    "max_leaf_nodes",
    "ccp_alpha",
    "train_acc",
    "val_acc",
    "fit_seconds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Random Forest tuning graphs from a results CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to random_forest_tuning_results_<subset>.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PNG graphs will be saved.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top configurations to show in the top-configurations chart.",
    )
    return parser.parse_args()


def infer_subset(csv_path: Path) -> str:
    match = re.search(r"random_forest_tuning_results_(.+)\.csv$", csv_path.name)
    return match.group(1) if match else csv_path.stem


def prepare_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required column(s): {missing_text}")

    df = df.copy()
    df["generalization_gap"] = df["train_acc"] - df["val_acc"]

    for column in ["max_depth", "max_leaf_nodes"]:
        df[column] = df[column].where(df[column].notna(), np.nan)

    if "completed_at" in df.columns:
        df["completed_at"] = pd.to_datetime(df["completed_at"], errors="coerce")

    if "grid_index" in df.columns:
        df = df.sort_values("grid_index")
    elif "completed_at" in df.columns:
        df = df.sort_values("completed_at")

    return df.reset_index(drop=True)


def default_output_dir(csv_path: Path) -> Path:
    subset = infer_subset(csv_path)
    return RANDOM_FOREST_DIR / f"tuning_graphs_{subset}"


def save_plot(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def percent_axis(ax: plt.Axes, axis: str = "y") -> None:
    formatter = plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)


def plot_accuracy_by_estimators(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        df.groupby("n_estimators", as_index=False)
        .agg(
            mean_train_acc=("train_acc", "mean"),
            mean_val_acc=("val_acc", "mean"),
            best_val_acc=("val_acc", "max"),
        )
        .sort_values("n_estimators")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grouped["n_estimators"], grouped["mean_train_acc"], marker="o", label="Mean train accuracy")
    ax.plot(grouped["n_estimators"], grouped["mean_val_acc"], marker="o", label="Mean validation accuracy")
    ax.plot(grouped["n_estimators"], grouped["best_val_acc"], marker="o", label="Best validation accuracy")
    ax.set_title("Accuracy vs Number of Trees")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Accuracy")
    percent_axis(ax)
    ax.grid(alpha=0.25)
    ax.legend()
    save_plot(fig, output_dir / "01_accuracy_vs_n_estimators.png")


def plot_train_vs_validation(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        df["train_acc"],
        df["val_acc"],
        c=df["n_estimators"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    lower = min(df["train_acc"].min(), df["val_acc"].min())
    upper = max(df["train_acc"].max(), df["val_acc"].max())
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--", label="Train = validation")
    ax.set_title("Training Accuracy vs Validation Accuracy")
    ax.set_xlabel("Training accuracy")
    ax.set_ylabel("Validation accuracy")
    percent_axis(ax, "x")
    percent_axis(ax, "y")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Number of trees")
    save_plot(fig, output_dir / "02_train_vs_validation_accuracy.png")


def plot_generalization_gap(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df["val_acc"],
        df["generalization_gap"],
        c=df["fit_seconds"],
        cmap="plasma",
        alpha=0.75,
        edgecolors="none",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Generalization Gap")
    ax.set_xlabel("Validation accuracy")
    ax.set_ylabel("Train accuracy - validation accuracy")
    percent_axis(ax, "x")
    percent_axis(ax, "y")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Fit seconds")
    save_plot(fig, output_dir / "03_generalization_gap.png")


def plot_validation_accuracy_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["val_acc"], bins=24, color="#3b82f6", edgecolor="white")
    ax.axvline(df["val_acc"].max(), color="#dc2626", linestyle="--", linewidth=1.5, label="Best validation accuracy")
    ax.set_title("Validation Accuracy Distribution")
    ax.set_xlabel("Validation accuracy")
    ax.set_ylabel("Number of configurations")
    percent_axis(ax, "x")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_plot(fig, output_dir / "04_validation_accuracy_distribution.png")


def plot_fit_time_vs_validation(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df["fit_seconds"],
        df["val_acc"],
        c=df["n_estimators"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    ax.set_title("Fit Time vs Validation Accuracy")
    ax.set_xlabel("Fit time per configuration, seconds")
    ax.set_ylabel("Validation accuracy")
    percent_axis(ax)
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Number of trees")
    save_plot(fig, output_dir / "05_fit_time_vs_validation_accuracy.png")


def format_config(row: pd.Series) -> str:
    depth = "None" if pd.isna(row["max_depth"]) else str(int(row["max_depth"]))
    leaf_nodes = "None" if pd.isna(row["max_leaf_nodes"]) else str(int(row["max_leaf_nodes"]))
    return (
        f"n={int(row['n_estimators'])}, depth={depth}, feat={row['max_features']}\n"
        f"split={int(row['min_samples_split'])}, leaf={int(row['min_samples_leaf'])}, "
        f"nodes={leaf_nodes}, alpha={row['ccp_alpha']}"
    )


def plot_top_configurations(df: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    top = df.sort_values(["val_acc", "train_acc"], ascending=[False, False]).head(top_n).copy()
    top["config_label"] = top.apply(format_config, axis=1)
    top = top.sort_values("val_acc", ascending=True)

    fig_height = max(6, top_n * 0.62)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    bars = ax.barh(top["config_label"], top["val_acc"], color="#0f766e")
    ax.set_title(f"Top {len(top)} Random Forest Configurations")
    ax.set_xlabel("Validation accuracy")
    percent_axis(ax, "x")
    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, top["val_acc"]):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.2%}", va="center", fontsize=8)

    save_plot(fig, output_dir / "06_top_configurations.png")


def plot_tuning_progress(df: pd.DataFrame, output_dir: Path) -> None:
    progress_df = df.copy()
    if "grid_index" in progress_df.columns:
        progress_df = progress_df.sort_values("grid_index")
        x_values = progress_df["grid_index"]
        x_label = "Grid index"
    elif "completed_at" in progress_df.columns:
        progress_df = progress_df.sort_values("completed_at")
        x_values = np.arange(1, len(progress_df) + 1)
        x_label = "Completed configuration order"
    else:
        x_values = np.arange(1, len(progress_df) + 1)
        x_label = "Configuration order"

    progress_df["best_so_far"] = progress_df["val_acc"].cummax()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x_values, progress_df["val_acc"], s=12, alpha=0.45, label="Validation accuracy")
    ax.plot(x_values, progress_df["best_so_far"], color="#dc2626", linewidth=2, label="Best so far")
    ax.set_title("Validation Accuracy Over Tuning Progress")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Validation accuracy")
    percent_axis(ax)
    ax.grid(alpha=0.25)
    ax.legend()
    save_plot(fig, output_dir / "07_tuning_progress.png")


def label_value(value: object) -> str:
    if pd.isna(value):
        return "None"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def ordered_labels(series: pd.Series) -> list[str]:
    labels = pd.Series(series.map(label_value).unique()).dropna().tolist()

    def sort_key(label: str) -> tuple[int, float, str]:
        if label == "None":
            return (2, 0.0, label)
        try:
            return (0, float(label), label)
        except ValueError:
            return (1, 0.0, label)

    return sorted(labels, key=sort_key)


def plot_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    row_param: str,
    col_param: str,
    filename: str,
    title: str,
) -> None:
    heatmap_df = df.copy()
    heatmap_df[row_param] = heatmap_df[row_param].map(label_value)
    heatmap_df[col_param] = heatmap_df[col_param].map(label_value)

    pivot = heatmap_df.pivot_table(
        index=row_param,
        columns=col_param,
        values="val_acc",
        aggfunc="max",
    )

    if pivot.empty:
        return

    pivot = pivot.reindex(index=ordered_labels(heatmap_df[row_param]))
    pivot = pivot.reindex(columns=ordered_labels(heatmap_df[col_param]))

    fig_width = max(7, 1.2 * len(pivot.columns))
    fig_height = max(5, 0.7 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")

    ax.set_title(title)
    ax.set_xlabel(col_param)
    ax.set_ylabel(row_param)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = pivot.iloc[row_idx, col_idx]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.1%}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Best validation accuracy")
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    save_plot(fig, output_dir / filename)


def plot_all_graphs(df: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    plot_accuracy_by_estimators(df, output_dir)
    plot_train_vs_validation(df, output_dir)
    plot_generalization_gap(df, output_dir)
    plot_validation_accuracy_distribution(df, output_dir)
    plot_fit_time_vs_validation(df, output_dir)
    plot_top_configurations(df, output_dir, top_n)
    plot_tuning_progress(df, output_dir)
    plot_heatmap(
        df,
        output_dir,
        row_param="max_depth",
        col_param="min_samples_leaf",
        filename="08_heatmap_max_depth_vs_min_samples_leaf.png",
        title="Best Validation Accuracy: Max Depth vs Min Samples Leaf",
    )
    plot_heatmap(
        df,
        output_dir,
        row_param="max_features",
        col_param="n_estimators",
        filename="09_heatmap_max_features_vs_n_estimators.png",
        title="Best Validation Accuracy: Max Features vs Number of Trees",
    )
    plot_heatmap(
        df,
        output_dir,
        row_param="min_samples_split",
        col_param="min_samples_leaf",
        filename="10_heatmap_min_samples_split_vs_min_samples_leaf.png",
        title="Best Validation Accuracy: Min Samples Split vs Min Samples Leaf",
    )


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir(csv_path).resolve()

    df = prepare_results(csv_path)
    plot_all_graphs(df, output_dir, args.top_n)

    best = df.sort_values("val_acc", ascending=False).iloc[0]
    print()
    print(f"Generated {len(list(output_dir.glob('*.png')))} graph(s) in {output_dir}")
    print(f"Best validation accuracy: {best['val_acc']:.2%}")
    print(f"Best configuration: {format_config(best).replace(chr(10), '; ')}")


if __name__ == "__main__":
    main()
