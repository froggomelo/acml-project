#!/usr/bin/env python3
"""
ACML FMA CNN Experiment Runner

What this script does:
1. Loads FMA metadata.
2. Extracts/caches log-mel spectrograms once.
3. Runs several CNN experiments automatically.
4. Saves per-experiment:
   - training history CSV
   - accuracy/loss graphs
   - confusion matrix
   - classification report
   - trained model
5. Saves one final experiment_summary.csv for your report.

Run on cluster using:
    python train_cnn_experiments.py
"""

import os
import json
import random
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Important for cluster jobs without display
import matplotlib.pyplot as plt

import librosa
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)

import tensorflow as tf
from tensorflow.keras import layers, models

import utils


# -----------------------------
# 1. Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# 2. Project paths
# -----------------------------
PROJECT_DIR = Path.cwd()
RESULTS_DIR = PROJECT_DIR / "results_cnn_experiments"
CACHE_DIR = PROJECT_DIR / "cache"

RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

AUDIO_DIR = os.environ.get("AUDIO_DIR", "./data/fma_small")
METADATA_PATH = os.environ.get("FMA_METADATA", "./data/fma_metadata/tracks.csv")

print("=" * 80)
print("ACML FMA CNN Experiment Runner")
print("Started:", datetime.now())
print("Project directory:", PROJECT_DIR)
print("Audio directory:", AUDIO_DIR)
print("Metadata path:", METADATA_PATH)
print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices("GPU"))
print("=" * 80)


# -----------------------------
# 3. Audio preprocessing
# -----------------------------
def extract_log_mel(
    file_path,
    sr=22050,
    duration=15,
    n_mels=128,
    n_fft=2048,
    hop_length=512
):
    """
    Load an audio file and convert it into a fixed-size log-mel spectrogram.
    """
    audio, sr = librosa.load(
        file_path,
        sr=sr,
        mono=True,
        duration=duration
    )

    target_length = sr * duration

    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel.astype(np.float32)


def load_or_create_dataset(duration=15, n_mels=128):
    """
    Extracts spectrograms once and caches them as an .npz file.
    If the cache exists, it loads directly from cache.
    """
    cache_file = CACHE_DIR / f"fma_small_logmel_{duration}s_{n_mels}mels.npz"

    if cache_file.exists():
        print(f"\nLoading cached spectrogram dataset from: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        splits = data["splits"]
        track_ids = data["track_ids"]
        failed_files = data["failed_files"]
        return X, y, splits, track_ids, failed_files

    print("\nNo cache found. Extracting spectrograms from MP3 files...")
    print("This may take time, but it only needs to happen once.")

    tracks = utils.load(METADATA_PATH)
    small = tracks[tracks["set", "subset"] <= "small"].copy()
    small = small[small["track", "genre_top"].notna()]

    print("Small subset shape:", small.shape)
    print("Class distribution:")
    print(small["track", "genre_top"].value_counts())

    X = []
    y = []
    splits = []
    track_ids = []
    failed_files = []

    for track_id, row in tqdm(small.iterrows(), total=len(small)):
        try:
            file_path = utils.get_audio_path(AUDIO_DIR, track_id)

            if not os.path.exists(file_path):
                failed_files.append(track_id)
                continue

            spec = extract_log_mel(
                file_path,
                duration=duration,
                n_mels=n_mels
            )

            X.append(spec)
            y.append(row["track", "genre_top"])
            splits.append(row["set", "split"])
            track_ids.append(track_id)

        except Exception:
            failed_files.append(track_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    splits = np.array(splits)
    track_ids = np.array(track_ids)
    failed_files = np.array(failed_files)

    print("X shape before channel:", X.shape)
    print("y shape:", y.shape)
    print("Failed files:", len(failed_files))

    # Add CNN channel dimension
    X = X[..., np.newaxis]

    print("X shape after channel:", X.shape)

    np.savez_compressed(
        cache_file,
        X=X,
        y=y,
        splits=splits,
        track_ids=track_ids,
        failed_files=failed_files
    )

    print(f"Cached dataset saved to: {cache_file}")

    return X, y, splits, track_ids, failed_files


# -----------------------------
# 4. CNN model builder
# -----------------------------
def build_cnn_model(
    input_shape,
    num_classes,
    filters=(32, 64, 128),
    dense_units=128,
    dropout=0.3,
    use_batchnorm=False,
    use_global_average_pooling=False,
    learning_rate=0.001
):
    """
    Builds a CNN model from configurable hyperparameters.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    for f in filters:
        model.add(layers.Conv2D(f, (3, 3), activation="relu", padding="same"))

        if use_batchnorm:
            model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D((2, 2)))

    if use_global_average_pooling:
        model.add(layers.GlobalAveragePooling2D())
    else:
        model.add(layers.Flatten())

    model.add(layers.Dense(dense_units, activation="relu"))

    if dropout > 0:
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(num_classes, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# 5. Plotting helpers
# -----------------------------
def plot_history(history_df, experiment_dir, experiment_name):
    """
    Saves accuracy and loss graphs.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["accuracy"], label="Training Accuracy")
    plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{experiment_name}: Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(experiment_dir / "accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["loss"], label="Training Loss")
    plt.plot(history_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{experiment_name}: Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(experiment_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_test, y_pred, class_names, experiment_dir, experiment_name):
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=True)
    plt.title(f"{experiment_name}: Confusion Matrix")
    plt.savefig(experiment_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# 6. Experiment runner
# -----------------------------
def run_single_experiment(
    config,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    class_names
):
    experiment_name = config["name"]
    experiment_dir = RESULTS_DIR / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("Starting experiment:", experiment_name)
    print("Config:")
    print(json.dumps(config, indent=4))
    print("=" * 80)

    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    model = build_cnn_model(
        input_shape=X_train.shape[1:],
        num_classes=len(class_names),
        filters=config["filters"],
        dense_units=config["dense_units"],
        dropout=config["dropout"],
        use_batchnorm=config["use_batchnorm"],
        use_global_average_pooling=config["use_global_average_pooling"],
        learning_rate=config["learning_rate"]
    )

    # Save model architecture summary as text
    with open(experiment_dir / "model_summary.txt", "w") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    callbacks = []

    csv_logger = tf.keras.callbacks.CSVLogger(
        experiment_dir / "training_history.csv",
        append=False
    )
    callbacks.append(csv_logger)

    if config["early_stopping"]:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
            restore_best_weights=True
        )
        callbacks.append(early_stop)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=2
    )

    # Save model
    model.save(experiment_dir / "model.keras")

    # Read history from CSV so early stopping epoch count is accurate
    history_df = pd.read_csv(experiment_dir / "training_history.csv")
    plot_history(history_df, experiment_dir, experiment_name)

    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    with open(experiment_dir / "classification_report.txt", "w") as f:
        f.write(report)

    save_confusion_matrix(y_test, y_pred, class_names, experiment_dir, experiment_name)

    actual_epochs = len(history_df)
    best_val_accuracy = float(history_df["val_accuracy"].max())
    best_val_loss = float(history_df["val_loss"].min())
    final_train_accuracy = float(history_df["accuracy"].iloc[-1])
    final_val_accuracy = float(history_df["val_accuracy"].iloc[-1])

    result = {
        "name": experiment_name,
        "filters": str(config["filters"]),
        "dense_units": config["dense_units"],
        "dropout": config["dropout"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "use_batchnorm": config["use_batchnorm"],
        "use_global_average_pooling": config["use_global_average_pooling"],
        "early_stopping": config["early_stopping"],
        "max_epochs": config["epochs"],
        "actual_epochs": actual_epochs,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "final_train_accuracy": final_train_accuracy,
        "final_val_accuracy": final_val_accuracy,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "experiment_folder": str(experiment_dir)
    }

    with open(experiment_dir / "results.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\nFinished experiment:", experiment_name)
    print("Test accuracy:", test_accuracy)
    print("Weighted F1:", weighted_f1)
    print("Actual epochs:", actual_epochs)

    # Clear memory before next experiment
    tf.keras.backend.clear_session()

    return result


def main():
    # You can change these two values later for a separate experiment.
    DURATION = int(os.environ.get("DURATION", "15"))
    N_MELS = int(os.environ.get("N_MELS", "128"))

    X, y, splits, track_ids, failed_files = load_or_create_dataset(
        duration=DURATION,
        n_mels=N_MELS
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    with open(RESULTS_DIR / "class_names.txt", "w") as f:
        for c in class_names:
            f.write(str(c) + "\n")

    train_mask = splits == "training"
    val_mask = splits == "validation"
    test_mask = splits == "test"

    X_train = X[train_mask]
    y_train = y_encoded[train_mask]

    X_val = X[val_mask]
    y_val = y_encoded[val_mask]

    X_test = X[test_mask]
    y_test = y_encoded[test_mask]

    print("\nDataset split:")
    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    # Normalize using training set only
    mean = np.mean(X_train)
    std = np.std(X_train)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    preprocessing_info = {
        "duration_seconds": DURATION,
        "n_mels": N_MELS,
        "mean_from_training_set": float(mean),
        "std_from_training_set": float(std),
        "num_failed_files": int(len(failed_files)),
        "num_classes": int(len(class_names)),
        "classes": list(map(str, class_names)),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
    }

    with open(RESULTS_DIR / "preprocessing_info.json", "w") as f:
        json.dump(preprocessing_info, f, indent=4)

    # ---------------------------------------------------------
    # EXPERIMENT LIST
    # Start simple. These will run one after another.
    # You can add/remove experiments here.
    # ---------------------------------------------------------
    experiments = [
        {
            "name": "exp01_basic_10_epochs",
            "filters": (32, 64, 128),
            "dense_units": 128,
            "dropout": 0.3,
            "batch_size": 32,
            "learning_rate": 0.001,
            "use_batchnorm": False,
            "use_global_average_pooling": False,
            "early_stopping": False,
            "patience": 0,
            "epochs": 10
        },
        {
            "name": "exp02_basic_30max_earlystop",
            "filters": (32, 64, 128),
            "dense_units": 128,
            "dropout": 0.3,
            "batch_size": 32,
            "learning_rate": 0.001,
            "use_batchnorm": False,
            "use_global_average_pooling": False,
            "early_stopping": True,
            "patience": 5,
            "epochs": 30
        },
        {
            "name": "exp03_dropout_05_30max_earlystop",
            "filters": (32, 64, 128),
            "dense_units": 128,
            "dropout": 0.5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "use_batchnorm": False,
            "use_global_average_pooling": False,
            "early_stopping": True,
            "patience": 5,
            "epochs": 30
        },
        {
            "name": "exp04_batchnorm_gap_50max_earlystop",
            "filters": (32, 64, 128),
            "dense_units": 128,
            "dropout": 0.3,
            "batch_size": 32,
            "learning_rate": 0.001,
            "use_batchnorm": True,
            "use_global_average_pooling": True,
            "early_stopping": True,
            "patience": 7,
            "epochs": 50
        },
        {
            "name": "exp05_deeper_batchnorm_gap_50max_earlystop",
            "filters": (32, 64, 128, 256),
            "dense_units": 256,
            "dropout": 0.4,
            "batch_size": 32,
            "learning_rate": 0.001,
            "use_batchnorm": True,
            "use_global_average_pooling": True,
            "early_stopping": True,
            "patience": 7,
            "epochs": 50
        },
    ]

    all_results = []

    for config in experiments:
        try:
            result = run_single_experiment(
                config,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                class_names
            )
            all_results.append(result)

            summary_df = pd.DataFrame(all_results)
            summary_df = summary_df.sort_values(
                by="test_accuracy",
                ascending=False
            )
            summary_df.to_csv(RESULTS_DIR / "experiment_summary.csv", index=False)

        except Exception as e:
            print("\nExperiment failed:", config["name"])
            print(str(e))
            traceback.print_exc()

            error_dir = RESULTS_DIR / config["name"]
            error_dir.mkdir(exist_ok=True)

            with open(error_dir / "error.txt", "w") as f:
                f.write(str(e))
                f.write("\n\n")
                f.write(traceback.format_exc())

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df.sort_values(by="test_accuracy", ascending=False)
        summary_df.to_csv(RESULTS_DIR / "experiment_summary.csv", index=False)

        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETE")
        print("Best experiments by test accuracy:")
        print(summary_df[[
            "name",
            "actual_epochs",
            "best_val_accuracy",
            "test_accuracy",
            "weighted_f1",
            "dropout",
            "use_batchnorm",
            "use_global_average_pooling"
        ]])
        print("\nSummary saved to:", RESULTS_DIR / "experiment_summary.csv")
        print("=" * 80)
    else:
        print("No experiments completed successfully.")

    print("Finished:", datetime.now())


if __name__ == "__main__":
    main()
