"""
Workflow:
  1. Load preprocessed data from fma_preprocessed/ (small or medium)
  2. Random search over hyperparameters (N_TRIALS)
  3. Retrain best config with multiple seeds for robust test evaluation
  4. Save all trial results and diagnostic plots for the report
"""

from pathlib import Path
import os
import warnings
import json
import random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SUBSET = "medium"              # "small" or "medium" — must exist in fma_preprocessed/

N_TRIALS         = 40           # random-search trials
SEARCH_EPOCHS    = 50           # max epochs per trial (early stopping caps this)
SEARCH_PATIENCE  = 8            # tighter patience during search to save time
SEARCH_BASE_SEED = 42           # same seed for every trial during search

FINAL_SEEDS      = [0, 1, 2, 3, 4]   # final retraining seeds for mean ± std
FINAL_EPOCHS     = 80
FINAL_PATIENCE   = 15

MODEL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODEL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PREFIX = RESULTS_DIR / f"mlp_{SUBSET}"

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

print(f"Loading preprocessed data ({SUBSET}) ...")

def _load_dotenv():
    for base in [Path(__file__).resolve().parents[1], Path.cwd()]:
        p = base / ".env"
        if p.exists():
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return

_load_dotenv()

if os.environ.get("PREPROCESSED_DIR"):
    PROCESSED_DIR = Path(os.environ["PREPROCESSED_DIR"]).expanduser().resolve()
elif os.environ.get("DATASET_DIR"):
    PROCESSED_DIR = Path(os.environ["DATASET_DIR"]).expanduser().resolve() / "fma_preprocessed"
else:
    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "fma_preprocessed"

if not PROCESSED_DIR.is_dir():
    raise FileNotFoundError(
        f"fma_preprocessed/ not found at {PROCESSED_DIR}.\n"
        "Run data_preprocessing.ipynb first."
    )

genre_names = (
    pd.read_csv(PROCESSED_DIR / f"genre_to_idx_{SUBSET}.csv")
    .sort_values("label")["genre"]
    .tolist()
)
num_classes = len(genre_names)

train_meta = pd.read_csv(PROCESSED_DIR / f"tracks_clean_{SUBSET}_training.csv")
val_meta   = pd.read_csv(PROCESSED_DIR / f"tracks_clean_{SUBSET}_validation.csv")
test_meta  = pd.read_csv(PROCESSED_DIR / f"tracks_clean_{SUBSET}_test.csv")

_features = pd.read_csv(PROCESSED_DIR / f"features_{SUBSET}.csv", index_col=0, header=[0, 1, 2])
_features.index = _features.index.astype(int)

def _get_Xy(meta, feats):
    ids = meta["track_id"].astype(int).to_numpy()
    X = np.nan_to_num(feats.reindex(ids).to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = meta["label"].to_numpy(dtype=np.int64)
    return X, y

X_train, y_train = _get_Xy(train_meta, _features)
X_val,   y_val   = _get_Xy(val_meta,   _features)
X_test,  y_test  = _get_Xy(test_meta,  _features)

print(f"Classes ({num_classes}): {genre_names}")
print(f"Data dir: {PROCESSED_DIR}")

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | "
      f"Features: {X_train.shape[1]}\n")

X_tr = torch.FloatTensor(X_train).to(device)
y_tr = torch.LongTensor(y_train).to(device)
X_v  = torch.FloatTensor(X_val).to(device)
y_v  = torch.LongTensor(y_val).to(device)
X_te = torch.FloatTensor(X_test).to(device)

criterion = nn.CrossEntropyLoss()

# ─────────────────────────────────────────────
# 2. REPRODUCIBILITY
# ─────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────
# 3. FLEXIBLE MLP
# ─────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, depth, dropout):
        super().__init__()
        layers = []
        prev   = input_dim
        width  = hidden_dim
        for _ in range(depth):
            layers += [
                nn.Linear(prev, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev  = width
            width = max(width // 2, 64)
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────────
# 4. EARLY STOPPING
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = None
        self.stop      = False

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

# ─────────────────────────────────────────────
# 5. SINGLE TRAINING RUN
# ─────────────────────────────────────────────

def train_once(params, epochs, patience, seed, verbose=False):
    set_seed(seed)

    model = MLP(
        input_dim   = X_train.shape[1],
        num_classes = num_classes,
        hidden_dim  = params["hidden_dim"],
        depth       = params["depth"],
        dropout     = params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = params["lr"],
        weight_decay = params["weight_decay"],
    )
    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size = params["batch_size"],
        shuffle    = True,
    )
    es = EarlyStopping(patience=patience)

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    best_state    = None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_out = model(X_v)
            v_loss  = criterion(val_out, y_v).item()
            v_acc   = (val_out.argmax(dim=1) == y_v).float().mean().item()

        history["train_loss"].append(epoch_loss / len(loader))
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_acc  = v_acc
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}: train={history['train_loss'][-1]:.4f} "
                  f"val_loss={v_loss:.4f} val_acc={v_acc:.4f}")

        es.step(v_loss)
        if es.stop:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_val_acc, best_val_loss, history

# ─────────────────────────────────────────────
# 6. RANDOM SEARCH
# ─────────────────────────────────────────────

def sample_params(rng):
    return {
        "lr":           float(10 ** rng.uniform(-4, -2)),     # log-uniform 1e-4 .. 1e-2
        "weight_decay": float(10 ** rng.uniform(-6, -3)),     # log-uniform 1e-6 .. 1e-3
        "dropout":      float(rng.uniform(0.1, 0.5)),
        "hidden_dim":   int(rng.choice([256, 512, 1024])),
        "depth":        int(rng.integers(2, 6)),              # 2..5 inclusive
        "batch_size":   int(rng.choice([32, 64, 128, 256])),
    }

def run_random_search(n_trials):
    rng = np.random.default_rng(SEARCH_BASE_SEED)
    results = []

    print("═" * 60)
    print(f"  RANDOM SEARCH — {n_trials} trials")
    print("═" * 60 + "\n")

    for trial in range(1, n_trials + 1):
        params = sample_params(rng)
        print(f"Trial {trial:3d}/{n_trials}  "
              f"lr={params['lr']:.2e}  wd={params['weight_decay']:.2e}  "
              f"dropout={params['dropout']:.2f}  hidden={params['hidden_dim']}  "
              f"depth={params['depth']}  batch={params['batch_size']}")

        _, val_acc, val_loss, _ = train_once(
            params,
            epochs   = SEARCH_EPOCHS,
            patience = SEARCH_PATIENCE,
            seed     = SEARCH_BASE_SEED,
            verbose  = False,
        )
        print(f"             → val_acc={val_acc:.4f}  val_loss={val_loss:.4f}\n")

        results.append({**params, "val_acc": val_acc, "val_loss": val_loss, "trial": trial})

    df = pd.DataFrame(results).sort_values("val_acc", ascending=False).reset_index(drop=True)
    return df

search_results = run_random_search(N_TRIALS)
search_results.to_csv(f"{OUT_PREFIX}_search_results.csv", index=False)

print("═" * 60)
print("  TOP 5 CONFIGS BY VALIDATION ACCURACY")
print("═" * 60)
print(search_results.head(5).to_string(index=False))

# Extract best params (clean numeric types for the model constructor)
best_row    = search_results.iloc[0]
best_params = {
    "lr":           float(best_row["lr"]),
    "weight_decay": float(best_row["weight_decay"]),
    "dropout":      float(best_row["dropout"]),
    "hidden_dim":   int(best_row["hidden_dim"]),
    "depth":        int(best_row["depth"]),
    "batch_size":   int(best_row["batch_size"]),
}
print(f"\nBest config: {best_params}")

# ─────────────────────────────────────────────
# 7. SEARCH DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for ax, param in zip(axes, ["lr", "weight_decay", "dropout",
                            "hidden_dim", "depth", "batch_size"]):
    if param in ("lr", "weight_decay"):
        ax.set_xscale("log")
    ax.scatter(search_results[param], search_results["val_acc"], alpha=0.6)
    ax.set_xlabel(param)
    ax.set_ylabel("Val Accuracy")
    ax.set_title(f"Val Acc vs {param}")
    ax.grid(True, alpha=0.3)
plt.suptitle(f"MLP ({SUBSET}) — Random Search: Hyperparameter vs Validation Accuracy", y=1.00)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_search_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 4))
plt.bar(range(len(search_results)), search_results["val_acc"].values, color="steelblue")
plt.axhline(search_results["val_acc"].max(), color="red", linestyle="--",
            label=f"Best: {search_results['val_acc'].max():.4f}")
plt.xlabel("Trial (ranked by val acc)")
plt.ylabel("Val Accuracy")
plt.title(f"MLP ({SUBSET}) — Random Search Trial Scores (sorted)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_search_ranked_trials.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 8. FINAL EVALUATION — BEST CONFIG, MULTIPLE SEEDS
# ─────────────────────────────────────────────

print("\n" + "═" * 60)
print(f"  FINAL EVALUATION — best config × {len(FINAL_SEEDS)} seeds")
print("═" * 60)

final_results     = []
best_test_acc     = -1.0
best_seed_history = None
best_seed_preds   = None

for seed in FINAL_SEEDS:
    print(f"\n  Seed {seed}")
    model, val_acc, val_loss, history = train_once(
        best_params,
        epochs   = FINAL_EPOCHS,
        patience = FINAL_PATIENCE,
        seed     = seed,
        verbose  = True,
    )
    model.eval()
    with torch.no_grad():
        test_preds = model(X_te).argmax(dim=1).cpu().numpy()
    test_acc = (test_preds == y_test).mean()

    print(f"  → seed={seed}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")
    final_results.append({
        "seed": seed, "val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc,
    })

    if test_acc > best_test_acc:
        best_test_acc     = test_acc
        best_seed_history = history
        best_seed_preds   = test_preds

final_df = pd.DataFrame(final_results)
final_df.to_csv(f"{OUT_PREFIX}_final_seeds.csv", index=False)

mean_test = final_df["test_acc"].mean()
std_test  = final_df["test_acc"].std()
mean_val  = final_df["val_acc"].mean()
std_val   = final_df["val_acc"].std()

print("\n" + "═" * 60)
print("  FINAL RESULTS")
print("═" * 60)
print(final_df.to_string(index=False))
print(f"\n  Val  Accuracy: {mean_val:.4f} ± {std_val:.4f}")
print(f"  Test Accuracy: {mean_test:.4f} ± {std_test:.4f}")

# ─────────────────────────────────────────────
# 9. BEST-RUN DIAGNOSTICS
# ─────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(best_seed_history["train_loss"], label="Train Loss")
ax1.plot(best_seed_history["val_loss"],   label="Val Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title(f"MLP ({SUBSET}) — Best-config training Loss"); ax1.legend()
ax2.plot(best_seed_history["val_acc"], color="green", label="Val Accuracy")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.set_title(f"MLP ({SUBSET}) — Best-config training Accuracy"); ax2.legend()
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_final_curves.png", dpi=150)
plt.show()

cm = confusion_matrix(y_test, best_seed_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=genre_names, yticklabels=genre_names, cmap="magma")
plt.title(f"MLP ({SUBSET}) — Confusion Matrix (Test, best seed)")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_final_confusion_matrix.png", dpi=150)
plt.show()

print("\nClassification report (best seed):")
print(classification_report(y_test, best_seed_preds, target_names=genre_names))

# ─────────────────────────────────────────────
# 10. SUMMARY JSON
# ─────────────────────────────────────────────

summary = {
    "subset":              SUBSET,
    "n_trials":            N_TRIALS,
    "best_params":         best_params,
    "search_best_val_acc": float(search_results["val_acc"].max()),
    "final_seeds":         FINAL_SEEDS,
    "final_val_acc_mean":  float(mean_val),
    "final_val_acc_std":   float(std_val),
    "final_test_acc_mean": float(mean_test),
    "final_test_acc_std":  float(std_test),
}
with open(f"{OUT_PREFIX}_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved:")
for suffix in [
    "search_results.csv", "search_diagnostics.png",
    "search_ranked_trials.png", "final_seeds.csv",
    "final_curves.png", "final_confusion_matrix.png",
    "summary.json",
]:
    print(f"  {OUT_PREFIX}_{suffix}")
