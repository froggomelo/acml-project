"""
Workflow:
  1. Load preprocessed data from fma_preprocessed/ (small or medium)
  2. Random search over hyperparameters with val-set scoring
  3. Retrain best config across FINAL_SEEDS for mean ± std test evaluation
  4. Save all trial results and diagnostic plots for the report

"""

from pathlib import Path
import os
import warnings
import json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SUBSET = "medium"             # "small" or "medium" — must exist in fma_preprocessed/

N_TRIALS          = 40        # random-search trials
SEARCH_MAX_ROUNDS = 500       # max boosting rounds per trial
SEARCH_PATIENCE   = 20        # early stopping patience during search
SEARCH_BASE_SEED  = 42        # same seed for every trial during search

FINAL_SEEDS       = [42, 0, 1]   # set to [42] for single-seed eval
FINAL_MAX_ROUNDS  = 1500
FINAL_PATIENCE    = 30

MODEL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODEL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PREFIX = RESULTS_DIR / f"xgb_{SUBSET}"

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

features = pd.read_csv(PROCESSED_DIR / f"features_{SUBSET}.csv", index_col=0, header=[0, 1, 2])
features.index = features.index.astype(int)

def _get_Xy(meta, feats):
    ids = meta["track_id"].astype(int).to_numpy()
    X = np.nan_to_num(feats.reindex(ids).to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = meta["label"].to_numpy(dtype=np.int64)
    return X, y

X_train, y_train = _get_Xy(train_meta, features)
X_val,   y_val   = _get_Xy(val_meta,   features)
X_test,  y_test  = _get_Xy(test_meta,  features)

print(f"Classes ({num_classes}): {genre_names}")
print(f"Data dir: {PROCESSED_DIR}")

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | "
      f"Features: {X_train.shape[1]}")


# ─────────────────────────────────────────────
# 2. CLASS WEIGHTS (for imbalanced medium)
# ─────────────────────────────────────────────

class_counts = pd.Series(y_train).value_counts().sort_index()
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\nTraining class distribution (max/min ratio: {imbalance_ratio:.2f}):")
for label, count in class_counts.items():
    print(f"  {genre_names[label]:25s} {count}")

sample_weights = compute_sample_weight("balanced", y_train)
print()

# ─────────────────────────────────────────────
# 3. SINGLE TRAINING RUN
# ─────────────────────────────────────────────

def train_once(params, seed, max_rounds, patience):
    model = XGBClassifier(
        **params,
        n_estimators          = max_rounds,
        early_stopping_rounds = patience,
        eval_metric           = "mlogloss",
        random_state          = seed,
        n_jobs                = -1,
        tree_method           = "hist",
        verbosity             = 0,
    )
    model.fit(
        X_train, y_train,
        sample_weight = sample_weights,
        eval_set      = [(X_train, y_train), (X_val, y_val)],
        verbose       = False,
    )
    val_preds   = model.predict(X_val)
    val_acc     = (val_preds == y_val).mean()
    val_logloss = model.evals_result()["validation_1"]["mlogloss"][model.best_iteration]
    return model, val_acc, val_logloss

# ─────────────────────────────────────────────
# 4. HYPERPARAMETER SAMPLING
# ─────────────────────────────────────────────

def sample_params(rng):
    #Sample one random hyperparameter configuration
    return {
        "learning_rate":    float(10 ** rng.uniform(-2, -0.5)),   # log-uniform 0.01 .. 0.32
        "max_depth":        int(rng.integers(3, 11)),             # 3..10
        "subsample":        float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "min_child_weight": int(rng.integers(1, 11)),             # 1..10
        "reg_alpha":        float(10 ** rng.uniform(-3, 0)),      # log-uniform 0.001 .. 1
        "reg_lambda":       float(10 ** rng.uniform(-1, 1)),      # log-uniform 0.1 .. 10
    }

# ─────────────────────────────────────────────
# 5. RANDOM SEARCH
# ─────────────────────────────────────────────

def run_random_search(n_trials):
    rng = np.random.default_rng(SEARCH_BASE_SEED)
    results = []

    print("═" * 60)
    print(f"  RANDOM SEARCH — {n_trials} trials")
    print("═" * 60 + "\n")

    for trial in range(1, n_trials + 1):
        params = sample_params(rng)
        print(f"Trial {trial:3d}/{n_trials}  "
              f"lr={params['learning_rate']:.3f}  depth={params['max_depth']}  "
              f"sub={params['subsample']:.2f}  col={params['colsample_bytree']:.2f}  "
              f"mcw={params['min_child_weight']}  "
              f"α={params['reg_alpha']:.3f}  λ={params['reg_lambda']:.2f}")

        model, val_acc, val_logloss = train_once(
            params,
            seed       = SEARCH_BASE_SEED,
            max_rounds = SEARCH_MAX_ROUNDS,
            patience   = SEARCH_PATIENCE,
        )
        print(f"             → val_acc={val_acc:.4f}  val_logloss={val_logloss:.4f}  "
              f"best_iter={model.best_iteration}\n")

        results.append({
            **params,
            "val_acc":     val_acc,
            "val_logloss": val_logloss,
            "best_iter":   model.best_iteration,
            "trial":       trial,
        })

    df = pd.DataFrame(results).sort_values("val_acc", ascending=False).reset_index(drop=True)
    return df

search_results = run_random_search(N_TRIALS)
search_results.to_csv(f"{OUT_PREFIX}_search_results.csv", index=False)

print("═" * 60)
print("  TOP 5 CONFIGS BY VALIDATION ACCURACY")
print("═" * 60)
print(search_results.head(5).to_string(index=False))

best_row = search_results.iloc[0]
best_params = {
    "learning_rate":    float(best_row["learning_rate"]),
    "max_depth":        int(best_row["max_depth"]),
    "subsample":        float(best_row["subsample"]),
    "colsample_bytree": float(best_row["colsample_bytree"]),
    "min_child_weight": int(best_row["min_child_weight"]),
    "reg_alpha":        float(best_row["reg_alpha"]),
    "reg_lambda":       float(best_row["reg_lambda"]),
}
print(f"\nBest config: {best_params}")

# ─────────────────────────────────────────────
# 6. SEARCH DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────

plot_params = ["learning_rate", "max_depth", "subsample", "colsample_bytree",
               "min_child_weight", "reg_alpha", "reg_lambda"]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for ax, param in zip(axes, plot_params):
    if param in ("learning_rate", "reg_alpha", "reg_lambda"):
        ax.set_xscale("log")
    ax.scatter(search_results[param], search_results["val_acc"], alpha=0.6)
    ax.set_xlabel(param)
    ax.set_ylabel("Val Accuracy")
    ax.set_title(f"Val Acc vs {param}")
    ax.grid(True, alpha=0.3)
axes[7].axis("off")
plt.suptitle(f"XGBoost ({SUBSET}) — Random Search: Hyperparameter vs Val Accuracy", y=1.00)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_search_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 4))
plt.bar(range(len(search_results)), search_results["val_acc"].values, color="steelblue")
plt.axhline(search_results["val_acc"].max(), color="red", linestyle="--",
            label=f"Best: {search_results['val_acc'].max():.4f}")
plt.xlabel("Trial (ranked by val acc)")
plt.ylabel("Val Accuracy")
plt.title(f"XGBoost ({SUBSET}) — Random Search Trial Scores (sorted)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_search_ranked_trials.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 7. FINAL EVALUATION — BEST CONFIG, MULTIPLE SEEDS
# ─────────────────────────────────────────────

print("\n" + "═" * 60)
print(f"  FINAL EVALUATION — best config × {len(FINAL_SEEDS)} seed(s)")
print("═" * 60)

final_results   = []
best_test_acc   = -1.0
best_seed_model = None
best_seed_preds = None

for seed in FINAL_SEEDS:
    print(f"\n  Seed {seed}")
    model, val_acc, val_logloss = train_once(
        best_params,
        seed       = seed,
        max_rounds = FINAL_MAX_ROUNDS,
        patience   = FINAL_PATIENCE,
    )
    test_preds = model.predict(X_test)
    test_acc   = (test_preds == y_test).mean()
    test_macro_f1 = f1_score(y_test, test_preds, average="macro")

    print(f"  → seed={seed}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  "
          f"macro_f1={test_macro_f1:.4f}  best_iter={model.best_iteration}")
    final_results.append({
        "seed":          seed,
        "val_acc":       val_acc,
        "val_logloss":   val_logloss,
        "test_acc":      test_acc,
        "test_macro_f1": test_macro_f1,
        "best_iter":     model.best_iteration,
    })

    if test_acc > best_test_acc:
        best_test_acc   = test_acc
        best_seed_model = model
        best_seed_preds = test_preds

final_df = pd.DataFrame(final_results)
final_df.to_csv(f"{OUT_PREFIX}_final_seeds.csv", index=False)

mean_test = final_df["test_acc"].mean()
std_test  = final_df["test_acc"].std() if len(FINAL_SEEDS) > 1 else 0.0
mean_val  = final_df["val_acc"].mean()
std_val   = final_df["val_acc"].std() if len(FINAL_SEEDS) > 1 else 0.0
mean_f1   = final_df["test_macro_f1"].mean()
std_f1    = final_df["test_macro_f1"].std() if len(FINAL_SEEDS) > 1 else 0.0

print("\n" + "═" * 60)
print("  FINAL RESULTS")
print("═" * 60)
print(final_df.to_string(index=False))
if len(FINAL_SEEDS) > 1:
    print(f"\n  Val  Accuracy : {mean_val:.4f} ± {std_val:.4f}")
    print(f"  Test Accuracy : {mean_test:.4f} ± {std_test:.4f}")
    print(f"  Test Macro-F1 : {mean_f1:.4f} ± {std_f1:.4f}")
else:
    print(f"\n  Val  Accuracy : {mean_val:.4f}")
    print(f"  Test Accuracy : {mean_test:.4f}")
    print(f"  Test Macro-F1 : {mean_f1:.4f}")

# ─────────────────────────────────────────────
# 8. BEST-RUN DIAGNOSTICS
# ─────────────────────────────────────────────

results    = best_seed_model.evals_result()
train_loss = results["validation_0"]["mlogloss"]
val_loss   = results["validation_1"]["mlogloss"]

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(train_loss, label="Train mlogloss")
ax.plot(val_loss,   label="Val mlogloss")
ax.axvline(best_seed_model.best_iteration, color="red", linestyle="--",
           label=f"Best iter = {best_seed_model.best_iteration}")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("mlogloss")
ax.set_title(f"XGBoost ({SUBSET}) — Best-config training curves")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_final_curves.png", dpi=150)
plt.show()

cm = confusion_matrix(y_test, best_seed_preds)
cm_size = max(8, int(num_classes * 0.8))
plt.figure(figsize=(cm_size, max(6, int(num_classes * 0.6))))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=genre_names, yticklabels=genre_names, cmap="magma")
plt.title(f"XGBoost ({SUBSET}) — Confusion Matrix (Test, best seed)")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_final_confusion_matrix.png", dpi=150)
plt.show()

feat_names = [str(c) for c in features.columns]
feat_imp = pd.Series(best_seed_model.feature_importances_, index=feat_names) \
             .sort_values(ascending=False)[:20]
plt.figure(figsize=(8, 6))
plt.barh(feat_imp.index[::-1], feat_imp.values[::-1], color="steelblue")
plt.title(f"XGBoost ({SUBSET}) — Top 20 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_feature_importance.png", dpi=150)
plt.show()

print("\nClassification report (best seed):")
print(classification_report(y_test, best_seed_preds, target_names=genre_names))

# ─────────────────────────────────────────────
# 9. SUMMARY JSON
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
    "final_test_f1_mean":  float(mean_f1),
    "final_test_f1_std":   float(std_f1),
}
with open(f"{OUT_PREFIX}_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved:")
for suffix in [
    "search_results.csv", "search_diagnostics.png",
    "search_ranked_trials.png", "final_seeds.csv",
    "final_curves.png", "final_confusion_matrix.png",
    "feature_importance.png", "summary.json",
]:
    print(f"  {OUT_PREFIX}_{suffix}")
