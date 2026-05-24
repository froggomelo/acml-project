# ACML Project: FMA Music Classification

Course project repository for music genre classification experiments on the Free Music Archive (FMA) dataset.

The normal workflow is:

1. Run setup.
2. Run data preprocessing.
3. Run the model notebook or script you want.

## Folder Structure Before Setup

```text
.
|-- README.md                  Project overview and workflow notes
|-- setup.sh                   Creates `.venv/`, installs dependencies, and downloads FMA data
|-- .env.example               Template for local paths and dataset settings
|-- code/
|   |-- data_preprocessing.ipynb
|   |-- random_forest/         Random Forest notebook, tuning helper, and outputs
|   |-- cnn/                   CNN notebook and older automated CNN experiment script
|   |-- crnn/                  CRNN notebooks and experiment outputs
|   |-- MLP/                   MLP scripts
|   `-- XGBoost/               XGBoost scripts
```

Before running `setup.sh`, create `.env` from `.env.example` and set your local paths.

## Configure `.env`

Create your local environment file:

```bash
cp .env.example .env
```

Open `.env` and set these variables:

| Variable | Required | Description |
|---|---|---|
| `DATASET_DIR` | yes | Directory where `fma_small/`, `fma_medium/`, and `fma_metadata/` live. Setup downloads data here. |
| `PREPROCESSED_DIR` | no | Where cleaned CSVs, filtered feature CSVs, and spectrograms are written. Defaults to `DATASET_DIR/fma_preprocessed`. |
| `DATASET_SIZE` | yes | Which subset to use: `small` (8 k tracks, 8 genres), `medium` (25 k tracks, 16 genres), or `both`. Controls which audio files are downloaded and which subsets the preprocessing notebook processes. |
| `DOWNLOAD_SPECTROGRAMS` | no | Set to `1` for setup to download the audio files selected by `DATASET_SIZE`. Default is `0` (only set up the Python environment, skip audio download). |
| `PREPROCESS_FOR` | no | Which spectrogram formats the preprocessing notebook generates: `cnn` (10-second clips), `crnn` (20-second clips), `both`, or `none`. Default is `both`. |
| `PYTHON_BIN` | no | Existing Python 3.10+ interpreter to use instead of auto-detection/source build. Useful when a managed Python already has `venv`, `ctypes`, and `sqlite3`. |
| `SETUP_BUILD_DIR` | no | Scratch directory for source builds. Defaults to `.setup-build/` inside the repository. |
| `MAKE_JOBS` | no | Number of parallel jobs to use when compiling local dependencies. Defaults to auto-detection. |

This file is local to your checkout; do not commit personal paths.

## Setup

From the project root, run:

```bash
bash setup.sh
```

This script:

- creates a local Python virtual environment in `.venv/`
- builds a local Python in `.python/` if no suitable Python 3.10+ interpreter is available; when needed, it also builds local SQLite/libffi for `sqlite3` and `ctypes`
- installs Python dependencies (NumPy, pandas, scikit-learn, XGBoost, librosa, TensorFlow, Optuna, Jupyter, and others)
- installs PyTorch and torchaudio (GPU build auto-detected from your NVIDIA driver; falls back to CPU)
- downloads and extracts `fma_metadata/`
- downloads and extracts `fma_small/` when `DOWNLOAD_SPECTROGRAMS=1` and `DATASET_SIZE=small` or `both`
- downloads and extracts `fma_medium/` when `DOWNLOAD_SPECTROGRAMS=1` and `DATASET_SIZE=medium` or `both`

Setup does not run preprocessing and does not generate cleaned CSVs, feature CSVs, or spectrogram files. Run `code/data_preprocessing.ipynb` separately for that.

### Folder Structure After Setup

After `setup.sh` finishes, your project should look like this:

```text
.
|-- README.md
|-- setup.sh
|-- .env
|-- .env.example
|-- .venv/                     Local Python environment created by setup
|-- .python/                   Local Python build, created only if setup cannot find a suitable interpreter
|-- code/                      Notebooks and model code
|-- fma_metadata/              Downloaded FMA metadata CSV files
|-- fma_small/                 Downloaded when DOWNLOAD_SPECTROGRAMS=1 and DATASET_SIZE=small or both
`-- fma_medium/                Downloaded when DOWNLOAD_SPECTROGRAMS=1 and DATASET_SIZE=medium or both
```

If you set `DATASET_DIR` to another location, the downloaded directories will be created there instead of inside the repository. The downloaded zip files are removed after extraction.

After setup finishes, activate the environment:

```bash
source .venv/bin/activate
```

If you need to force CPU-only PyTorch, run setup as:

```bash
PYTORCH_BUILD=cpu bash setup.sh
```

If setup previously failed with `ModuleNotFoundError: No module named '_ctypes'`, remove the partial local environments and rerun:

```bash
rm -rf .venv .python
bash setup.sh
```

Alternatively, use a known-good Python explicitly:

```bash
PYTHON_BIN=/path/to/python3 bash setup.sh
```

If your dataset is stored somewhere outside the repository, set `DATASET_DIR` in `.env` before running setup. The notebooks also read it at runtime.

## Data Preprocessing

`setup.sh` downloads the requested FMA files, but it does not create the cleaned metadata, filtered tabular features, or spectrogram arrays. Run preprocessing manually from the project root when those generated files are needed:

```bash
jupyter notebook
```

Open and run:

```text
code/data_preprocessing.ipynb
```

The notebook reads `DATASET_DIR`, `PREPROCESSED_DIR`, `DATASET_SIZE`, and `PREPROCESS_FOR` from `.env`. 

The notebook creates `fma_preprocessed/` (or the directory set in `PREPROCESSED_DIR`), including:

- `tracks_clean_small.csv`
- `tracks_clean_small_training.csv`
- `tracks_clean_small_validation.csv`
- `tracks_clean_small_test.csv`
- `tracks_clean_medium.csv`
- `tracks_clean_medium_training.csv`
- `tracks_clean_medium_validation.csv`
- `tracks_clean_medium_test.csv`
- `genre_to_idx_small.csv`
- `genre_to_idx_medium.csv`
- `features_small.csv`
- `features_medium.csv`
- `spectrograms_small_10_manifest.csv` when `PREPROCESS_FOR=cnn` or `both`
- `spectrograms_medium_10_manifest.csv` when `DATASET_SIZE=medium` or `both` and `PREPROCESS_FOR=cnn` or `both`
- `spectrograms_small_20_manifest.csv` when `PREPROCESS_FOR=crnn` or `both`
- `spectrograms_medium_20_manifest.csv` when `DATASET_SIZE=medium` or `both` and `PREPROCESS_FOR=crnn` or `both`
- `spectrograms_<size>_<10|20>_training.csv`
- `spectrograms_<size>_<10|20>_validation.csv`
- `spectrograms_<size>_<10|20>_test.csv`
- `spectrograms_<size>_<10|20>_extraction_errors.csv` when any MP3 cannot be decoded
- `spectrograms_<size>_<10|20>/`

For feature-based models, the notebook loads `fma_metadata/features.csv` with its three-row FMA header, filters it to the cleaned track ids, and writes `features_small.csv` and `features_medium.csv` with 518 feature columns. Random Forest, MLP, and XGBoost then align those feature rows with the cleaned split CSVs by `track_id`; Random Forest flattens the multi-level feature names when it loads them.

For spectrogram-based models, the notebook converts local MP3 files to mono 16 kHz audio, center-crops or zero-pads each clip, computes a 64-bin mel spectrogram with `n_fft=400` and `hop_length=160`, converts it to log power, applies per-clip z-score normalization, and saves `float32` `.npy` arrays. `PREPROCESS_FOR=cnn` generates 10-second spectrograms with shape `(1, 64, 1001)`, while `PREPROCESS_FOR=crnn` generates 20-second spectrograms with shape `(1, 64, 2001)`. Additional decoding failures during extraction are omitted from the manifest and written to `spectrograms_<subset>_<10|20>_extraction_errors.csv`.

Run preprocessing before running any model. Random Forest, MLP, and XGBoost read the cleaned CSV files and `features_<subset>.csv`. The CNN and CRNN read the spectrogram manifest CSVs and the `.npy` spectrogram files. If you only need the tabular models, set `PREPROCESS_FOR=none` to skip spectrogram generation.

## Random Forest

Open and run:

```text
code/random_forest/random_forest_pipeline.ipynb
```

Before running all cells, choose the dataset size near the top of the notebook:

```python
SUBSET = "medium"  # or "small"
```

The random forest notebook:

- loads cleaned train, validation, and test CSVs from `fma_preprocessed/`
- joins them with handcrafted audio features from `fma_preprocessed/features_<subset>.csv`
- trains a random forest classifier
- optionally runs the hyperparameter tuning grid
- saves the best model checkpoint and evaluation outputs in `code/random_forest/results/`

Important outputs include:

- `code/random_forest/results/best_random_forest_<subset>.pkl`
- `code/random_forest/results/random_forest_tuning_results_<subset>.csv`
- `code/random_forest/results/random_forest_tuning_results_<subset>.jsonl`
- `code/random_forest/results/confusion_matrix_random_forest_<subset>.png`

To regenerate tuning plots from an existing tuning CSV, run:

```bash
python code/random_forest/generate_random_forest_tuning_graphs.py \
  --csv code/random_forest/results/random_forest_tuning_results_medium.csv
```

By default, regenerated tuning plots are saved under `code/random_forest/results/tuning_graphs_<subset>/`.

Use the small CSV instead if you trained the small subset:

```bash
python code/random_forest/generate_random_forest_tuning_graphs.py \
  --csv code/random_forest/results/random_forest_tuning_results_small.csv
```

## MLP And XGBoost

The MLP and XGBoost scripts read the shared tabular outputs from `fma_preprocessed/`; they do not need their own preprocessing step.

Required files for each subset are:

- `tracks_clean_<subset>_training.csv`
- `tracks_clean_<subset>_validation.csv`
- `tracks_clean_<subset>_test.csv`
- `genre_to_idx_<subset>.csv`
- `features_<subset>.csv`

Generate those files by running `code/data_preprocessing.ipynb` with `DATASET_SIZE` set to the subset you need and `PREPROCESS_FOR=none` to skip spectrogram generation.

At the top of each script, set which dataset to run:

```python
SUBSET = "medium"  # or "small"
```

Then run:

```bash
python code/MLP/MLP.py
python code/XGBoost/XGBoost.py
```

MLP outputs are saved to `code/MLP/results/`, including:

- `mlp_<subset>_search_results.csv`
- `mlp_<subset>_search_diagnostics.png`
- `mlp_<subset>_search_ranked_trials.png`
- `mlp_<subset>_final_seeds.csv`
- `mlp_<subset>_final_curves.png`
- `mlp_<subset>_final_confusion_matrix.png`
- `mlp_<subset>_summary.json`

XGBoost outputs are saved to `code/XGBoost/results/`, including:

- `xgb_<subset>_final_seeds.csv`
- `xgb_<subset>_final_curves.png`
- `xgb_<subset>_final_confusion_matrix.png`
- `xgb_<subset>_feature_importance.png`
- `xgb_<subset>_summary.json`

## CNN

Open and run:

```text
code/cnn/cnn.ipynb
```

This notebook trains a deeper CNN with batch normalization and global average pooling from preprocessed spectrograms.

At the top of the notebook, set which dataset to run:

```python
DATASET_SIZE = "medium"  # or "small"
```

The notebook reads `PREPROCESSED_DIR` from your `.env`; if it is not set it defaults to `DATASET_DIR/fma_preprocessed`.

The CNN notebook supports these optional runtime settings (set in `.env` or as shell exports before launching Jupyter):

```bash
export BATCH_SIZE=32
export NUM_EPOCHS=30
export LR=0.001
export DROPOUT=0.4
export PATIENCE=7
export NUM_WORKERS=4
```

Important outputs are written to `code/cnn/results_cnn_<subset>/` (e.g. `results_cnn_medium/`), including:

- `best_model.pt`
- `final_model.pt`
- `training_history.csv`
- `accuracy.png`
- `loss.png`
- `classification_report.txt`
- `confusion_matrix.png`
- `results.json`

There is also an older automated CNN experiment script:

```bash
python code/cnn/train_cnn_experiments.py
```

That script extracts its own cached log-mel dataset and runs several CNN experiments. It may require extra TensorFlow/Keras setup depending on your environment.
Its experiment outputs are saved to `code/cnn/results_cnn_experiments/`.

## CRNN

Two CRNN notebooks are available:

| Notebook | Description |
|---|---|
| `code/crnn/crnn_basic.ipynb` | Baseline CRNN — random-crops a fixed window from the full 20-second spectrogram each epoch |
| `code/crnn/crnn_divide_and_conquer.ipynb` | Divide-and-conquer CRNN — splits each spectrogram into overlapping snippets and aggregates predictions per track |

Open and run either notebook. Before running all cells, choose the dataset size near the top:

```python
SUBSET = "small"  # or "medium"
```

Both notebooks read `DATASET_DIR` from your `.env` file via `python-dotenv`, so they work regardless of which directory Jupyter is launched from.

Each CRNN notebook:

- loads spectrogram manifests from `fma_preprocessed/`
- trains a convolutional recurrent neural network with PyTorch
- can run Optuna hyperparameter tuning
- saves curves, reports, checkpoints, and comparison summaries under `code/crnn/results/<RUN_NAME>/`

The main CRNN settings are defined near the top of the notebook:

```python
BATCH_SIZE = 64
LR = 5e-4
MAX_EPOCHS = 100
SPEC_AUG = True
RAND_CROP = True
CROP_FRAMES = 256
CNN_MIN_FILTER = 32
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
DROPOUT = 0.1
WEIGHT_DECAY = 1e-5
RUN_NAME = "spec_aug"
```

Outputs for each run include:

- `best_crnn.pth`
- `accuracy_curve.png`
- `loss_curve.png`
- `confusion_matrix.png`

If Optuna is not installed after setup, install it in the active environment:

```bash
pip install optuna
```

## Repository Notes

Large datasets, generated spectrograms, model checkpoints, and most CSV outputs are ignored by Git. If a model complains that files are missing, rerun setup and then rerun the preprocessing notebook.

## Running on MSCluster

On MSCluster, set `DATASET_DIR` in `.env` to your project directory, e.g.:

```bash
DATASET_DIR=/home/<user>/acml-project
```

**Terminal 1 — on the cluster:**

Allocate an interactive shell on a compute node:

```bash
srun -N1 --ntasks=16 -p bigbatch --pty bash
```

Navigate to the project, activate the environment, and start Jupyter from the `code/` directory:

```bash
cd acml-project/
source .venv/bin/activate
cd code/
jupyter lab --no-browser --port=8888
```

**Terminal 2 — on your local machine:**

Once the compute node is allocated (e.g. `mscluster21`), open an SSH tunnel through the login node to that specific node:

```bash
ssh -J <user>@<mscluster-ip> -N -L 8888:localhost:8888 <user>@mscluster<node>
```

Replace `<user>` with your username and `mscluster<node>` with whichever node was allocated. Then open:

```text
http://localhost:8888
```
