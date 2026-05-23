# ACML FMA Music Classification

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
|-- .env.example               Template for local paths and preprocessing settings
|-- code/
|   |-- data_preprocessing.ipynb
|   |-- random_forest/         Random Forest notebook, tuning helper, and outputs
|   |-- cnn/                   CNN notebook and older automated CNN experiment script
|   |-- crnn/                  CRNN notebooks and experiment outputs
|   |-- MLP/                   MLP scripts
|   `-- XGBoost/               XGBoost scripts
```

Before running `setup.sh`, create `.env` from `.env.example` and set your local paths.

## Setup

Create your local environment file first:

```bash
cp .env.example .env
```

Open `.env` and set `DATASET_DIR` to the folder where the FMA data should live. Set `DATASET_SIZE=small`, `medium`, or `both`. Setup always generates the cleaned metadata CSVs and tabular feature CSVs used by Random Forest, MLP, and XGBoost. Set `DOWNLOAD_SPECTROGRAMS=1` only if setup should also generate spectrogram `.npy` files for CNN/CRNN. This file is local to your machine, so do not commit personal paths or secrets from it.

From the project root, run:

```bash
bash setup.sh
```

This script:

- creates a local Python virtual environment in `.venv/`
- installs the main Python dependencies
- installs PyTorch and torchaudio
- downloads and extracts `fma_small/`
- downloads and extracts `fma_metadata/`
- downloads and extracts `fma_medium/` when `DATASET_SIZE=both`, `DOWNLOAD_MEDIUM=1`, or medium spectrograms are requested
- generates cleaned CSVs and feature CSVs
- generates spectrograms when `DOWNLOAD_SPECTROGRAMS=1`

### Folder Structure After Setup

After `setup.sh` finishes, your project should look like this:

```text
.
|-- README.md
|-- setup.sh
|-- .env
|-- .env.example
|-- .venv/                     Local Python environment created by setup
|-- code/                      Notebooks and model code
|-- fma_small/                 Downloaded FMA small audio files
|-- fma_metadata/              Downloaded FMA metadata CSV files
|-- fma_medium/                Downloaded only when medium audio is requested
`-- fma_preprocessed/          Cleaned CSVs/features, plus optional spectrograms
```

If you set `DATASET_DIR` to another location, `fma_small/`, `fma_metadata/`, and optional `fma_medium/` will be created there instead of inside the repository. The downloaded zip files are removed after extraction.

After setup finishes, activate the environment:

```bash
source .venv/bin/activate
```

If you need to force CPU-only PyTorch, run setup as:

```bash
PYTORCH_BUILD=cpu bash setup.sh
```

If your dataset is stored somewhere outside the repository, set `DATASET_DIR` before running the notebooks:

```bash
export DATASET_DIR=/path/to/dataset/root
```

You can also set the same value in `.env`; `setup.sh` reads it automatically.

That directory should contain `fma_small/` and `fma_metadata/`.

## Data Preprocessing

`setup.sh` runs the shared preprocessing script automatically after the downloads and environment setup:

```bash
code/data_preprocessing.py
```

When `DOWNLOAD_SPECTROGRAMS=0`, setup calls it with `PREPROCESS_FOR=none`, so it only writes the cleaned CSVs and tabular feature CSVs. You can also run it manually:

```bash
DATASET_SIZE=small PREPROCESS_FOR=none python code/data_preprocessing.py
```

Use `DATASET_SIZE=both` if you want both `small` and `medium` CSV outputs. You can also run the notebook version manually. Start Jupyter from the project root:

```bash
jupyter notebook
```

Open and run:

```text
code/data_preprocessing.ipynb
```

The preprocessing step creates `fma_preprocessed/`, including:

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
- `spectrograms_medium_10_manifest.csv` when `DATASET_SIZE=medium`/`both` and `PREPROCESS_FOR=cnn` or `both`
- `spectrograms_small_20_manifest.csv` when `PREPROCESS_FOR=crnn` or `both`
- `spectrograms_medium_20_manifest.csv` when `DATASET_SIZE=medium`/`both` and `PREPROCESS_FOR=crnn` or `both`
- `spectrograms_<size>_<10|20>/`

Run preprocessing before running any model. Random Forest, MLP, and XGBoost use the cleaned CSV files plus `features_<subset>.csv`. The CNN and CRNN use the saved spectrogram manifests and `.npy` spectrogram files.

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
- saves the best model checkpoint and evaluation outputs in `code/random_forest/`

Important outputs include:

- `code/random_forest/best_random_forest_<subset>.pkl`
- `code/random_forest/random_forest_tuning_results_<subset>.csv`
- `code/random_forest/random_forest_tuning_results_<subset>.jsonl`
- `code/random_forest/confusion_matrix_random_forest_<subset>.png`

To regenerate tuning plots from an existing tuning CSV, run:

```bash
python code/random_forest/generate_random_forest_tuning_graphs.py \
  --csv code/random_forest/random_forest_tuning_results_medium.csv
```

Use the small CSV instead if you trained the small subset:

```bash
python code/random_forest/generate_random_forest_tuning_graphs.py \
  --csv code/random_forest/random_forest_tuning_results_small.csv
```

## MLP And XGBoost

The MLP and XGBoost scripts read the shared tabular outputs from `fma_preprocessed/`; they do not need their own preprocessing step.

Required files for each subset are:

- `tracks_clean_<subset>_training.csv`
- `tracks_clean_<subset>_validation.csv`
- `tracks_clean_<subset>_test.csv`
- `genre_to_idx_<subset>.csv`
- `features_<subset>.csv`

Generate those files with:

```bash
DATASET_SIZE=both PREPROCESS_FOR=none python code/data_preprocessing.py
```

Then choose `SUBSET = "small"` or `SUBSET = "medium"` near the top of each script and run:

```bash
python code/MLP/MLP.py
python code/XGBoost/XGBoost.py
```

## CNN

Open and run:

```text
code/cnn/cnn.ipynb
```

This notebook trains a deeper CNN with batch normalization and global average pooling from preprocessed spectrograms.

By default, the notebook expects a medium spectrogram preprocessing folder:

```text
fma_preprocessed_medium/
```

If you want to point it at another preprocessing output, set these environment variables before starting Jupyter:

```bash
export FMA_PROJECT_DIR=/path/to/project/root
export PROCESSED_DIR=/path/to/preprocessed/folder
export MANIFEST_PATH=/path/to/preprocessed/folder/spectrograms_manifest.csv
export RESULTS_DIR=/path/to/cnn/results
```

For example, to run against the default small preprocessing output from `code/data_preprocessing.ipynb`, use:

```bash
export FMA_PROJECT_DIR=$(pwd)
export PROCESSED_DIR=$(pwd)/fma_preprocessed
export MANIFEST_PATH=$(pwd)/fma_preprocessed/spectrograms_manifest.csv
export RESULTS_DIR=$(pwd)/results_cnn
jupyter notebook
```

The CNN notebook supports these optional runtime settings:

```bash
export BATCH_SIZE=32
export NUM_EPOCHS=30
export LR=0.001
export DROPOUT=0.4
export PATIENCE=7
export NUM_WORKERS=4
```

Important outputs are written to `RESULTS_DIR`, including:

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

## CRNN

Open and run:

```text
code/crnn/crnn.ipynb
```

Before running all cells, choose the dataset size near the top of the notebook:

```python
SUBSET = "small"  # or "medium"
```

The CRNN notebook:

- loads spectrogram manifests from `fma_preprocessed/`
- trains a convolutional recurrent neural network with PyTorch
- can run Optuna hyperparameter tuning
- saves curves, reports, checkpoints, and comparison summaries

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

Important outputs are written under:

```text
results/<RUN_NAME>/
```

The notebook also appends summary metrics to:

```text
results/comparison_summary.csv
```

If Optuna is not installed after setup, install it in the active environment:

```bash
pip install optuna
```

## Repository Notes

Large datasets, generated spectrograms, model checkpoints, and most CSV outputs are ignored by Git. If a model complains that files are missing, rerun setup and then rerun the preprocessing notebook.

## Running on MSCluster

On MSCluster, set `DATASET_DIR` in `.env` to your project directory:

```bash
DATASET_DIR=/home/<user>/acml-project
```

Allocate an interactive shell:

```bash
srun -N1 --ntasks=16 -p bigbatch --pty bash
```

From the project root, load the `.env` values and activate the Python environment:

```bash
set -a
source .env
set +a
source .venv/bin/activate
```

To run Jupyter on the cluster and open it in your browser, start Jupyter on the cluster:

```bash
jupyter lab --no-browser --port=8888
```

In a separate terminal on your local machine, create an SSH tunnel:

```bash
ssh -L 8888:localhost:8888 username@remote-server
```

Then open:

```text
http://localhost:8888
```
