
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. RESOLVE PATHS
# ─────────────────────────────────────────────

PROJECT_CANDIDATES = [Path.cwd(), Path.cwd().parent]
for candidate in PROJECT_CANDIDATES:
    metadata_path = candidate / "fma" / "data" / "fma_metadata" / "tracks.csv"
    features_path = candidate / "fma" / "data" / "fma_metadata" / "features.csv"
    if metadata_path.exists() and features_path.exists():
        PROJECT_DIR   = candidate.resolve()
        METADATA_PATH = metadata_path.resolve()
        FEATURES_PATH = features_path.resolve()
        break
else:
    raise FileNotFoundError(
        "Could not find fma_metadata/tracks.csv and fma_metadata/features.csv. "
        "Run this script from your project root."
    )

PROCESSED_DIR = PROJECT_DIR / "fma_preprocessed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Detect which audio directories are present
AVAILABLE_AUDIO = {}
for size in ("fma_small", "fma_medium"):
    audio_dir = PROJECT_DIR / "fma" / "data" / size
    if audio_dir.exists():
        AVAILABLE_AUDIO[size] = audio_dir.resolve()

print(f"Project dir  : {PROJECT_DIR}")
print(f"Metadata     : {METADATA_PATH}")
print(f"Features     : {FEATURES_PATH}")
print(f"Output dir   : {PROCESSED_DIR}")
print(f"Audio dirs   : {list(AVAILABLE_AUDIO.keys()) or 'none'}")

if not AVAILABLE_AUDIO:
    raise FileNotFoundError(
        "Neither fma_small/ nor fma_medium/ found under fma/data/. "
        "Download at least one subset before running this script."
    )

# ─────────────────────────────────────────────
# 2. SUBSET CONFIG
# ─────────────────────────────────────────────
SUBSET_CONFIG = {
    "small": {
        "subset_values":  ["small"],
        "audio_dir_name": "fma_small",
    },
    "medium": {
        "subset_values":  ["small", "medium"],   # medium is a superset of small
        "audio_dir_name": "fma_medium",
    },
}

# fma_medium contains every track that fma_small does. So if the user
# downloaded fma_medium but not fma_small, we can still produce the small
# output by reading those tracks from fma_medium.
if "fma_small" not in AVAILABLE_AUDIO and "fma_medium" in AVAILABLE_AUDIO:
    SUBSET_CONFIG["small"]["audio_dir_name"] = "fma_medium"

# ─────────────────────────────────────────────
# 3. TRACK IDS KNOWN TO BE BROKEN
#    (from creation.ipynb — ffmpeg / header / Nyquist errors)
# ─────────────────────────────────────────────

FAILED = [
    1440, 26436, 28106, 29166, 29167, 29168, 29169, 29170, 29171, 29172,
    29173, 29179, 38903, 43903, 56757, 57603, 59361, 62095, 62954, 62956,
    62957, 62959, 62971, 75461, 80015, 86079, 92345, 92346, 92347, 92348,
    92349, 92350, 92351, 92352, 92353, 92354, 92355, 92356, 92357, 92358,
    92359, 92360, 92361, 96426, 104623, 106719, 109714, 114448, 114501, 114528,
    115235, 117759, 118003, 118004, 127827, 130296, 130298, 131076, 135804, 136486,
    144769, 144770, 144771, 144773, 144774, 144775, 144776, 144777, 144778, 152204,
    154923,
]

# ─────────────────────────────────────────────
# 4. HELPERS
# ─────────────────────────────────────────────

def is_file_nonempty(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False

def keep(mask, df, label):
    old = len(df)
    df  = df[mask]
    print(f"  {label:45s} {old - len(df):>5d} dropped, {len(df):>5d} left")
    return df

def track_id_to_audio_path(track_id: int, audio_dir: Path) -> Path:
    filename = f"{int(track_id):06d}.mp3"
    return audio_dir / filename[:3] / filename

# ─────────────────────────────────────────────
# 5. LOAD tracks.csv AND features.csv (ONCE)
# ─────────────────────────────────────────────

print("\nLoading tracks.csv ...")
tracks      = pd.read_csv(METADATA_PATH, index_col=0, header=[0, 1])
tracks_flat = tracks.reset_index(drop=True)

metadata_all = pd.DataFrame({
    "track_id" : tracks.index.astype(int).to_numpy(),
    "subset"   : tracks_flat[("set",   "subset")    ].to_numpy(),
    "split"    : tracks_flat[("set",   "split")     ].to_numpy(),
    "genre"    : tracks_flat[("track", "genre_top") ].to_numpy(),
    "duration" : tracks_flat[("track", "duration")  ].to_numpy(),
    "bit_rate" : tracks_flat[("track", "bit_rate")  ].to_numpy(),
    "title"    : tracks_flat[("track", "title")     ].to_numpy(),
    "artist"   : tracks_flat[("artist","name")      ].to_numpy(),
})

print(f"Total tracks in tracks.csv: {len(metadata_all):,}")
print(metadata_all["subset"].value_counts())

print("\nLoading features.csv ...")
features_all = pd.read_csv(FEATURES_PATH, index_col=0, header=[0, 1, 2])
print(f"Features shape: {features_all.shape}")

# ─────────────────────────────────────────────
# 6. SAVE HELPER
# ─────────────────────────────────────────────

def save_clean_csvs(frame, subset_name, genre_to_idx):
    base_path = PROCESSED_DIR / f"tracks_clean_{subset_name}.csv"
    frame.to_csv(base_path, index=False)
    print(f"  Wrote {base_path.name} ({len(frame):,} rows)")

    for split_name in ("training", "validation", "test"):
        split_df   = frame[frame["split"] == split_name]
        split_path = PROCESSED_DIR / f"tracks_clean_{subset_name}_{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        print(f"  Wrote {split_path.name} ({len(split_df):,} rows)")

    genre_map_path = PROCESSED_DIR / f"genre_to_idx_{subset_name}.csv"
    pd.DataFrame(
        sorted(genre_to_idx.items(), key=lambda kv: kv[1]),
        columns=["genre", "label"],
    ).to_csv(genre_map_path, index=False)
    print(f"  Wrote {genre_map_path.name}")

# ─────────────────────────────────────────────
# 7. PROCESS EACH SUBSET
# ─────────────────────────────────────────────

for subset_name, cfg in SUBSET_CONFIG.items():
    audio_dir_name = cfg["audio_dir_name"]
    if audio_dir_name not in AVAILABLE_AUDIO:
        print(f"\n=== Skipping {subset_name} subset ({audio_dir_name}/ not on disk) ===")
        continue

    audio_dir = AVAILABLE_AUDIO[audio_dir_name]
    print(f"\n=== Cleaning {subset_name} subset (audio from {audio_dir_name}/) ===")

    metadata = metadata_all.copy()
    metadata = keep(metadata["subset"].isin(cfg["subset_values"]), metadata,
                    f"subset in {cfg['subset_values']}")
    metadata = keep(metadata["genre"].notna(), metadata, "genre not null")

    metadata["audio_path"] = metadata["track_id"].apply(
        lambda tid: track_id_to_audio_path(tid, audio_dir)
    )
    metadata = keep(metadata["audio_path"].map(is_file_nonempty), metadata,
                    "audio file present and non-empty")
    metadata = keep(~metadata["track_id"].isin(FAILED), metadata,
                    "not in FAILED list")

    metadata = metadata.sort_values("track_id").reset_index(drop=True)

    # Encode genre labels (alphabetical → consistent ordering)
    genres_sorted     = sorted(metadata["genre"].unique())
    genre_to_idx      = {g: i for i, g in enumerate(genres_sorted)}
    metadata["label"] = metadata["genre"].map(genre_to_idx).astype(int)

    print(f"\nGenre distribution ({subset_name}, {len(genres_sorted)} classes):")
    print(metadata["genre"].value_counts().sort_index())
    print(f"\nSplit distribution ({subset_name}):")
    print(metadata["split"].value_counts())

    # Align features
    valid_ids       = metadata["track_id"].values
    subset_features = features_all.loc[features_all.index.isin(valid_ids)].fillna(0)
    print(f"\nFeatures shape after alignment: {subset_features.shape}")

    # Save
    print(f"\nSaving {subset_name} CSVs ...")
    save_clean_csvs(metadata, subset_name, genre_to_idx)
    features_out = PROCESSED_DIR / f"features_{subset_name}.csv"
    subset_features.to_csv(features_out)
    print(f"  Wrote {features_out.name} "
          f"({subset_features.shape[0]:,} rows, {subset_features.shape[1]:,} features)")

print("\nPreprocessing complete.")
