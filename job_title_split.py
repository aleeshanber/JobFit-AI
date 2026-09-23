"""Job-title-level train/test split for JobFit AI (Step 9 methodology).

The ML train/test methodology uses a JOB-TITLE-LEVEL holdout: 20 unique job
titles train the model, the other 8 job titles are held out and NEVER appear
in training, validation, preprocessing fitting, or model fitting.

Everything here is deterministic (random_state=42) and the exact 20/8 title
split is persisted permanently to ARTIFACTS/job_title_split.json so the split
is reproducible and inspectable (`train_job_titles` / `test_job_titles`).

This module only computes/stores/validates the title split. It does not train
anything and never reads matched_score as a feature.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "ARTIFACTS"
SPLIT_MANIFEST = ARTIFACTS_DIR / "job_title_split.json"
CLEAN_CSV = ROOT / "DATA" / "cleaned_jobs.csv"

RANDOM_STATE = 42
TRAIN_COUNT = 20
TEST_COUNT = 8


def load_job_titles(csv_path=CLEAN_CSV):
    """Unique, sorted job titles from the cleaned dataset (the 28-title catalog)."""
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[])
    return sorted(df["job_title"].unique()), df


def make_job_title_split(seed=RANDOM_STATE, test_count=TEST_COUNT):
    """Deterministic 20/8 job-title split.

    Title-level split (not row-level, not candidate-level): entire job titles
    are assigned to one side. Deterministic under np.random.RandomState(seed).
    Returns (train_titles, test_titles), both sorted lists.
    """
    titles, _ = load_job_titles()
    titles_arr = np.array(titles)
    rng = np.random.RandomState(seed)
    shuffled = titles_arr.copy()
    rng.shuffle(shuffled)
    test_titles = sorted(shuffled[:test_count].tolist())
    train_titles = sorted(shuffled[test_count:].tolist())
    return train_titles, test_titles


def validate_job_title_split(train_titles, test_titles, df=None):
    """The 7 mandatory split assertions; raise ValueError with a clear message.

    1. exactly 20 train titles
    2. exactly 8 test titles
    3. union covers all 28 dataset titles
    4. train/test intersection is empty
    5. every unique title is in exactly one side
    6. no train-title row appears in the test identity check (via sets)
    7. no test-title row appears in the train identity check (via sets)
    """
    errors = []
    train_titles = sorted(train_titles)
    test_titles = sorted(test_titles)
    all_titles_sorted = sorted(set(train_titles) | set(test_titles))

    if len(train_titles) != TRAIN_COUNT:
        errors.append(f"train titles count {len(train_titles)} != {TRAIN_COUNT}")
    if len(test_titles) != TEST_COUNT:
        errors.append(f"test titles count {len(test_titles)} != {TEST_COUNT}")

    overlap = sorted(set(train_titles) & set(test_titles))
    if overlap:
        errors.append(f"train/test title overlap: {overlap}")

    if df is None:
        df_titles = all_titles_sorted
    else:
        df_titles = sorted(df["job_title"].unique())

    if all_titles_sorted != df_titles:
        missing = sorted(set(df_titles) - set(all_titles_sorted))
        extra = sorted(set(all_titles_sorted) - set(df_titles))
        errors.append(
            f"split titles != dataset titles (missing={missing}, extra={extra})")

    if df is not None:
        train_row_leak = df[df["job_title"].isin(train_titles)].shape[0] == 0 and \
            len(train_titles) > 0
        test_row_leak = df[df["job_title"].isin(test_titles)].shape[0] == 0 and \
            len(test_titles) > 0
        # A side is only valid if its titles actually have rows.
        if train_row_leak:
            errors.append("no training-title rows present in dataset")
        if test_row_leak:
            errors.append("no test-title rows present in dataset")

    if errors:
        raise ValueError("Job-title split validation failed:\n- " + "\n- ".join(errors))
    return True


def get_train_test_indices(df, train_titles, test_titles):
    """Row index arrays for the two sides of a title-level split."""
    tr_idx = df.index[df["job_title"].isin(train_titles)].to_numpy()
    te_idx = df.index[df["job_title"].isin(test_titles)].to_numpy()
    return tr_idx, te_idx


def save_split_manifest(manifest, manifest_path=SPLIT_MANIFEST):
    """Persist the manifest; returns the dict written."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


def build_split_manifest(df, train_titles, test_titles, seed=RANDOM_STATE):
    """Manifest dict with exact titles + row counts for the permanent record."""
    tr_idx, te_idx = get_train_test_indices(df, train_titles, test_titles)
    return {
        "random_state": seed,
        "total_unique_titles": int(df["job_title"].nunique()),
        "train_count": int(len(train_titles)),
        "test_count": int(len(test_titles)),
        "train_rows": int(len(tr_idx)),
        "test_rows": int(len(te_idx)),
        "train_job_titles": sorted(train_titles),
        "test_job_titles": sorted(test_titles),
    }


def load_split_manifest(manifest_path=SPLIT_MANIFEST):
    """Load the saved manifest; None when it does not exist."""
    if not Path(manifest_path).exists():
        return None
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def get_split(df, save=True, manifest_path=SPLIT_MANIFEST):
    """Deterministic canonical split: manifest if saved, else compute+save.

    Returns (train_titles, test_titles, manifest). The manifest is validated
    against the dataset before reuse so an out-of-date manifest is caught.
    """
    existing = load_split_manifest(manifest_path)
    if existing is not None:
        train_titles = sorted(existing["train_job_titles"])
        test_titles = sorted(existing["test_job_titles"])
        validate_job_title_split(train_titles, test_titles, df)
        manifest = existing
        manifest["train_rows"] = int(
            df["job_title"].isin(train_titles).sum())
        manifest["test_rows"] = int(
            df["job_title"].isin(test_titles).sum())
    else:
        train_titles, test_titles = make_job_title_split()
        validate_job_title_split(train_titles, test_titles, df)
        manifest = build_split_manifest(df, train_titles, test_titles)
        if save:
            save_split_manifest(manifest, manifest_path)
    return train_titles, test_titles, manifest


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_CSV, keep_default_na=False, na_values=[])
    train_titles, test_titles, manifest = get_split(df, save=True)
    print("job-title split (seed=42) — 20 train / 8 test")
    print("TRAIN titles:", train_titles)
    print("TEST titles:", test_titles)
    print("train_rows", manifest["train_rows"], "| test_rows",
          manifest["test_rows"])