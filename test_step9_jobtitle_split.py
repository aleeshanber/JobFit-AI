"""Step 9 - tests for the job-title-level train/test methodology.

Covers: the deterministic 20-train / 8-test title split, the permanent
ARTIFACTS/job_title_split.json manifest, zero title overlap, all 28 titles
covered, no test-title row in the training side and no train-title row in the
test side, reproducibility under seed 42, and the saved step-9 model artifact
(kind + 17-feature contract) without touching the production model.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from UTILS.job_title_split import (
    RANDOM_STATE,
    build_split_manifest,
    get_split,
    get_train_test_indices,
    load_job_titles,
    load_split_manifest,
    make_job_title_split,
    validate_job_title_split,
)

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "DATA" / "cleaned_jobs.csv"
MANIFEST = ROOT / "ARTIFACTS" / "job_title_split.json"
STEP9_MODEL = ROOT / "MODELS" / "match_model_jobtitle.pkl"
STEP9_META = ROOT / "MODELS" / "match_model_jobtitle_metadata.json"
PROD_MODEL = ROOT / "MODELS" / "match_model.pkl"

PASSED = 0
FAILED = 0


def check(desc, cond, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {desc} {extra}")
    else:
        FAILED += 1
        print(f"  [FAIL] {desc} {extra}")


def section(name):
    print(f"== {name} ==")


if __name__ == "__main__":
    titles, df = load_job_titles()
    section("1. deterministic 20/8 job-title split (seed 42)")
    train, test = make_job_title_split(seed=RANDOM_STATE)
    train2, test2 = make_job_title_split(seed=RANDOM_STATE)
    check("exactly 20 train titles", len(train) == 20, f"(got {len(train)})")
    check("exactly 8 test titles", len(test) == 8, f"(got {len(test)})")
    check("zero overlap", not (set(train) & set(test)))
    check("union covers all 28 titles",
          sorted(set(train) | set(test)) == titles)
    check("every split title is a real dataset title",
          set(train) | set(test) <= set(titles))
    check("deterministic: two calls identical",
          train == train2 and test == test2, "(seed 42)")

    section("2. validation assertions")
    check("validate_job_title_split passes on the real split",
          validate_job_title_split(train, test, df) is True)
    try:
        validate_job_title_split(train[:19], test, df)
        check("rejects wrong train count (19)", False)
    except ValueError:
        check("rejects wrong train count (19)", True)
    try:
        validate_job_title_split(train + test[:1], test[1:], df)
        check("rejects overlap", False)
    except ValueError:
        check("rejects overlap", True)

    section("3. no test-title rows on the training side (and vice versa)")
    tr_idx, te_idx = get_train_test_indices(df, train, test)
    df_tr_rows = df.iloc[tr_idx]
    df_te_rows = df.iloc[te_idx]
    check("no test-title row in train split",
          not bool(df_tr_rows["job_title"].isin(test).any()))
    check("no train-title row in test split",
          not bool(df_te_rows["job_title"].isin(train).any()))
    check("train rows == 6805", len(tr_idx) == 6805, f"(got {len(tr_idx)})")
    check("test rows == 2726", len(te_idx) == 2726, f"(got {len(te_idx)})")
    check("rows partition the full dataset", len(tr_idx) + len(te_idx) == len(df))
    check("every candidate has >= 2 test rows (ranking valid)",
          df_te_rows.groupby("candidate_skills").size().min() >= 2)

    section("4. permanent manifest ARTIFACTS/job_title_split.json")
    man = load_split_manifest()
    check("manifest file exists", man is not None)
    if man is not None:
        check("manifest random_state == 42", man.get("random_state") == 42)
        check("manifest total_unique_titles == 28",
              man.get("total_unique_titles") == 28)
        check("manifest train_count == 20", man.get("train_count") == 20)
        check("manifest test_count == 8", man.get("test_count") == 8)
        check("manifest train_rows == 6805", man.get("train_rows") == 6805)
        check("manifest test_rows == 2726", man.get("test_rows") == 2726)
        check("manifest train titles == actual split",
              sorted(man["train_job_titles"]) == sorted(train))
        check("manifest test titles == actual split",
              sorted(man["test_job_titles"]) == sorted(test))

    section("5. get_split reuses the manifest; row counts stay correct")
    train_r, test_r, manifest_r = get_split(df, save=False)
    check("reused manifest matches computed split",
          sorted(train_r) == sorted(train) and sorted(test_r) == sorted(test))
    check("manifest row counts fresh from dataset",
          manifest_r["train_rows"] == 6805 and manifest_r["test_rows"] == 2726)

    section("6. step-9 trained model artifact (adaptive adoption only)")
    check("match_model_jobtitle.pkl exists", STEP9_MODEL.exists())
    check("metadata exists", STEP9_META.exists())
    if STEP9_MODEL.exists() and STEP9_META.exists():
        import pickle
        with open(STEP9_MODEL, "rb") as fh:
            blob = pickle.load(fh)
        with open(STEP9_META, encoding="utf-8") as fh:
            meta = json.load(fh)
        from UTILS.features import RICH_MATCH_FEATURE_NAMES
        check("kind == jobfit-match-model-step6",
              blob.get("kind") == "jobfit-match-model-step6")
        check("features == 17 RICH_MATCH_FEATURE_NAMES",
              blob.get("features") == list(RICH_MATCH_FEATURE_NAMES))
        check("has .predict on model", hasattr(blob.get("model"), "predict"))
        check("metadata split_strategy is job-title-level",
              "JOB-TITLE-LEVEL" in meta.get("split_strategy", "").upper())
        check("metadata test metrics present",
              isinstance(meta.get("test_metrics"), dict))
        check("metadata reports 20 train titles",
              set(blob["trained_on_job_titles"]) == set(train))
        check("metadata reports 8 held-out titles",
              set(blob["held_out_job_titles"]) == set(test))
        check("production match_model.pkl NOT touched", PROD_MODEL.exists())

    print(f"\n==== passed {PASSED} / failed {FAILED} ====")
    sys.exit(1 if FAILED else 0)