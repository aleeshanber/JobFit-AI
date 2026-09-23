"""Step 9 (methodology) - job-title-level train/test for the matching model.

Re-trains the matching model using a JOB-TITLE-LEVEL holdout instead of the
historical candidate-signature 70/15/15 split:

  * 20 unique job titles train the model (all their rows).
  * 8 unique job titles are held out completely: none of their rows touch
    training, validation, preprocessing fitting (TF-IDF), or model fitting.
  * The split is deterministic (seed 42) and persisted to
    ARTIFACTS/job_title_split.json.

No leakage rules enforced here:
  * split happens BEFORE any learned preprocessing;
  * the shared TF-IDF skill space is fit ONLY on the 20 training job titles;
  * evaluation of the 8 held-out titles happens exactly once.

Artifacts:
  * MODELS/match_model_jobtitle.pkl      - kind=jobfit-match-model-step6, same
    17-feature RICH_MATCH_FEATURE_NAMES contract (ready to adopt in the app;
    the production MODELS/match_model.pkl is left untouched).
  * MODELS/match_model_jobtitle_metadata.json - full train/test metrics + split.
  * ARTIFACTS/job_title_split.json       - permanent 20/8 title manifest.
"""

import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from UTILS.features import (
    MATCH_FEATURE_NAMES,
    RICH_MATCH_FEATURE_NAMES,
    build_skills_tfidf,
    candidate_job_match_features,
    candidate_job_match_features_rich,
    load_candidate_profiles,
)
from UTILS.job_title_split import (
    RANDOM_STATE,
    build_split_manifest,
    get_split,
    validate_job_title_split,
)
from UTILS.train_final_model import candidate_ranking_metrics, ndcg_at_k
from UTILS.train_match_model import group_key

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "DATA" / "cleaned_jobs.csv"
RAW_CSV = ROOT / "DATA" / "resume_data.csv"
OUTPUT_MODEL = ROOT / "MODELS" / "match_model_jobtitle.pkl"
OUTPUT_META = ROOT / "MODELS" / "match_model_jobtitle_metadata.json"
SEED = RANDOM_STATE
K = 10

PASS = []
FAIL = []


def check(desc, cond, extra=""):
    tag = "PASS" if cond else "FAIL"
    PASS.append(desc) if cond else FAIL.append(desc)
    print(f"  [{tag}] {desc} {extra}")
    return cond


def job_corpus(df):
    """One non-empty required-skills document per job title."""
    docs, seen = [], set()
    for _, r in df.sort_values("job_title").iterrows():
        if r["job_title"] in seen or not str(r["required_skills"]).strip():
            continue
        seen.add(r["job_title"])
        docs.append(r["required_skills"])
    return docs


def eval_regression(y_true, pred, kind):
    pred = np.asarray(pred, dtype=float)
    return {
        "kind": kind,
        "n_rows": int(len(y_true)),
        "MAE": round(float(mean_absolute_error(y_true, pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, pred))), 4),
        "R2": round(float(r2_score(y_true, pred)), 4),
        "pred_min": round(float(np.min(pred)), 4),
        "pred_max": round(float(np.max(pred)), 4),
        "pred_mean": round(float(np.mean(pred)), 4),
        "pred_nan": int(np.isnan(pred).sum()),
        "pred_inf": int(np.isinf(pred).sum()),
        "preds_outside_0_1": int(((pred < 0) | (pred > 1)).sum()),
    }


def eval_ranking(df, idx, pred, kind):
    """Per-candidate ranking metrics on the given rows (dedup (grp,title))."""
    groups = defaultdict(lambda: {"true": [], "pred": []})
    seen = set()
    y = df["_y"].to_numpy()
    pred = np.asarray(pred, dtype=float)
    for pos, i in enumerate(idx):
        key = (df["_grp"].iloc[i], df["job_title"].iloc[i])
        if key in seen:
            continue
        seen.add(key)
        g = df["_grp"].iloc[i]
        groups[g]["true"].append(y[i])
        groups[g]["pred"].append(pred[pos])
    metrics = candidate_ranking_metrics(groups, k=K)
    metrics["kind"] = kind
    metrics["n_rows"] = int(len(idx))
    return metrics


def global_mean_baseline(y_train):
    return float(np.mean(y_train))


def run(seed=SEED):
    t0 = time.time()
    print("=" * 82)
    print("STEP 9 METHODOLOGY - JOB-TITLE-LEVEL TRAIN/TEST (20 train / 8 test)")
    print("=" * 82)

    df = pd.read_csv(CLEAN_CSV, keep_default_na=False, na_values=[])
    df["_y"] = df["matched_score"].astype(float)
    df["_grp"] = df["candidate_skills"].map(group_key)
    profiles = load_candidate_profiles(RAW_CSV)

    print("\n=== A. deterministic job-title split (seed 42) ===")
    train_titles, test_titles, manifest = get_split(df, save=True)
    validate_job_title_split(train_titles, test_titles, df)
    tr_idx, te_idx = np.where(df["job_title"].isin(train_titles))[0], \
        np.where(df["job_title"].isin(test_titles))[0]
    check("exactly 20 train titles", len(train_titles) == 20, f"(train_rows={len(tr_idx)})")
    check("exactly 8 test titles", len(test_titles) == 8, f"(test_rows={len(te_idx)})")
    check("zero title overlap", not (set(train_titles) & set(test_titles)))
    check("union == 28 dataset titles",
          len(set(train_titles) | set(test_titles)) == df["job_title"].nunique())
    check("no test-title row in train split",
          not bool(df.iloc[tr_idx]["job_title"].isin(test_titles).any()))
    check("no train-title row in test split",
          not bool(df.iloc[te_idx]["job_title"].isin(train_titles).any()))
    check("manifest saved to ARTIFACTS/job_title_split.json",
          (ROOT / "ARTIFACTS" / "job_title_split.json").exists())
    print("  TEST titles:", test_titles)
    print("  TRAIN titles:", train_titles)

    print("\n=== B. preprocessing fit ONLY on the 20 training titles ===")
    df_tr = df[df["job_title"].isin(train_titles)]
    # One doc per title — a title with only empty required_skills contributes
    # nothing (same shared-vectorizer behavior as the original Step-6 pipeline).
    docs_by_title = {}
    for _, r in df_tr.sort_values("job_title").iterrows():
        if r["job_title"] in docs_by_title or not str(r["required_skills"]).strip():
            continue
        docs_by_title[r["job_title"]] = r["required_skills"]
    corpus_docs = list(docs_by_title.values())
    check("TF-IDF corpus built from training-job titles only",
          set(docs_by_title) <= set(train_titles),
          extra=f"(docs={len(corpus_docs)} from {len(docs_by_title)} train titles)")
    check("TF-IDF corpus excludes all test-job titles",
          not (set(docs_by_title) & set(test_titles)))
    tfidf = build_skills_tfidf(corpus_docs)

    print("\n=== C. 17-feature matrix, same shared vectorizer for both sides ===")
    def build_X(idx):
        rows = []
        for i in idx:
            r = df.iloc[i]
            rows.append(candidate_job_match_features_rich(
                r["candidate_skills"], r["required_skills"],
                profile=profiles.get(r["candidate_skills"]),
                job_experience_required=r["experience_required"],
                job_education_required=r["education_required"],
                skills_tfidf=tfidf))
        return np.vstack([np.asarray(v, dtype=float) for v in rows])

    Xtr, Xte = build_X(tr_idx), build_X(te_idx)
    ytr, yte = df["_y"].to_numpy()[tr_idx], df["_y"].to_numpy()[te_idx]
    check("train matrix 17 features", Xtr.shape[1] == 17 == len(RICH_MATCH_FEATURE_NAMES))
    check("test matrix 17 features", Xte.shape[1] == 17 == len(RICH_MATCH_FEATURE_NAMES))
    check("feature matrices finite",
          bool(np.isfinite(Xtr).all()) and bool(np.isfinite(Xte).all()))

    print("\n=== D. Random Forest (200, seed 42, min_leaf 5) on 20 train titles ===")
    rf = RandomForestRegressor(n_estimators=200, random_state=SEED,
                               min_samples_leaf=5)
    rf.fit(Xtr, ytr)
    check("model NOT trained on any test-title row",
          len(tr_idx) + len(te_idx) == len(df) and
          not bool(df.iloc[tr_idx]["job_title"].isin(test_titles).any()))
    print("  feature importances (RF, 17 features):")
    imp = {n: round(v, 4) for n, v in
           zip(RICH_MATCH_FEATURE_NAMES, rf.feature_importances_)}
    for k, v in imp.items():
        print(f"    {k:24s} {v}")
    dom = imp["candidate_count"] + imp["required_count"]
    overlap_imp = sum(v for k, v in imp.items()
                      if k in ("overlap_count", "jaccard", "req_coverage",
                               "cand_precision"))
    check("count-pair dominance below overlap+compat+new sum",
          dom < sum(imp.values()) - dom, extra=f"(dom={dom:.3f})")

    print("\n=== E. evaluation: TRAIN (20 titles) vs TEST (8 held-out titles) ===")
    pred_tr = rf.predict(Xtr)
    pred_te = rf.predict(Xte)
    train_reg = eval_regression(ytr, pred_tr, "train")
    test_reg = eval_regression(yte, pred_te, "test")
    rm = global_mean_baseline(ytr)
    test_gm = eval_regression(yte, np.full(len(yte), rm), "test-global-mean")
    train_rank = eval_ranking(df, tr_idx, pred_tr, "train")
    test_rank = eval_ranking(df, te_idx, pred_te, "test")

    print("  --- TRAIN (20 titles, model-fit check) ---")
    print("   ", train_reg)
    print("    ranking:", train_rank)
    print("  --- TEST (8 held-out titles, single evaluation) ---")
    print("   ", test_reg)
    print("    ranking:", test_rank)
    print("  --- TEST baselines (held-out) ---")
    print("    global-mean:", test_gm)

    check("TRAIN R2 > 0.4", train_reg["R2"] > 0.4, f"(R2={train_reg['R2']})")
    check("TEST MAE < 0.16", test_reg["MAE"] < 0.16, f"(MAE={test_reg['MAE']})")
    check("TEST NDCG@10 >= 0.75", (test_rank.get("NDCG@10") or 0) >= 0.75,
          f"(NDCG@10={test_rank.get('NDCG@10')})")
    check("TEST predictions in [0,1]",
          test_reg["preds_outside_0_1"] == 0)

    print("\n=== F. save artifacts ===")
    blob = {
        "kind": "jobfit-match-model-step6",
        "model": rf,
        "features": list(RICH_MATCH_FEATURE_NAMES),
        "target": "matched_score",
        "trained_on_job_titles": list(train_titles),
        "held_out_job_titles": list(test_titles),
        "split_manifest": manifest,
    }
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MODEL, "wb") as fh:
        pickle.dump(blob, fh)
    meta = {
        "model_type": "RandomForestRegressor(200, seed=42, min_leaf=5)",
        "target": "matched_score (candidate-job relevance)",
        "features": list(RICH_MATCH_FEATURE_NAMES),
        "feature_importances": imp,
        "count_pair_dominance": round(dom, 4),
        "overlap_quality_importance": round(overlap_imp, 4),
        "split_strategy": "JOB-TITLE-LEVEL holdout: 20 train titles / 8 held-out test titles",
        "random_seed": SEED,
        "split_manifest": manifest,
        "train_metrics": dict(train_reg, **train_rank),
        "test_metrics": dict(test_reg, **test_rank),
        "test_global_mean_baseline": test_gm,
        "profile_source": "DATA/resume_data.csv candidate-side fields",
        "tfidf_corpus": "required-skills documents of the 20 TRAINING job titles only",
        "production_match_model_touched": False,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    check("saved match_model_jobtitle.pkl", OUTPUT_MODEL.exists())
    check("saved match_model_jobtitle_metadata.json", OUTPUT_META.exists())
    print("  saved:", OUTPUT_MODEL)
    print("  saved:", OUTPUT_META)
    print("  NOTE: production MODELS/match_model.pkl left untouched.")

    print("\n=== G. reproducibility: same split under seed 42 (repeat call) ===")
    train_titles2, test_titles2, _ = get_split(df, save=False)
    check("identical 20 train titles", train_titles2 == train_titles)
    check("identical 8 test titles", test_titles2 == test_titles)

    print(f"\n==== passed {len(PASS)} / failed {len(FAIL)} ====")
    print("elapsed", round(time.time() - t0, 1), "s")
    return {
        "train_titles": train_titles,
        "test_titles": test_titles,
        "manifest": manifest,
        "train_metrics": dict(train_reg, **train_rank),
        "test_metrics": dict(test_reg, **test_rank),
        "passed": len(PASS),
        "failed": len(FAIL),
    }


if __name__ == "__main__":
    result = run()
    sys.exit(1 if result["failed"] else 0)