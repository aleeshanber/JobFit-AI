"""Step 3 - new ML formulation baseline for JobFit AI.

Resume -> job recommendation. The target is ``matched_score`` (candidate-job
relevance) for each (candidate, job) row, NOT the 28-class job_title.

This script:
  * builds candidate-job overlap features using the shared Step-2 preprocessing
  * splits rows by CANDIDATE GROUP so the same resume never straddles a split
  * trains a supervised regression baseline (Ridge) validating the formulation
  * reports regression metrics AND within-candidate ranking metrics

It is a formulation-validation baseline only. It is deliberately NOT wired to
production (nothing in APP/ or the Flask routes reads the artifact it saves).
"""

import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from UTILS.features import (
    MATCH_FEATURE_NAMES,
    candidate_job_match_features,
    candidate_skill_tokens,
)

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "DATA" / "cleaned_jobs.csv"
OUT_MODEL = ROOT / "MODELS" / "match_baseline.pkl"
OUT_REPORT = ROOT / "MODELS" / "match_baseline_report.json"
SEED = 42


def load_rows():
    df = pd.read_csv(CLEAN_CSV, keep_default_na=False, na_values=[])
    return df


def group_key(candidate_skills):
    toks = candidate_skill_tokens(candidate_skills)
    return "|".join(sorted(toks))


def build_design(df):
    feats = np.array(
        [list(candidate_job_match_features(c, r))
         for c, r in zip(df["candidate_skills"], df["required_skills"])],
        dtype=float,
    )
    y = df["matched_score"].astype(float).to_numpy()
    return feats, y


def group_split(df, seed=SEED, train=0.70, val=0.15):
    rng = np.random.RandomState(seed)
    keys = df["candidate_skills"].map(group_key)
    groups = np.array(pd.unique(keys))
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * train)
    n_val = int(n * val)

    def ids(subset):
        sel = set(subset.tolist())
        return df.index[keys.isin(sel)].to_numpy()

    tr = ids(groups[:n_train])
    va = ids(groups[n_train:n_train + n_val])
    te = ids(groups[n_train + n_val:])
    return tr, va, te


def ndcg_at_k(rel_true, rank_pred, k=10):
    order = np.argsort(-np.asarray(rank_pred, dtype=float))
    sorted_rel = np.asarray(rel_true, dtype=float)[order]
    dcg = sum((2 ** v - 1) / np.log2(i + 2) for i, v in enumerate(sorted_rel[:k]))
    ideal = np.sort(np.asarray(rel_true, dtype=float))[::-1][:k]
    idcg = sum((2 ** v - 1) / np.log2(i + 2) for i, v in enumerate(ideal)) if len(ideal) else 0.0
    return dcg / idcg if idcg > 0 else 0.0


def ranking_metrics(pairs, k=10):
    """pairs: list of dicts {'true': [], 'pred': []} per candidate."""
    ndcg, rec, pre, hit, mrr, spear = [], [], [], [], [], []
    for p in pairs:
        t = np.asarray(p["true"], dtype=float)
        r = np.asarray(p["pred"], dtype=float)
        if len(t) < 2 or np.all(t == t[0]):
            continue
        ndcg.append(ndcg_at_k(t, r, k))
        order = np.argsort(-r, kind="mergesort")
        ranked_true = t[order]
        ranked_pred = r[order]
        top = ranked_true[:k]
        rel = t >= np.percentile(t, 90)
        top_rel = rel[order][:k]
        rec.append(float(top_rel.sum() / max(rel.sum(), 1)))
        pre.append(float(top_rel.mean()) if len(top_rel) else 0.0)
        hit.append(1.0 if top_rel.sum() > 0 else 0.0)
        for j, idx in enumerate(order):
            if rel[idx]:
                mrr.append(1.0 / (j + 1))
                break
        else:
            mrr.append(0.0)
        ss = spearmanr(t, r)
        spear.append(ss.correlation if np.isfinite(ss.correlation) else np.nan)
    spear = [x for x in spear if x == x]
    return {
        f"NDCG@{k}": np.mean(ndcg),
        f"Recall@{k}": np.mean(rec),
        f"Precision@{k}": np.mean(pre),
        f"Hit@{k}": np.mean(hit),
        "MRR": np.mean(mrr),
        "mean_Spearman": np.mean(spear) if spear else np.nan,
    }


def main():
    df = load_rows()
    print(f"rows: {len(df)} | candidates(groups): {df['candidate_skills'].map(group_key).nunique()} | titles: {df['job_title'].nunique()}")

    tr_idx, va_idx, te_idx = group_split(df)
    print(f"split rows train/val/test: {len(tr_idx)}/{len(va_idx)}/{len(te_idx)} "
          f"(groups {df['candidate_skills'].map(group_key).iloc[tr_idx].nunique()}/"
          f"{df['candidate_skills'].map(group_key).iloc[va_idx].nunique()}/"
          f"{df['candidate_skills'].map(group_key).iloc[te_idx].nunique()})")

    feats, y = build_design(df)

    # A title-average model: predicts each job's TRAIN mean. If it scores well on
    # within-candidate ranking while knowing nothing about the candidate, that
    # shows how job-only fields could "memorize" job identity - the 28-title trap.
    title_mean_map = {
        t: float(g["matched_score"].astype(float).mean())
        for t, g in df.iloc[tr_idx].groupby("job_title")
    }

    def eval_split(name, idx, model=None, naive="global_mean"):
        X, yt = feats[idx], y[idx]
        if model is None:
            if naive == "title_mean":
                pred = np.array([title_mean_map.get(t, yt.mean()) for t in df["job_title"].iloc[idx]])
            else:
                pred = np.full(len(idx), float(y.mean()))
        else:
            pred = model.predict(X)
        mae = mean_absolute_error(yt, pred)
        rmse = float(np.sqrt(mean_squared_error(yt, pred)))
        r2 = r2_score(yt, pred)
        pairs = [{"true": [], "pred": []}]
        # build per-candidate pred lists for ranking
        keys = df["candidate_skills"].map(group_key)
        coll = defaultdict(lambda: {"true": [], "pred": []})
        for pos, i in enumerate(idx):
            coll[keys[i]]["true"].append(yt[pos])
            coll[keys[i]]["pred"].append(pred[pos])
        rank = ranking_metrics(list(coll.values()))
        print(f"\n[{name}] n={len(idx)}")
        print(f"  MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}")
        print("  ranking:", {k: round(v, 4) if v == v else None for k, v in rank.items()})
        return {"name": name, "n": int(len(idx)), "MAE": round(mae, 4),
                "RMSE": round(rmse, 4), "R2": round(r2, 4), **{k: round(v, 4) if v == v else None for k, v in rank.items()}}

    results = {}

    # Title-average naive reference on TEST (job constants only, no candidate signal)
    res = eval_split("TEST title-average (job-only, no candidate)", te_idx, naive="title_mean")
    results["title_average_test"] = res

    # Baseline A: Ridge on shared overlap features
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(feats[tr_idx], y[tr_idx])
    results["validation_ridge"] = eval_split("VALIDATION Ridge", va_idx, ridge)
    results["test_ridge"] = eval_split("TEST Ridge", te_idx, ridge)

    # Comparator B: tree-based regressor, same features, no leakage change
    rf = RandomForestRegressor(n_estimators=200, random_state=SEED, min_samples_leaf=5)
    rf.fit(feats[tr_idx], y[tr_idx])
    results["test_randomforest"] = eval_split("TEST RandomForest", te_idx, rf)

    r2_ridge = results["test_ridge"]["R2"]
    winner = "ridge" if (results["test_ridge"]["MAE"] <= results["test_randomforest"]["MAE"]) else "randomforest"

    with open(OUT_MODEL, "wb") as fh:
        pickle.dump({"kind": "jobfit-match-baseline-step3", "model": ridge, "features": MATCH_FEATURE_NAMES}, fh)
    report = {
        "note": "step-3 formulation validation baseline; NOT connected to production",
        "rows": int(len(df)),
        "titles": int(df["job_title"].nunique()),
        "candidate_groups": int(df["candidate_skills"].map(group_key).nunique()),
        "split": {"train": int(len(tr_idx)), "validation": int(len(va_idx)), "test": int(len(te_idx)),
                  "seed": SEED},
        "matched_score": {"min": float(y.min()), "max": float(y.max()),
                          "mean": float(y.mean()), "median": float(np.median(y)),
                          "nulls": int(0), "unique": int(len(np.unique(y)))},
        "target": "matched_score (candidate-job relevance)",
        "features": MATCH_FEATURE_NAMES,
        "results": results,
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\nsaved:", OUT_MODEL)
    print("saved:", OUT_REPORT)
    print("chosen baseline model:", winner)
    return report


if __name__ == "__main__":
    main()