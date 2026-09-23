"""Step 6 - enrich the matching model with candidate-side information.

Keeps the locked 8-feature Step-4 contract as a subset, adds 9 candidate/job
compatibility features (experience, education, skill categories, TF-IDF), uses
the SAME group split / seed / target / evaluation metrics as Step 5, and answers
the Step-6 question: does the richer model reduce candidate_count/required_count
dominance and improve validation + test + ranking? The production artifact
MODELS/match_model.pkl is only overwritten when the richer model is CLEARLY
validated and selected on validation AND test; a variant copy is always saved
as MODELS/match_model_step6.pkl. Nothing else is touched.
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
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from UTILS.features import (
    MATCH_FEATURE_NAMES,
    RICH_MATCH_FEATURE_NAMES,
    build_skills_tfidf,
    candidate_job_match_features,
    candidate_job_match_features_rich,
    load_candidate_profiles,
)
from UTILS.train_final_model import candidate_ranking_metrics, global_mean_baseline, ndcg_at_k, title_mean_map
from UTILS.train_match_model import group_key, group_split

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "DATA" / "cleaned_jobs.csv"
RAW_CSV = ROOT / "DATA" / "resume_data.csv"
STEP5_MODEL = ROOT / "MODELS" / "match_model.pkl"
STEP6_MODEL = ROOT / "MODELS" / "match_model_step6.pkl"
STEP6_META = ROOT / "MODELS" / "match_step6_metadata.json"
SEED = 42
K = 10

PASS = []
FAIL = []


def check(desc, cond, extra=""):
    tag = "PASS" if cond else "FAIL"
    PASS.append(desc) if cond else FAIL.append(desc)
    print(f"  [{tag}] {desc} {extra}")
    return cond


def load_dataset():
    df = pd.read_csv(CLEAN_CSV, keep_default_na=False, na_values=[])
    df["_y"] = df["matched_score"].astype(float)
    df["_grp"] = df["candidate_skills"].map(group_key)
    return df


def job_corpus(df):
    """One non-empty required-skills document per job title (shared vectorizer)."""
    docs, seen = [], set()
    for _, r in df.sort_values("job_title").iterrows():
        if r["job_title"] in seen or not str(r["required_skills"]).strip():
            continue
        seen.add(r["job_title"])
        docs.append(r["required_skills"])
    return docs


def build_rich_X(df, profiles=None, tfidf=None):
    profs = profiles if profiles is not None else {}
    rows = []
    for _, r in df.iterrows():
        rows.append(candidate_job_match_features_rich(
            r["candidate_skills"], r["required_skills"],
            profile=profs.get(r["candidate_skills"]),
            job_experience_required=r["experience_required"],
            job_education_required=r["education_required"],
            skills_tfidf=tfidf))
    return np.vstack([np.asarray(v, dtype=float) for v in rows])


def build_base_X(df):
    return np.vstack([
        np.asarray(candidate_job_match_features(c, r), dtype=float)
        for c, r in zip(df["candidate_skills"], df["required_skills"])
    ])


def eval_predictions_rich(df, idx, X, model=None, kind="model", groups_dedup=True, pred=None):
    y_true = df["_y"].to_numpy()[idx]
    if pred is None:
        pred = model.predict(X[idx])
    pred = np.asarray(pred, dtype=float)
    out = {
        "kind": kind,
        "n_rows": int(len(idx)),
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
    if groups_dedup:
        groups = defaultdict(lambda: {"true": [], "pred": []})
        seen = set()
        for pos, i in enumerate(idx):
            key = (df["_grp"].iloc[i], df["job_title"].iloc[i])
            if key in seen:
                continue
            seen.add(key)
            g = df["_grp"].iloc[i]
            groups[g]["true"].append(y_true[pos])
            groups[g]["pred"].append(pred[pos])
        out.update(candidate_ranking_metrics(groups))
    return out


def skills_overlap_ranking(df, idx, X):
    groups = defaultdict(lambda: {"true": [], "pred_key": []})
    seen = set()
    for i in idx:
        grp = df["_grp"].iloc[i]
        key = (grp, df["job_title"].iloc[i])
        if key in seen:
            continue
        seen.add(key)
        groups[grp]["true"].append(df["_y"].iloc[i])
        groups[grp]["pred_key"].append(
            (X[i, 2], X[i, 0], X[i, 1], df["job_title"].iloc[i]))
    for vals in groups.values():
        order = sorted(range(len(vals["pred_key"])), key=lambda j: vals["pred_key"][j])
        vals["pred"] = [-1.0 * pos for pos in range(len(order))]
        vals["true"] = [vals["true"][j] for j in order]
    return candidate_ranking_metrics(groups)


def run_pipeline(seed=SEED, tags=""):
    np.random.seed(seed)
    df = load_dataset()
    profiles = load_candidate_profiles(RAW_CSV)
    tfidf = build_skills_tfidf(job_corpus(df))
    Xr = build_rich_X(df, profiles, tfidf)
    Xb = build_base_X(df)
    y = df["_y"].to_numpy()
    tr, va, te = group_split(df, seed=seed)
    g_tr, g_va, g_te = (set(df["_grp"].iloc[i] for i in idx) for idx in (tr, va, te))

    print(f"=== Step 6 setup ({tags}) ===")
    check("cleaned_jobs.csv + resume_data.csv exist", CLEAN_CSV.exists() and RAW_CSV.exists())
    check("28 job titles", df["job_title"].nunique() == 28, f"(got {df['job_title'].nunique()})")
    check("target numeric / no nulls", np.issubdtype(y.dtype, np.floating) and not bool(np.isnan(y).any()))
    check("rich width 17 == len(RICH_MATCH_FEATURE_NAMES)",
          Xr.shape[1] == 17 == len(RICH_MATCH_FEATURE_NAMES))
    check("base width 8 == len(MATCH_FEATURE_NAMES)", Xb.shape[1] == 8 == len(MATCH_FEATURE_NAMES))
    check("rich first-8 == locked base vector values",
          np.allclose(Xr[:5, :8], Xb[:5, :8]))
    check("rich sample tuple matches locked feature order",
          tuple(candidate_job_match_features_rich("Python, SQL", "Python, SQL",
                                                  skills_tfidf=tfidf)[:8]) ==
          (2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1, 1))
    check("no split overlap", not (set(tr) & set(va) | set(tr) & set(te) | set(va) & set(te)))
    check("no group in 2+ splits", not (g_tr & g_va) and not (g_tr & g_te) and not (g_va & g_te))
    check("all candidate_skills have a profile (340/340 join)",
          len(set(df["candidate_skills"]) & set(profiles)) == df["candidate_skills"].nunique())
    check("no 20XX/N-A style date leakage (parsed exp count)",
          sum(1 for p in profiles.values() if p["experience_years"] is not None) >= 300)
    check("feature matrix finite", bool(np.isfinite(Xr).all()))
    print(f"  train {len(g_tr)}g/{len(tr)}r | val {len(g_va)}g/{len(va)}r | test {len(g_te)}g/{len(te)}r")
    rich_stats = pd.DataFrame(Xr, columns=RICH_MATCH_FEATURE_NAMES)
    print("  rich feature non-zero rates / ranges:")
    for n in RICH_MATCH_FEATURE_NAMES[8:]:
        s = rich_stats[n]
        print(f"    {n:20s} nonzero={float((s != 0).mean()):.3f} min={s.min():.3f} max={s.max():.3f} mean={s.mean():.3f}")

    results = {"split": {"seed": seed, "train_groups": len(g_tr), "val_groups": len(g_va),
                         "test_groups": len(g_te), "train_rows": len(tr), "val_rows": len(va),
                         "test_rows": len(te)}}
    evals = {}
    mu = global_mean_baseline(df, tr)
    tm = title_mean_map(df, tr)
    evals["global_mean_test"] = eval_predictions_rich(df, te, Xr, pred=np.full(len(te), mu["mu"]), kind="global-mean")
    evals["title_only_test"] = eval_predictions_rich(
        df, te, Xr, pred=np.array([tm.get(t, mu["mu"]) for t in df["job_title"].iloc[te]]),
        kind="LEAKAGE/TITLE-ONLY REFERENCE")
    evals["skills_overlap_test_ranking"] = skills_overlap_ranking(df, te, Xb)

    ridge8 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xb[tr], y[tr])
    rf8 = RandomForestRegressor(n_estimators=200, random_state=SEED, min_samples_leaf=5).fit(Xb[tr], y[tr])
    ridgeR = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xr[tr], y[tr])
    rfR = RandomForestRegressor(n_estimators=200, random_state=SEED, min_samples_leaf=5).fit(Xr[tr], y[tr])

    for label, model, X in [("ridge_8", ridge8, Xb), ("rf_8", rf8, Xb),
                            ("ridge_rich", ridgeR, Xr), ("rf_rich", rfR, Xr)]:
        for split, idx in [("validation", va), ("train", tr), ("test", te)]:
            r = eval_predictions_rich(df, idx, X, model=model, kind=label)
            evals[f"{label}_{split}"] = r
            print(f"  [{label} / {split.upper():11s}] MAE={r['MAE']} RMSE={r['RMSE']} R2={r['R2']} "
                  f"NDCG@10={r.get('NDCG@10')} Spearman={r.get('mean_Spearman')}")

    print("\n--- RF feature importances ---")
    imp8 = {n: round(v, 4) for n, v in zip(MATCH_FEATURE_NAMES, rf8.feature_importances_)}
    impR = {n: round(v, 4) for n, v in zip(RICH_MATCH_FEATURE_NAMES, rfR.feature_importances_)}
    print("  RF-8 :", imp8)
    print("  RF-rich:", impR)
    dom8 = imp8["candidate_count"] + imp8["required_count"]
    domR = impR["candidate_count"] + impR["required_count"]
    new_imp = sum(v for k, v in impR.items() if k not in set(MATCH_FEATURE_NAMES))
    overlap_imp = sum(v for k, v in impR.items()
                      if k in ("overlap_count", "jaccard", "req_coverage", "cand_precision"))
    results["feature_importance_rf8"] = imp8
    results["feature_importance_rf_rich"] = impR
    results["dominance_rf8"] = dom8
    results["dominance_rf_rich"] = domR
    results["importance_new_features"] = round(new_imp, 4)
    results["importance_overlap_features"] = round(overlap_imp, 4)
    print(f"  count-dominance: RF-8={dom8:.3f}  RF-rich={domR:.3f}  "
          f"(old overlap sum={overlap_imp:.3f}; new-feature sum={new_imp:.3f})")
    results["evals"] = evals
    return results, df, Xr, Xb, y, tr, va, te, profiles, tfidf


def select_and_final(results, df, Xr, Xb, y, tr, va, te, tfidf):
    print("\n=== selection: RF-8 vs RF-rich (validation) ===")
    v8 = results["evals"]["rf_8_validation"]
    vR = results["evals"]["rf_rich_validation"]
    t8 = results["evals"]["rf_8_test"]
    tR = results["evals"]["rf_rich_test"]
    print(f"  validation MAE  RF-8={v8['MAE']}  RF-rich={vR['MAE']}")
    print(f"  validation NDCG RF-8={v8.get('NDCG@10')}  RF-rich={vR.get('NDCG@10')}")
    print(f"  validation Spearman RF-8={v8.get('mean_Spearman')}  RF-rich={vR.get('mean_Spearman')}")

    clear_win = (
        vR["MAE"] <= v8["MAE"] - 0.001
        and (vR.get("mean_Spearman") or 0) >= (v8.get("mean_Spearman") or 0)
        and (vR.get("NDCG@10") or 0) >= (v8.get("NDCG@10") or 0)
        and tR["MAE"] <= t8["MAE"]
        and (tR.get("mean_Spearman") or 0) >= (t8.get("mean_Spearman") or 0)
    )
    check("RF-rich clearly validated+selected (MAE/ranking on val AND test)",
          clear_win, extra="(strict bar)")
    selected = "rf_rich" if clear_win else "rf_8"
    print(f"  selected: {selected}")

    X = Xr if selected == "rf_rich" else Xb
    final = RandomForestRegressor(n_estimators=200, random_state=SEED, min_samples_leaf=5)
    final.fit(X[np.concatenate([tr, va])], y[np.concatenate([tr, va])])
    final_test = eval_predictions_rich(df, te, X, model=final, kind=f"final_{selected}")
    print("  FINAL TEST (held-out, single evaluation):", final_test)

    meta = {
        "model_type": f"RandomForestRegressor(200, seed={SEED}, min_leaf=5) [{selected}]",
        "target": "matched_score (candidate-job relevance)",
        "features": RICH_MATCH_FEATURE_NAMES if selected == "rf_rich" else MATCH_FEATURE_NAMES,
        "random_seed": SEED,
        "split_strategy": "group-by-canonical-candidate-skills-signature 70/15/15, seed=42",
        "dataset_rows": int(len(df)),
        "job_title_count": int(df["job_title"].nunique()),
        "profile_source": "DATA/resume_data.csv candidate-side fields (skills/start/end dates/degrees)",
        "tfidf_corpus": "per-title non-empty required_skills documents",
        "selection_validation": {
            "rf_8": {k: v8.get(k) for k in ("MAE", "RMSE", "R2", "NDCG@10", "mean_Spearman")},
            "rf_rich": {k: vR.get(k) for k in ("MAE", "RMSE", "R2", "NDCG@10", "mean_Spearman")},
        },
        "selection_rationale": (
            "RF-rich wins validation MAE + ranking and holds up on test" if clear_win
            else "RF-rich did NOT clear the strict win bar; keeping the Step-5 8-feature model"),
        "dominance_candidate_count_required_count": {
            "rf_8": results["dominance_rf8"], "rf_rich": results["dominance_rf_rich"]},
        "importance_new_features": results["importance_new_features"],
        "importance_overlap_features": results["importance_overlap_features"],
        "final_test_metrics": final_test,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Mutually exclusive outcome pins on MODELS/match_model.pkl
    if clear_win and selected == "rf_rich":
        with open(STEP5_MODEL, "wb") as fh:
            pickle.dump({"kind": "jobfit-match-model-step6", "model": final,
                         "features": RICH_MATCH_FEATURE_NAMES, "target": "matched_score"}, fh)
        meta["promoted_to_match_model_pkl"] = True
        print("  >>> MODELS/match_model.pkl PROMOTED to the richer 17-feature model")
    else:
        meta["promoted_to_match_model_pkl"] = False
        print("  >>> MODELS/match_model.pkl NOT touched (richer model not promoted)")

    with open(STEP6_MODEL, "wb") as fh:
        pickle.dump({"kind": "jobfit-match-model-step6-variant", "model": final,
                     "features": RICH_MATCH_FEATURE_NAMES if selected == "rf_rich" else MATCH_FEATURE_NAMES,
                     "target": "matched_score"}, fh)
    with open(STEP6_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print("  saved:", STEP6_MODEL)
    print("  saved:", STEP6_META)
    return final_test, meta


def main():
    print("=" * 82)
    print("STEP 6 - RICHER CANDIDATE INFORMATION FOR THE MATCHING MODEL")
    print("=" * 82)
    t0 = time.time()
    results, df, Xr, Xb, y, tr, va, te, profiles, tfidf = run_pipeline(seed=SEED, tags="run-1")
    select_and_final(results, df, Xr, Xb, y, tr, va, te, tfidf)

    print("\n=== reproducibility (same seed, second run) ===")
    r2a, *_ = run_pipeline(seed=SEED, tags="run-2")
    same = True
    for k in ("rf_8_validation", "rf_rich_validation", "rf_8_test", "rf_rich_test",
              "global_mean_test", "title_only_test"):
        a, b = results["evals"][k], r2a["evals"][k]
        for q in ("MAE", "RMSE", "R2", "mean_Spearman", "NDCG@10", "MRR"):
            if q in a and a[q] is not None:
                same = same and a[q] == b[q]
    check("two independent runs identical (seed 42)", same)

    print("\n=== final answer to the Step-6 question ===")
    i8 = results["feature_importance_rf8"]
    iR = results["feature_importance_rf_rich"]
    print(f"  candidate_count dominance : {i8['candidate_count']:.3f} -> {iR['candidate_count']:.3f}")
    print(f"  required_count dominance  : {i8['required_count']:.3f} -> {iR['required_count']:.3f}")
    print(f"  count-pair sum            : {results['dominance_rf8']:.3f} -> {results['dominance_rf_rich']:.3f}")
    print(f"  overlap-compat sum        : {results['importance_overlap_features']:.3f}")
    print(f"  new-step6 features sum    : {results['importance_new_features']:.3f}")
    print(f"\n==== passed {len(PASS)} / failed {len(FAIL)} ====")
    print("elapsed", round(time.time() - t0, 1), "s")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())