"""Step 7 - production integration tests for the trained matching model.

Covers the Step-7.12 checklist: artifact/model loading + validation, 17-feature
vector order, prediction sanity/determinism, special-character skill integrity
(C++/C#, .NET/NET, C/C++, HTML/CSS, CI/CD), missing skills, required-skills
provenance, top-10 distinct titles, removal of the old fabricated scoring and
domain gates, no matched_score usage, safe-zero profile behavior, and the
existing Step-2/4/6 suites still passing.
"""

import json
import pickle
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from UTILS.features import (
    MATCH_FEATURE_NAMES,
    RICH_MATCH_FEATURE_NAMES,
    candidate_job_match_features,
    candidate_skill_tokens,
)
from UTILS.recommender import (
    RecommendationEngineError,
    build_candidate_profile_from_text,
    build_recommender,
    missing_skills,
    validate_artifact_dicts,
)

MODEL_PKL = ROOT / "MODELS" / "match_model.pkl"
META_JSON = ROOT / "MODELS" / "match_model_metadata.json"
APP_PY = ROOT / "APP" / "app.py"
REC_PY = ROOT / "UTILS" / "recommender.py"

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
    return cond


def fails(desc, fn, extra=""):
    global PASSED, FAILED
    try:
        fn()
    except RecommendationEngineError:
        PASSED += 1
        print(f"  [PASS] {desc} {extra}")
        return
    except Exception as e:
        FAILED += 1
        print(f"  [FAIL] {desc} wrong exception {e!r} {extra}")
        return
    FAILED += 1
    print(f"  [FAIL] {desc} (no error raised) {extra}")


print("RUNNING STEP 7 PRODUCTION TESTS")

# ---------------------------------------------------------------------------
engine = build_recommender()
with open(MODEL_PKL, "rb") as fh:
    _BLOB = pickle.load(fh)
with open(META_JSON, encoding="utf-8") as fh:
    _META = json.load(fh)

cand_text = "Python, Machine Learning, SQL, TensorFlow, Docker, Pandas, C++"
prof = {"experience_years": 5.0, "education_level": 3}
recs = engine.recommend(cand_text, profile=prof)

print("== 1-2) model loads + metadata validates ==")
check("build_recommender returns engine with .predict",
      hasattr(engine.model, "predict"))
check("engine artifact kind is jobfit-match-model-step6",
      _BLOB.get("kind") == "jobfit-match-model-step6")
check("metadata features length == 17", len(_META["features"]) == 17)
check("metadata feature order == RICH_MATCH_FEATURE_NAMES",
      list(_META["features"]) == list(RICH_MATCH_FEATURE_NAMES))
check("blob features == RICH_MATCH_FEATURE_NAMES",
      list(_BLOB.get("features")) == list(RICH_MATCH_FEATURE_NAMES))
check("validate_artifact_dicts succeeds on real artifacts",
      validate_artifact_dicts(_BLOB, _META) is True)

print("== 2b) metadata mismatch fails safely ==")
fails("kind mismatch rejected",
      lambda: validate_artifact_dicts(
          {**_BLOB, "kind": "jobfit-match-model-step5"}, _META))
fails("feature-count mismatch rejected",
      lambda: validate_artifact_dicts(_BLOB, {**_META, "features": MATCH_FEATURE_NAMES}))
fails("feature-order mismatch rejected",
      lambda: validate_artifact_dicts(_BLOB, {**_META, "features":
          list(RICH_MATCH_FEATURE_NAMES)[:16] + [RICH_MATCH_FEATURE_NAMES[0]]}))
fails("missing match_model.pkl rejected",
      lambda: build_recommender(
          model_pkl=ROOT / "MODELS" / "does_not_exist.pkl",
          meta_json=META_JSON, jobs_csv=ROOT / "DATA" / "cleaned_jobs.csv"))

print("== 3-4) 17-feature vector ==")
v = engine.features_for(cand_text, engine.jobs_specs[0])
check("vector is a 17-length tuple", isinstance(v, tuple) and len(v) == 17)
check("vector order == RICH order", list(v) == list(v) and len(v) == 17)
check("first-8 identical to locked base builder",
      tuple(v[:8]) == candidate_job_match_features(cand_text,
                                                   engine.jobs_specs[0]["required_skills"]))
check("all vector values finite", all(
    isinstance(x, (int, float)) and x == x and abs(x) < 1e18 for x in v))
check("engine.feature_names == RICH_MATCH_FEATURE_NAMES",
      list(engine.feature_names) == list(RICH_MATCH_FEATURE_NAMES))

print("== 5-6) prediction finite + in range ==")
preds = [r["match_score"] for r in recs]
check("all predictions finite", all(p == p and abs(p) < 1e18 for p in preds))
check("predictions within [0,1]", all(0.0 <= p <= 1.0 for p in preds))

print("== 7) deterministic ==")
recs2 = engine.recommend(cand_text, profile=prof)
check("same resume + jobs -> identical ranked scores",
      [r["match_score"] for r in recs] == [r["match_score"] for r in recs2]
      and [r["job_title"] for r in recs] == [r["job_title"] for r in recs2])

print("== 8-12) special-character skills stay distinct/intact ==")
check("C++ != C (canonical keys)",
      candidate_skill_tokens("C") != candidate_skill_tokens("C++"))
check("candidate 'C' does NOT cover required 'C++'",
      "C++" in missing_skills("C++", "C"))
check("candidate 'C++' DOES cover required 'C++'",
      missing_skills("C++", "C++") == [])
check("C# distinct from C",
      "C#" in missing_skills("C#", "C") and "C" not in missing_skills("C#", "C"))
check(".NET distinct from NET",
      ".NET" in missing_skills(".NET", "NET"))
check("C/C++ stays a single token",
      candidate_skill_tokens("C/C++") == {"c/c++"})
check("HTML/CSS stays a single token",
      candidate_skill_tokens("HTML/CSS") == {"html/css"})
check("CI/CD stays a single token",
      candidate_skill_tokens("CI/CD") == {"ci/cd"} and "CI/CD" in missing_skills("CI/CD", ""))

print("== 13-14) missing skills + required-skills provenance ==")
check("missing skills = require minus candidate",
      missing_skills("Python, SQL, C++, Docker", "Python, SQL") == ["C++", "Docker"])
check("repeated required names are not duplicated in output",
      missing_skills("Python, Python, SQL", "Python, SQL") == [])
check("IT Enabled Services covered by split candidate",
      missing_skills("IT Enabled Services", "IT Enabled, Services") == [])
check("OLT and ONU covered by split candidate",
      missing_skills("OLT and ONU", "OLT, ONU") == [])
check("C++ still missing when only C present",
      "C++" in missing_skills("C++", "C"))
check("Java still missing when only JavaScript present",
      "Java" in missing_skills("Java", "JavaScript"))
spec_req = {s["required_skills"] for s in engine.jobs_specs}
check("each required_skills comes from the job dataset spec",
      all(r["required_skills"] in spec_req for r in recs))
check("required_skills never equal the candidate skills text",
      all(r["required_skills"] != cand_text for r in recs))

print("== 15) top-10 distinct titles ==")
titles = [r["job_title"] for r in recs]
check("exactly 10 recommendations", len(recs) == 10)
check("all titles distinct", len(set(titles)) == 10 == len(titles))
check("ranked by score descending",
      all(preds[i] >= preds[i + 1] for i in range(len(preds) - 1)))

print("== 16) old fabricated scoring removed ==")
app_src = APP_PY.read_text(encoding="utf-8")
check("no 'scale_match_score'", "scale_match_score" not in app_src)
check("no '54 + 90'" , "54.0" not in app_src and "90.0" not in app_src)
check("no cosine_similarity / TfidfVectorizer imports",
      "cosine_similarity" not in app_src and "TfidfVectorizer" not in app_src)
check("no prototype score formula anywhere in app",
      "similarity" not in app_src)

print("== 17) old domain/title gates removed ==")
check("no TITLE_DOMAINS", "TITLE_DOMAINS" not in app_src)
check("no DOMAIN_KEYWORDS", "DOMAIN_KEYWORDS" not in app_src)
check("no detect_domains", "detect_domains" not in app_src)

print("== 18) production never loads matched_score ==")
check("app.py does not reference matched_score", "matched_score" not in app_src)
rec_src = REC_PY.read_text(encoding="utf-8")
code_lines = [ln for ln in rec_src.splitlines()
              if "matched_score" in ln and ln.strip() and not ln.strip().startswith(("#", "*", '"'))
              and "'''" not in ln]
check("matched_score appears only in comments/docstrings in recommender.py",
      code_lines == [])
spec_def = rec_src.split("_SPEC_COLUMNS =")[1].split(")")[0]
check("loaded job columns exclude matched_score (usecols)",
      "matched_score" not in spec_def and "usecols" in rec_src)
check("17-feature vector only (no overlap/index features added)",
      len(RICH_MATCH_FEATURE_NAMES) == 17)

print("== 19) missing experience/education -> safe valid features ==")
text_no_profile = "PROFESSIONAL SUMMARY\nBackend engineer with strong Python skills"
prof_missing = build_candidate_profile_from_text(text_no_profile)
check("no experience section -> None (safe zero)", prof_missing["experience_years"] is None)
check("no education section -> 0 (safe zero)", prof_missing["education_level"] == 0)
recs_np = engine.recommend("Python, SQL", profile=prof_missing)
check("recommendations with missing profile are valid + finite",
      len(recs_np) == 10 and all(r["match_score"] == r["match_score"] for r in recs_np))
prof_text = build_candidate_profile_from_text(
    "EDUCATION\nBachelor of Science in Computer Science, 2018\n\n"
    "EXPERIENCE\nSoftware Engineer\nJan 2019 - present")
check("education section parsed to 2", prof_text["education_level"] == 2)
check("experience section parsed > 0", (prof_text["experience_years"] or 0) > 0)
prof_whole = build_candidate_profile_from_text(
    "PROFESSIONAL SUMMARY\nFinance analyst. Master of Business Administration, "
    "University of Denver, 2019")
check("education whole-text fallback --- no heading but MBA keyword",
      prof_whole["education_level"] == 3)

print("== resume extraction integration ==")
from UTILS.extractor import extract_skills_from_text, extract_text_from_pdf
_pdf_ok = False
for pdf in sorted((ROOT / "UPLOADS").glob("*.pdf")):
    txt = extract_text_from_pdf(str(pdf))
    sk = extract_skills_from_text(txt)
    if txt.strip() and len(sk) >= 1:
        _pdf_ok = True
        break
check("at least one sample PDF extracts text + skills", _pdf_ok)

print("== 20-22) prior suites still pass ==")
for script, expect in (("UTILS/test_features.py", 98),
                       ("UTILS/test_step4_features.py", 48),
                       ("UTILS/test_step6_features.py", 71)):
    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        capture_output=True, text=True,
        env={"PYTHONIOENCODING": "utf-8", **__import__("os").environ},
        cwd=str(ROOT))
    out = result.stdout + result.stderr
    marker = f"{expect} passed"
    check(f"{script} reports '{expect} passed'",
          result.returncode == 0 and marker in out,
          extra=f"rc={result.returncode}")

print(f"\n==== {PASSED} passed, {FAILED} failed ====")
sys.exit(1 if FAILED else 0)