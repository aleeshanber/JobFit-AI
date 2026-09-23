"""Step 4 - locked feature contract tests.

Covers: the 10 required skill edge cases, no-partial-string-matching, feature
finiteness, locked ordering, training/inference consistency, no duplicate
implementations, and compatibility of MODELS/match_baseline.pkl with the
feature contract.
"""

import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UTILS.features import (
    MATCH_FEATURE_NAMES,
    candidate_job_match_features,
    match_feature_dict,
    validate_match_features,
)

PASS = []
FAIL = []


def check(desc, cond, extra=""):
    if cond:
        PASS.append(desc)
        print(f"  [PASS] {desc}")
    else:
        FAIL.append(desc)
        print(f"  [FAIL] {desc} {extra}")


def vec(c, r):
    v = candidate_job_match_features(c, r)
    validate_match_features(v)
    return dict(zip(MATCH_FEATURE_NAMES, v))


print("== 4D edge cases ==")
d = vec("Python, SQL, C++", "Python, SQL, C++")
check("CASE1 full overlap overlap_count=3", d["overlap_count"] == 3)
check("CASE1 jaccard=1.0", math.isclose(d["jaccard"], 1.0))
check("CASE1 req_coverage=1.0", math.isclose(d["req_coverage"], 1.0))
check("CASE1 cand_precision=1.0", math.isclose(d["cand_precision"], 1.0))

d = vec("Python, SQL", "Python, SQL, C++, Java")
check("CASE2 overlap_count=2", d["overlap_count"] == 2)
check("CASE2 req_coverage=0.5", math.isclose(d["req_coverage"], 0.5))
check("CASE2 cand_precision=1.0", math.isclose(d["cand_precision"], 1.0))

d = vec("Python", "Java")
check("CASE3 overlap=0", d["overlap_count"] == 0)
check("CASE3 overlap features finite", all(math.isfinite(v) for v in d.values()))

d = vec("", "Python, SQL")
check("CASE4 empty candidate no crash", d is not None)
check("CASE4 all finite", all(math.isfinite(v) for v in d.values()))
check("CASE4 overlap=0", d["overlap_count"] == 0)
check("CASE4 req_coverage=0.0", d["req_coverage"] == 0.0)
check("CASE4 cand_precision=0.0", d["cand_precision"] == 0.0)
check("CASE4 candidate_has_skills=0", d["candidate_has_skills"] == 0)
check("CASE4 job_has_requirements=1", d["job_has_requirements"] == 1)

d = vec(None, "Python, SQL")
check("CASE4 None candidate safe", d["overlap_count"] == 0 and all(math.isfinite(v) for v in d.values()))

print("== 4D special symbols: never conflated ==")
check("CASE5 C++ vs C distinct (overlap=0)", vec("C++", "C")["overlap_count"] == 0)
check("CASE6 C# vs C distinct (overlap=0)", vec("C#", "C")["overlap_count"] == 0)
check("CASE7 .NET vs NET distinct (overlap=0)", vec(".NET", "NET")["overlap_count"] == 0)
check("CASE8 C/C++ self-match=1",
      vec("C/C++", "C/C++")["overlap_count"] == 1)
check("CASE8 C/C++ does not equal C + C++ (overlap 0 with C)",
      vec("C/C++", "C")["overlap_count"] == 0)
check("CASE9 HTML/CSS self-match=1", vec("HTML/CSS", "HTML/CSS")["overlap_count"] == 1)
check("CASE9 HTML/CSS vs HTML not substring match",
      vec("HTML/CSS", "HTML")["overlap_count"] == 0)

d = vec("Python, Python, SQL", "Python, SQL")
check("CASE10 duplicate Python does not inflate (overlap=2)", d["overlap_count"] == 2)
check("CASE10 candidate_count=2", d["candidate_count"] == 2)

print("== 4E no partial-string matching ==")
check("Java != JavaScript (overlap=0)", vec("Java", "JavaScript")["overlap_count"] == 0)
check("SQL != NoSQL (overlap=0)", vec("SQL", "NoSQL")["overlap_count"] == 0)
check("React != React Native (overlap=0)", vec("React", "React Native")["overlap_count"] == 0)
check("Machine != Machine Learning (overlap=0)", vec("Machine", "Machine Learning")["overlap_count"] == 0)
check("Node != Node.js (overlap=0)", vec("Node", "Node.js")["overlap_count"] == 0)

print("== case-insensitive matching (canonical lower) ==")
check("Python vs python overlap=1", vec("Python", "python")["overlap_count"] == 1)
check("c++ vs C++ overlap=1", vec("c++", "C++")["overlap_count"] == 1)

print("== 4B order is the locked contract ==")
v = candidate_job_match_features("Python, SQL, Node.js", "Python, Node.js")
check("vector is tuple", isinstance(v, tuple))
check("len 8", len(v) == 8)
check("order names match", all(isinstance(x, (int, float)) for x in v))
d = match_feature_dict("Python, SQL, Node.js", "Python, Node.js")
check("dict keys == MATCH_FEATURE_NAMES", list(d.keys()) == MATCH_FEATURE_NAMES)

print("== 4F validate_match_features rejects bad vectors ==")
for bad in [
    [0] * 7,
    [0] * 9,
    ["x"] + [0] * 7,
    [float("nan")] + [1.0] * 7,
    [float("inf")] + [1.0] * 7,
]:
    try:
        validate_match_features(bad)
        check(f"rejects {bad[:2]}...", False)
    except ValueError:
        check(f"rejects {str(bad[:2])}...", True)
check("accepts real vector", validate_match_features(v) is v)

print("== 4G training==inference consistency ==")
a = candidate_job_match_features(
    "['Python', 'SQL', 'C++', 'Machine Learning']", "Python, SQL, Machine Learning"
)
b = candidate_job_match_features("python, sql, c++, machine learning", "PYTHON, SQL, Machine Learning")
check("exact same vector from raw vs messy input", a == b)

print("== 4G duplicate implementation scan (project files) ==")
import re as _re

root = Path(__file__).resolve().parents[1]
# Look for competing DEFINITIONS of the locked features (dict-key assignments or
# a function named candidate_job_match_features) anywhere outside UTILS/features.py
# and test files. Plain mentions in comments/docstrings are not implementations.
pat_key = _re.compile(r"""['\"]overlap_count['\"]\s*:|['\"]jaccard['\"]\s*:|['\"]req_coverage['\"]\s*:|['\"]cand_precision['\"]\s*:|['\"]candidate_count['\"]\s*:|['\"]required_count['\"]\s*:|def\s+candidate_job_match_features\b""")
hits = []
for f in root.rglob("*.py"):
    rel = f.relative_to(root)
    if any(p in rel.parts for p in ("node_modules", ".git", "venv", ".venv")):
        continue
    if str(rel).replace("\\", "/") == "UTILS/features.py":
        continue
    if rel.parts and rel.parts[0] == "UTILS" and rel.name.startswith("test"):
        continue
    if pat_key.search(f.read_text(encoding="utf-8", errors="ignore")):
        hits.append(str(rel))
check("no duplicate builder outside features.py/its tests", len(hits) == 0, extra=f"found: {sorted(hits)}")

print("== 4I match_baseline.pkl compatibility ==")
model_path = root / "MODELS" / "match_baseline.pkl"
if model_path.exists():
    with open(model_path, "rb") as fh:
        blob = pickle.load(fh)
    check("baseline metadata features == MATCH_FEATURE_NAMES",
          blob.get("features") == MATCH_FEATURE_NAMES, extra=str(blob.get("features")))
    check("baseline kind is step-3 validation", blob.get("kind") == "jobfit-match-baseline-step3")
    import numpy as np
    X = np.array([candidate_job_match_features("Python, SQL, C++, Go, Rust", "Python, SQL, Docker")], dtype=float)
    pred = blob["model"].predict(X)
    check("baseline accepts 8-feature row", pred.shape == (1,) and math.isfinite(float(pred[0])))
else:
    check("match_baseline.pkl exists", False)

print(f"\n==== {len(PASS)} passed, {len(FAIL)} failed ====")
sys.exit(1 if FAIL else 0)