"""Focused regression tests for the scoring gate (overlap-based penalty).

Verifies that zero/low skill overlap produces appropriately low scores,
while overlapping skills produce meaningful non-zero scores.

Run:  python UTILS/test_scoring_gate.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from UTILS.recommender import build_recommender
from UTILS.features import (
    candidate_job_match_features,
    candidate_job_match_features_rich,
    RICH_MATCH_FEATURE_NAMES,
)

PASSED = 0
FAILED = 0


def check(label, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}")


engine = build_recommender()

# ---------- Test A: Zero-overlap resumes must score 0 ----------

print("\n== A) Zero-overlap resumes must score 0 ==")

network_skills = "Networking, TCP/IP, Cisco, LAN, WAN, Firewall, VPN"
ml_skills = "Python, TensorFlow, PyTorch, Scikit-learn, Machine Learning"
db_skills = "MySQL, MongoDB, Redis, Elasticsearch, PostgreSQL"

# Jobs whose required skills genuinely have zero overlap with these resumes
zero_overlap_cases = [
    ("NETWORK", network_skills, "Database Administrator (DBA)"),
    ("NETWORK", network_skills, "AI Engineer"),
    ("NETWORK", network_skills, "HR Officer"),
    ("ML", ml_skills, "Database Administrator (DBA)"),
    ("ML", ml_skills, "HR Officer"),
    ("ML", ml_skills, "Network Support Engineer"),
    ("DB", db_skills, "Machine Learning (ML) Engineer"),
    ("DB", db_skills, "AI Engineer"),
    ("DB", db_skills, "HR Officer"),
    ("DB", db_skills, "Network Support Engineer"),
]

for resume_label, resume_skills, job_title in zero_overlap_cases:
    for spec in engine.jobs_specs:
        if spec["job_title"] != job_title:
            continue
        vec = candidate_job_match_features_rich(
            resume_skills, spec["required_skills"],
            job_experience_required=spec["experience_required"],
            job_education_required=spec["education_required"],
            skills_tfidf=engine.tfidf,
        )
        jaccard = vec[1]
        score = engine.score_row(resume_skills, {}, spec)
        check(
            f"{resume_label} vs {job_title}: jaccard={jaccard:.3f} -> score={score:.3f}",
            jaccard == 0.0 and score == 0.0,
        )

# ---------- Test B: Overlapping skills produce non-zero scores ----------

print("\n== B) Overlapping skills produce non-zero scores ==")

# DB resume vs DBA: 3 overlapping skills
for spec in engine.jobs_specs:
    if spec["job_title"] == "Database Administrator (DBA)":
        vec = candidate_job_match_features_rich(
            db_skills, spec["required_skills"],
            job_experience_required=spec["experience_required"],
            job_education_required=spec["education_required"],
            skills_tfidf=engine.tfidf,
        )
        score = engine.score_row(db_skills, {}, spec)
        check(f"DB vs DBA: overlap={vec[0]:.0f}, jaccard={vec[1]:.3f} -> score>0.1",
              score > 0.1)
        check(f"DB vs DBA: score in (0, 1)", 0.0 < score <= 1.0)

# ML resume vs AI Engineer: overlapping ML skills
for spec in engine.jobs_specs:
    if spec["job_title"] == "AI Engineer":
        vec = candidate_job_match_features_rich(
            ml_skills, spec["required_skills"],
            job_experience_required=spec["experience_required"],
            job_education_required=spec["education_required"],
            skills_tfidf=engine.tfidf,
        )
        score = engine.score_row(ml_skills, {}, spec)
        check(f"ML vs AI Engineer: overlap={vec[0]:.0f}, jaccard={vec[1]:.3f} -> score>0.1",
              score > 0.1)

# ---------- Test C: Score is always in [0, 1] ----------

print("\n== C) All scores in [0, 1] ==")

for resume_label, resume_skills in [
    ("NETWORK", network_skills),
    ("ML", ml_skills),
    ("DB", db_skills),
]:
    results = engine.recommend(resume_skills, profile=None, top_n=10)
    for rec in results:
        check(
            f"{resume_label} -> {rec['job_title'][:35]}: score in [0,1]",
            0.0 <= rec["match_score"] <= 1.0,
        )

# ---------- Test D: Rank order makes sense ----------

print("\n== D) Rank order makes sense ==")

# DB resume: DBA should rank higher than ML Engineer (which has empty reqs)
db_results = engine.recommend(db_skills, profile=None, top_n=28)
db_scores = {r["job_title"]: r["match_score"] for r in db_results}
dba_score = db_scores.get("Database Administrator (DBA)", 0)
ml_score = db_scores.get("Machine Learning (ML) Engineer", 0)
check(f"DB resume: DBA ({dba_score:.3f}) > ML ({ml_score:.3f})",
      dba_score > ml_score)

# ML resume: AI Engineer (has overlapping skills) should rank higher than HR
ml_results = engine.recommend(ml_skills, profile=None, top_n=28)
ml_scores = {r["job_title"]: r["match_score"] for r in ml_results}
ai_score = ml_scores.get("AI Engineer", 0)
hr_score = ml_scores.get("HR Officer", 0)
check(f"ML resume: AI Engineer ({ai_score:.3f}) > HR ({hr_score:.3f})",
      ai_score > hr_score)

# Network resume: irrelevant jobs should be 0
net_results = engine.recommend(network_skills, profile=None, top_n=28)
net_scores = {r["job_title"]: r["match_score"] for r in net_results}
check(f"Network resume: ML Engineer score=0",
      net_scores.get("Machine Learning (ML) Engineer", -1) == 0.0)
check(f"Network resume: DBA score=0",
      net_scores.get("Database Administrator (DBA)", -1) == 0.0)
check(f"Network resume: AI Engineer score=0",
      net_scores.get("AI Engineer", -1) == 0.0)
check(f"Network resume: HR Officer score=0",
      net_scores.get("HR Officer", -1) == 0.0)

# ---------- Test E: Edge cases ----------

print("\n== E) Edge cases ==")

# Empty candidate skills
for spec in engine.jobs_specs[:3]:
    score = engine.score_row("", {}, spec)
    check(f"Empty candidate vs {spec['job_title'][:30]}: score=0", score == 0.0)

# Empty required skills
score = engine.score_row(network_skills, {}, {
    "job_title": "test",
    "required_skills": "",
    "experience_required": "",
    "education_required": "",
})
check("Non-empty candidate vs empty required: score=0", score == 0.0)

# ---------- Test F: Gated score <= raw model prediction ----------

print("\n== F) Gated score <= raw model prediction ==")

import numpy as np

for resume_label, resume_skills in [
    ("DB", db_skills),
    ("ML", ml_skills),
]:
    for spec in engine.jobs_specs:
        vec = candidate_job_match_features_rich(
            resume_skills, spec["required_skills"],
            job_experience_required=spec["experience_required"],
            job_education_required=spec["education_required"],
            skills_tfidf=engine.tfidf,
        )
        raw_pred = float(np.asarray(engine.model.predict(
            np.asarray(vec, dtype=float).reshape(1, -1))).reshape(-1)[0])
        gated = engine.score_row(resume_skills, {}, spec)
        check(
            f"{resume_label} vs {spec['job_title'][:30]}: gated({gated:.3f}) <= raw({raw_pred:.3f})",
            gated <= raw_pred + 0.001,  # small float tolerance
        )

# ---------- Summary ----------
print(f"\n{'='*60}")
total = PASSED + FAILED
print(f"  {PASSED}/{total} passed, {FAILED} failed")
if FAILED:
    print("  *** FAILURES DETECTED ***")
    sys.exit(1)
else:
    print("  All tests passed")
