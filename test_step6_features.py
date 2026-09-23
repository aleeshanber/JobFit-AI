"""Step 6 - tests for the richer candidate-side matching features.

Covers: date/education/experience parsers, category classifier (with the
special-character skills), shared TF-IDF similarity, the 17-feature builder
(identical first-8 vs the locked Step-4 contract), safe-zero behavior, profile
loading (read-only candidate fields, full join), leakage audit (no target/job
fields), and vectorizer determinism.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from UTILS.features import (
    RICH_MATCH_FEATURE_NAMES,
    build_skills_tfidf,
    candidate_job_match_features,
    candidate_job_match_features_rich,
    education_level_from_text,
    experience_years_from_dates,
    job_experience_min_years,
    load_candidate_profiles,
    parse_resume_dates,
    rich_match_feature_dict,
    skill_categories,
    skills_tfidf_similarity,
    validate_match_features_rich,
)

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "DATA" / "cleaned_jobs.csv"
RAW = ROOT / "DATA" / "resume_data.csv"

PASSED = 0
FAILED = 0
NAME = None


def section(name):
    global NAME
    NAME = name
    print(f"== {name} ==")


def check(desc, cond, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {desc} {extra}")
    else:
        FAILED += 1
        print(f"  [FAIL] {desc} {extra}")


def fails(desc, fn, check_fn, extra=""):
    global PASSED, FAILED
    try:
        fn()
    except Exception as e:
        if check_fn(e):
            PASSED += 1
            print(f"  [PASS] {desc} {extra}")
            return
        FAILED += 1
        print(f"  [FAIL] {desc} wrong exception {e!r} {extra}")
        return
    FAILED += 1
    print(f"  [FAIL] {desc} (no exception raised) {extra}")


def corpus():
    df = pd.read_csv(CLEAN, keep_default_na=False, na_values=[])
    docs, seen = [], set()
    for _, r in df.sort_values("job_title").iterrows():
        if r["job_title"] in seen or not str(r["required_skills"]).strip():
            continue
        seen.add(r["job_title"])
        docs.append(r["required_skills"])
    return docs


# ---------------------------------------------------------------------------
print("RUNNING STEP 6 TESTS")
section("date parser")
check("May 2019 -> (2019,5)", parse_resume_dates("['May 2019']") == [(2019, 5)])
check("2/2019 -> (2019,2)", parse_resume_dates("['2/2019']") == [(2019, 2)])
check("year-only -> (2014,1)", parse_resume_dates("['2014']") == [(2014, 1)])
check("Current anchored", parse_resume_dates("['Current']") == [(2023, 10)])
check("Present/Ongoing/Now anchored",
      parse_resume_dates("['Present','Ongoing','Now']") == [(2023, 10)] * 3)
check("None/N/A -> None entries",
      parse_resume_dates("[None, 'N/A', 'N/A']") == [None, None, None])
check("mixed list aligned",
      parse_resume_dates("['May 2019', None, 'N/A', 'Current', '2/2019']")
      == [(2019, 5), None, None, (2023, 10), (2019, 2)])
check("empty cell -> []", parse_resume_dates('') == [] and parse_resume_dates('[]') == [])
check("non-list/no-crash", isinstance(parse_resume_dates("garbage"), list))

section("experience years")
check("3y + 3y spans = 7.0",
      experience_years_from_dates("['May 2015','Jan 2017']", "['May 2019','Jan 2020']") == 7.0)
check("Current anchor > 0",
      experience_years_from_dates("['May 2015']", "['Current']") > 8.0)
check("only N/A -> None", experience_years_from_dates("['N/A']", "['N/A']") is None)
check("empty starts -> None", experience_years_from_dates('', "['Current']") is None)
check("never negative",
      experience_years_from_dates("['Jan 2020']", "['May 2015']") == 0.0)

section("education level")
check("Bachelor+MBA -> 3", education_level_from_text("Bachelor of Science in CS", "MBA") == 3)
check("PhD -> 4", education_level_from_text("Doctor of Philosophy (PhD)") == 4)
check("Diploma -> 1", education_level_from_text("Diploma in Mechanical") == 1)
check("unknown -> 0", education_level_from_text("") == 0)
check("job MBA text -> 3", education_level_from_text(
    "Master of Business Administration (MBA), Bachelor of Business Administration") == 3)
check("no false 'ms office' hit",
      education_level_from_text("Proficient in ms-office (word/excel/powerpoint)") == 0)

section("job year parser")
check("at least 5 -> 5", job_experience_min_years("At least 5 years") == 5.0)
check("at least 5 year(s) -> 5", job_experience_min_years("At least 5 year(s)") == 5.0)
check("3 to 7 -> 3", job_experience_min_years("3 to 7 years") == 3.0)
check("5 to 8 -> 5", job_experience_min_years("5 to 8 years") == 5.0)
check("empty -> None", job_experience_min_years("") is None)

section("skill categories (special characters preserved)")
c = skill_categories("C++, C#, Python, SQL, HTML/CSS, Node.js, .NET, Excel, Docker")
check("C++ lands in programming", "programming" in c)
check("C# is its own match (no C explosion)",
      "programming" in skill_categories("C#") and skill_categories("C#") == {"programming"})
check(".NET not mangled", "programming" in skill_categories(".NET"))
check("HTML/CSS -> web", "web" in skill_categories("HTML/CSS"))
check("SQL -> database", "database" in skill_categories("SQL"))
check("Excel -> office_tools", "office_tools" in skill_categories("Excel"))
check("Docker -> infrastructure", "infrastructure" in skill_categories("Docker"))
check("empty -> set()", skill_categories("") == set())

section("TF-IDF similarity (shared corpus)")
v = build_skills_tfidf(corpus())
check("identical set sim ~1", skills_tfidf_similarity("C++, Python", "Python, C++", v) > 0.999)
check("permuted set sim ~1",
      skills_tfidf_similarity("Python, SQL, Docker", "SQL, Docker, Python", v) > 0.999)
check("disjoint sim == 0",
      skills_tfidf_similarity("Quickbooks, Excel", "Kubernetes, Terraform", v) == 0.0)
check("empty side -> 0", skills_tfidf_similarity("", "Python, SQL", v) == 0.0
      and skills_tfidf_similarity("Python", "", v) == 0.0)
check("in [0,1]", 0.0 <= skills_tfidf_similarity("Python, C++, SQL", "SQL, Java", v) <= 1.0)
v2 = build_skills_tfidf(corpus())
check("vectorizer deterministic (same corpus)",
      v2.transform(["Python, SQL"]).toarray().shape
      == v.transform(["Python, SQL"]).toarray().shape
      and np.allclose(v2.transform(["Python, SQL"]).toarray(),
                      v.transform(["Python, SQL"]).toarray()))

section("17-feature builder")
prof = {"experience_years": 6.0, "education_level": 3}
vec = candidate_job_match_features_rich(
    "Python, SQL, C++, Docker", "Python, SQL, Docker",
    profile=prof, job_experience_required="3 to 7 years",
    job_education_required="Bachelor of Science", skills_tfidf=v)
check("returns tuple", isinstance(vec, tuple))
check("width 17", len(vec) == 17 == len(RICH_MATCH_FEATURE_NAMES))
check("validates finite", validate_match_features_rich(vec) is vec)
check("first-8 == locked base builder",
      np.allclose(np.asarray(vec[:8], dtype=float),
                  np.asarray(candidate_job_match_features(
                      "Python, SQL, C++, Docker", "Python, SQL, Docker"), dtype=float)))
expect = candidate_job_match_features("Python, SQL, C++, Docker", "Python, SQL, Docker")
check("values exact vs base", tuple(vec[:8]) == expect)
check("cand_exp_years carried", vec[8] == 6.0)
check("exp_gap = 6-3", vec[9] == 3.0)
check("edu_gap = 3-2", vec[11] == 1.0)
check("cat_overlap >= 1", vec[13] >= 1.0)
check("tfidf > 0 here (python+sql shared)", vec[15] > 0.0)
check("has_profile==1", vec[16] == 1.0)

section("safe zeros (no profile / empty sides)")
z = candidate_job_match_features_rich("", "", profile=None,
                                      job_experience_required="", job_education_required="",
                                      skills_tfidf=v)
check("no NaN", all(q == q for q in z))
check("no inf", all(abs(q) < float("inf") for q in z))
check("base fields zero", tuple(z[0:4]) == (0.0, 0.0, 0.0, 0.0))
check("cand_exp=0", z[8] == 0.0 and z[9] == 0.0)
check("cand_edu=0", z[10] == 0.0 and z[11] == 0.0)
check("cats zero", z[12] == 0.0 and z[13] == 0.0 and z[14] == 0.0)
check("tfidf=0", z[15] == 0.0)
check("has_profile=0", z[16] == 0.0)
z2 = candidate_job_match_features_rich(
    "Python, SQL", "Python", profile=None,
    job_experience_required="At least 5 years", job_education_required="Bachelor",
    skills_tfidf=v)
check("missing profile -> zero gaps even when job requires",
      z2[8] == 0.0 and z2[9] == 0.0 and z2[10] == 0.0 and z2[11] == 0.0 and z2[16] == 0.0)

section("dict helper / reporting")
d = rich_match_feature_dict(
    "Python, SQL", "Python, SQL", profile=prof,
    job_experience_required="At least 2 years", job_education_required="Bachelor",
    skills_tfidf=v)
check("dict keys == RICH_MATCH_FEATURE_NAMES order",
      list(d.keys()) == RICH_MATCH_FEATURE_NAMES)

section("profile loader (read-only candidate fields, full join)")
profs = load_candidate_profiles(RAW)
df = pd.read_csv(CLEAN, keep_default_na=False, na_values=[])
check("340 profiles", len(profs) == df["candidate_skills"].nunique() == 340)
check("every clean candidate_skills joins", set(df["candidate_skills"]) <= set(profs))
check("profile values are float/int or None", all(
    (p["experience_years"] is None or isinstance(p["experience_years"], (int, float)))
    and isinstance(p["education_level"], int) and 0 <= p["education_level"] <= 4
    for p in profs.values()))
check("profiles do not contain target/job fields",
      all(set(p) <= {"experience_years", "education_level"} for p in profs.values()))

section("leakage audit")
check("no feature name is the target or title",
      not any(k in ("matched_score", "job_title") for k in RICH_MATCH_FEATURE_NAMES))
check("builder signature has no target/title args",
      not any(k in str(candidate_job_match_features_rich.__code__.co_varnames)
              for k in ("matched_score", "job_title")))
check("RICH names extend the locked 8 with 9 new",
      RICH_MATCH_FEATURE_NAMES[:8] == [
          "overlap_count", "jaccard", "req_coverage", "cand_precision",
          "candidate_count", "required_count", "candidate_has_skills",
          "job_has_requirements"] and len(RICH_MATCH_FEATURE_NAMES) == 17)

section("validator rejects bad vectors")
fails("rejects 16/18 length", lambda: validate_match_features_rich((0.0,) * 16),
      lambda e: isinstance(e, ValueError))
fails("rejects string member", lambda: validate_match_features_rich((0.0,) * 16 + ("x",)),
      lambda e: isinstance(e, ValueError))
fails("rejects nan", lambda: validate_match_features_rich((0.0,) * 16 + (float("nan"),)),
      lambda e: isinstance(e, ValueError))
fails("rejects inf", lambda: validate_match_features_rich((0.0,) * 16 + (float("inf"),)),
      lambda e: isinstance(e, ValueError))

print(f"\n==== {PASSED} passed, {FAILED} failed ====")
sys.exit(1 if FAILED else 0)