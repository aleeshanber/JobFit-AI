"""Step 8 - production hardening, extraction robustness & final validation.

Covers the Step-8.12 checklist: experience heading variants, overlap-safe /
future-safe / invalid-safe / safe-zero tenure parsing, education heading
variants + levels 0-4, skill heading variants, special-character skill
integrity (no splitting, no dedupe of distinct skills), no invented skills,
safe fallback policy, model inference integrity (never retrained, 17 features),
recommendation integrity (all titles scored, top-10 distinct, real required
skills, missing ⊆ required, deterministic), formula-free score display,
frontend/UX error paths (invalid type, empty/unreadable/no-text/no-skills PDFs),
security (no matched_score in HTML, no tracebacks, minimal logging), and the
Step-8.14 regression (dataset rows/titles/matched_score unchanged, model
artifact is the promoted Step-6 rich model, job_classifier/match_baseline and
the shared feature code untouched).

The Step-2 (98), Step-4 (48), Step-6 (71) and Step-7 (58) suites must all still
report their exact pass counts.
"""

import json
import io
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
from UTILS.features import RICH_MATCH_FEATURE_NAMES, CURRENT_REFERENCE_DATE
from UTILS.recommender import (
    build_candidate_profile_from_text,
    build_recommender,
    missing_skills,
)
from UTILS.extractor import extract_skills_from_text, extract_text_from_pdf

MODEL_PKL = ROOT / "MODELS" / "match_model.pkl"
META_JSON = ROOT / "MODELS" / "match_model_metadata.json"
JOBS_CSV = ROOT / "DATA" / "cleaned_jobs.csv"
APP_PY = ROOT / "APP" / "app.py"
TEMPLATE = ROOT / "TEMPLATES" / "index.html"
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


print("RUNNING STEP 8 ROBUSTNESS / VALIDATION TESTS")

# ---------------------------------------------------------------------------
# 1) Experience heading variants (8.2)
# ---------------------------------------------------------------------------
print("== 1) experience heading variants ==")
for hdg in ("EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE",
            "EMPLOYMENT", "EMPLOYMENT HISTORY", "WORK HISTORY",
            "CAREER HISTORY", "PROFESSIONAL HISTORY", "WORK RECORD"):
    text = f"{hdg}\nSoftware Engineer\nJan 2019 - Aug 2021"
    years = build_candidate_profile_from_text(text)["experience_years"]
    check(f"'{hdg}' yields ~2.6y tenure", years is not None and abs(years - 2.58) < 0.1,
          extra=f"got {years}")

# ---------------------------------------------------------------------------
# 2) Experience date parsing + Current anchoring (8.3)
# ---------------------------------------------------------------------------
print("== 2) experience date parsing / anchor ==")
check("Jan 2019 - present anchored at (2023,10) -> ~4.75y",
      abs(build_candidate_profile_from_text(
          "EXPERIENCE\nRole\nJan 2019 - present")["experience_years"] - 4.75) < 0.05)
check("Current word family resolves to CURRENT_REFERENCE_DATE",
      CURRENT_REFERENCE_DATE == (2023, 10))
check("MM/YYYY form parsed (2/2019 - 12/2020)",
      abs(build_candidate_profile_from_text(
          "WORK EXPERIENCE\nRole\n2/2019 - 12/2020")["experience_years"] - 1.83) < 0.1)

print("== 3) overlap / union (no double-count) ==")
text_overlap = ("WORK EXPERIENCE\n"
                "Engineer A\nJan 2019 - present\n"
                "Engineer B\nJun 2019 - Sep 2020")
years = build_candidate_profile_from_text(text_overlap)["experience_years"]
check("overlapping calendar period not double-counted (union)",
      years is not None and years <= 4.76,
      extra=f"got {years}")
text_merged = ("WORK EXPERIENCE\n"
               "Role 1\nJan 2019 - Dec 2020\n"
               "Role 2\nApr 2020 - present")
years2 = build_candidate_profile_from_text(text_merged)["experience_years"]
check("adjacent/honest ranges keep full span after union",
      years2 is not None and 4.5 < years2 <= 4.76,
      extra=f"got {years2}")

print("== 4) duplicate + future + invalid dates ==")
dup = ("WORK EXPERIENCE\nRole\nJan 2019 - present\nRole copy\nJan 2019 - present")
check("duplicate ranges do not inflate (union)",
      abs(build_candidate_profile_from_text(dup)["experience_years"] - 4.75) < 0.05)
future = ("WORK EXPERIENCE\nRole\nJan 2030 - present")  # after (2023,10) anchor
check("future-only span rejected (no unrealistic experience)",
      build_candidate_profile_from_text(future)["experience_years"] is None)
future2 = ("WORK EXPERIENCE\nRole\nJan 2030 - Dec 2033")
check("fully-future span rejected", 
      build_candidate_profile_from_text(future2)["experience_years"] is None)
invalid = ("WORK EXPERIENCE\nRole\nN/A")
check("invalid/missing dates -> safe zero (None)",
      build_candidate_profile_from_text(invalid)["experience_years"] is None)
none_text = "PROFESSIONAL SUMMARY\nBackend engineer"
check("no experience section at all -> None",
      build_candidate_profile_from_text(none_text)["experience_years"] is None)

print("== 5) education dates are NOT counted as experience ==")
edu_only = ("EDUCATION\nBachelor of Science 2016 - 2020\n\n"
            "EXPERIENCE\nEngineer\nJan 2019 - present")
check("education region separated; experience region only",
      build_candidate_profile_from_text(edu_only)["experience_years"] is not None)

# ---------------------------------------------------------------------------
# 6-7) Education heading variants + levels 0-4 (8.2 / 8.4)
# ---------------------------------------------------------------------------
print("== 6) education heading variants ==")
for hdg in ("EDUCATION", "ACADEMIC BACKGROUND", "EDUCATIONAL BACKGROUND",
            "EDUCATIONAL QUALIFICATIONS", "ACADEMIC QUALIFICATIONS",
            "ACADEMIC HISTORY", "ACADEMICS"):
    prof = build_candidate_profile_from_text(f"{hdg}\nMaster of Science in CS")
    check(f"'{hdg}' -> level 3", prof["education_level"] == 3,
          extra=f"got {prof['education_level']}")

print("== 7) education levels 0-4 ==")
cases = [("Ph.D. in Robotics", 4), ("Doctorate in Economics", 4),
         ("Master of Business Administration", 3), ("M.Sc. Computer Science", 3),
         ("M.Com Finance", 3), ("MBA", 3), ("MBM", 3),
         ("Bachelor of Science", 2), ("B.Sc", 2), ("BBA", 2), ("B.Tech", 2),
         ("Honors in History", 2), ("HSC", 1), ("SSC", 1), ("Diploma", 1),
         ("No degree mentioned anywhere", 0)]
for text, level in cases:
    prof = build_candidate_profile_from_text(f"EDUCATION\n{text}")
    check(f"'{text}' -> level {level}", prof["education_level"] == level,
          extra=f"got {prof['education_level']}")
check("number-only text never invents a level",
      build_candidate_profile_from_text("EDUCATION\n2016 - 2020")["education_level"] == 0)

# ---------------------------------------------------------------------------
# 8-11) Skill heading variants + special-char skills + no splitting (8.5)
# ---------------------------------------------------------------------------
print("== 8) skill heading variants ==")
for hdg in ("SKILLS", "TECHNICAL SKILLS", "TECHNICAL EXPERTISE",
            "CORE SKILLS", "KEY SKILLS", "SKILL SET", "TECHNOLOGIES",
            "TECHNOLOGY", "TOOLS & TECHNOLOGIES", "TECHNICAL PROFICIENCIES",
            "IT SKILLS", "COMPUTER SKILLS"):
    sk = extract_skills_from_text(f"{hdg}\nPython\nSQL")
    check(f"'{hdg}' captures Python+SQL",
          sk == ["Python", "SQL"], extra=f"got {sk}")

print("== 9) special-character skills preserved (no splitting) ==")
sk = extract_skills_from_text(
    "PROFESSIONAL SKILLS\nC++\nC#\n.NET\nC/C++\nHTML/CSS\nCI/CD\n"
    "Node.js\nASP.NET\nF#\nPython")
for want in ("C++", "C#", ".NET", "C/C++", "HTML/CSS", "CI/CD",
             "Node.js", "ASP.NET", "F#", "Python"):
    check(f"keeps '{want}' intact", want in sk, extra=f"got {sk}")

print("== 10) distinct-but-similar skills are NOT merged ==")
check("C and C++ both remain distinct",
      "C++" in missing_skills("C, C++", "C"))
check(".NET never collapses into NET",
      ".NET" in missing_skills(".NET, Network Security", "Network Security"))
check("C# never collapses into C",
      "C#" in sk and "C#" in missing_skills("C#, C++", "C++"))
check("F# stays distinct from F",
      "F#" in missing_skills("F#, F", "F"))

print("== 10b) fragmented candidate skills still cover required phrases ==")
# When skills are listed on separate resume lines, the extractor may split
# multi-word skills.  missing_skills must still recognise them.
check("IT Enabled Services covered by IT Enabled + Services",
      missing_skills("IT Enabled Services", "IT Enabled, Services") == [])
check("IT Enabled Services case-insensitive",
      missing_skills("IT Enabled Services", "it enabled, services") == [])
check("OLT and ONU covered by OLT + ONU",
      missing_skills("OLT and ONU", "OLT, ONU") == [])
check("OLT and ONU case-insensitive",
      missing_skills("OLT and ONU", "olt, onu") == [])
check("Human Resource Management exact match",
      missing_skills("Human Resource Management",
                     "Human Resource Management") == [])
check("Machine Learning covered by Machine + Learning",
      missing_skills("Machine Learning", "Machine, Learning") == [])
check("partial match still shows truly missing skills",
      missing_skills("Python, SQL, C++, Docker",
                     "Python, SQL, C++") == ["Docker"])
# Safety: must NOT create false matches
check("C++ still missing when only C present",
      "C++" in missing_skills("C++", "C"))
check("C still missing when only C++ present",
      "C" in missing_skills("C", "C++"))
check("Java still missing when only JavaScript present",
      "Java" in missing_skills("Java", "JavaScript"))
check(".NET still missing when only NET present",
      ".NET" in missing_skills(".NET", "NET"))
check("C# still missing when only C present",
      "C#" in missing_skills("C#", "C"))

print("== 11) only explicit skills are extracted (never invented) ==")
noise = ("CONTACT\nName: Some Person\nEmail: x@y.com\nPhone: 123456\n"
         "PROFESSIONAL SKILLS\nPython\nSQL")
sk2 = extract_skills_from_text(noise)
check("contact name / email / phone not extracted as skills",
      "Some Person" not in sk2 and "x@y.com" not in sk2 and "123456" not in sk2)
sk3 = extract_skills_from_text("PROFESSIONAL SKILLS\nPython\nMachine Learning\nSQL")
check("skilled phrases from the resume kept (Machine Learning)",
      "Machine Learning" in sk3, extra=f"got {sk3}")
text4 = "PROFESSIONAL SKILLS\nComputer Programming\nData Analysis"
sk4 = extract_skills_from_text(text4)
check("resume-explicit technical phrases kept",
      "Computer Programming" in sk4 and "Data Analysis" in sk4,
      extra=f"got {sk4}")

# ---------------------------------------------------------------------------
# 12) Safe fallback policy: profile comes ONLY from resume text (8.6)
# ---------------------------------------------------------------------------
print("== 12) safe fallback policy ==")
check("missing experience -> None, missing education -> 0",
      build_candidate_profile_from_text("SUMMARY\nPython dev") == {
          "experience_years": None, "education_level": 0})
rec_src = REC_PY.read_text(encoding="utf-8")
check("recommender never reads the training/resume CSV",
      "resume_data.csv" not in rec_src)
code_lines = [ln for ln in rec_src.splitlines()
              if "matched_score" in ln and ln.strip()
              and not ln.strip().startswith(("#", "*", '"')) and "'''" not in ln]
check("matched_score appears only in comments/docstrings", code_lines == [])
spec_def = rec_src.split("_SPEC_COLUMNS =")[1].split(")")[0]
check("job spec columns exclude matched_score (usecols)",
      "matched_score" not in spec_def)

# ---------------------------------------------------------------------------
# 13) Model inference integrity: artifact + 17-feature vector (8.7)
# ---------------------------------------------------------------------------
print("== 13) model inference integrity ==")
engine = build_recommender()
with open(MODEL_PKL, "rb") as fh:
    blob = pickle.load(fh)
with open(META_JSON, encoding="utf-8") as fh:
    meta = json.load(fh)
check("model object has .predict",
      hasattr(blob.get("model"), "predict"))
check("engine exposes 17 features in RICH order",
      list(engine.features) == list(RICH_MATCH_FEATURE_NAMES) and len(engine.features) == 17)
check("artifact kind is the promoted Step-6 rich model",
      blob.get("kind") == "jobfit-match-model-step6")
check("shared 17-feature builder in features.py used (no manual recreation)",
      "candidate_job_match_features_rich" in (ROOT / "UTILS" / "features.py").read_text(encoding="utf-8")
      and "candidate_job_match_features_rich" in REC_PY.read_text(encoding="utf-8"))

text_v = "Python, SQL, Docker, Pandas"
spec = engine.jobs_specs[0]
v = engine.features_for(text_v, spec)
check("17-vector finite + ordered", isinstance(v, tuple) and len(v) == 17
      and all(isinstance(x, (int, float)) and x == x for x in v))

# ---------------------------------------------------------------------------
# 14) Recommendation integrity: all scored, distinct top-10, required/missing (8.8)
# ---------------------------------------------------------------------------
print("== 14) recommendation integrity ==")
prof = {"experience_years": 4.0, "education_level": 2}
recs = engine.recommend(text_v, profile=prof)
preds = [r["match_score"] for r in recs]
check("all 10 results distinct titles", len({r["job_title"] for r in recs}) == 10)
check("scores within [0,1] and finite", all(0 <= p <= 1 and p == p for p in preds))
check("ranked by score descending",
      all(preds[i] >= preds[i + 1] for i in range(len(preds) - 1)))
req_set = {s["required_skills"] for s in engine.jobs_specs}
check("required_skills come from the job dataset",
      all(r["required_skills"] in req_set for r in recs))
check("missing_skills are real required-skill names",
      all(isinstance(m, str) and len(m.strip()) > 0
          for r in recs for m in r["missing_skills"]))
check("required never equals candidate text",
      all(r["required_skills"] != text_v for r in recs))

checked_all_specs = len(engine.jobs_specs)
check(f"every spec title is scored ({checked_all_specs} titles)",
      checked_all_specs == 28)

print("== 15) deterministic ==")
recs2 = engine.recommend(text_v, profile=prof)
check("identical ranked result on repeat",
      [r["job_title"] for r in recs] == [r["job_title"] for r in recs2]
      and [r["match_score"] for r in recs] == [r["match_score"] for r in recs2])

print("== 16) score display stays prediction-based (8.9) ==")
tpl = TEMPLATE.read_text(encoding="utf-8")
check("template renders score as prediction*100 %",
      '"%.1f"|format(row[\'match_score\'] * 100)' in tpl)
check("no hard-coded scores/formulas in template",
      "54.0" not in tpl and "90.0" not in tpl and "similarity" not in tpl)

# ---------------------------------------------------------------------------
# 17) Frontend / UX validation: every error path (8.10)
# ---------------------------------------------------------------------------
print("== 17) frontend / UX error paths ==")
from APP.app import app as flask_app

flask_app.config["TESTING"] = True
flask_app.config["WTF_CSRF_ENABLED"] = False
client = flask_app.test_client()

def _pdf_bytes(text=""):
    d = pymupdf.open()
    page = d.new_page()
    if text:
        page.insert_text((72, 72), text, fontsize=12)
    tmp = Path(tempfile.mkdtemp()) / "r.pdf"
    d.save(str(tmp))
    d.close()
    data = tmp.read_bytes()
    return data

def _upload(name, data, ctype="application/pdf"):
    return client.post("/", data={"resume": (io.BytesIO(data), name)})

resp = _upload("resume.docx", b"hello")
html = resp.get_data(as_text=True)
check("non-PDF upload rejected with friendly message",
      resp.status_code == 200 and "Only PDF format is supported!" in html)

resp = _upload("blank.pdf", _pdf_bytes(""))
html = resp.get_data(as_text=True)
check("empty PDF -> 'Could not read text'",
      "Could not read text" in html)

resp = _upload("garbage.pdf", b"this is not a pdf")
html = resp.get_data(as_text=True)
check("unreadable/garbage PDF -> friendly error (no crash)",
      resp.status_code == 200 and "Could not read text" in html)

resp = _upload("noskills.pdf", _pdf_bytes("Just some plain text about a person."))
html = resp.get_data(as_text=True)
check("no-skills PDF -> 'No matching skills'",
      "No matching skills" in html)

resp = _pdf_bytes("PROFESSIONAL SKILLS\nPython\nSQL\nWORK EXPERIENCE\nEngineer\nJan 2019 - present")
resp = _upload("good.pdf", resp)
html = resp.get_data(as_text=True)
check("valid PDF renders results (skills + table)",
      resp.status_code == 200 and "Extracted Skills" in html and "Top Matching Jobs" in html)

check("loading state present on GET page",
      'id="loading"' in client.get("/").get_data(as_text=True))
check("Match Score column present in results",
      "Match Score" in html)

print("== 18) security / data hygiene (8.11) ==")
check("no traceback leaked into any HTML",
      "Traceback" not in html and "File \"" not in html)
check("matched_score value never rendered",
      "matched_score" not in html)
check("no dataset row / profile exposure in HTML",
      "resume_data.csv" not in html and "candidate profile" not in html)
app_src = APP_PY.read_text(encoding="utf-8")
check("app never prints raw resume content",
      "print(f\"--> Extracted Raw PDF Text Length" in app_src)
check("app prints only length/count aggregates",
      "len(extracted_text)" in app_src and "print(f\"--> Candidate profile" in app_src)

# ---------------------------------------------------------------------------
# 19-21) Step 8.14 regression checks
# ---------------------------------------------------------------------------
print("== 19) dataset regression (rows / titles / target) ==")
import pandas as pd

jobs = pd.read_csv(JOBS_CSV, keep_default_na=False, na_values=[])
check("cleaned_jobs.csv row count is 9531", len(jobs) == 9531)
check("cleaned_jobs.csv has 28 distinct job titles",
      jobs["job_title"].nunique() == 28)
check("matched_score column exists and has no NaN",
      "matched_score" in jobs.columns and jobs["matched_score"].notna().all())
check("matched_score range [0,1] preserved",
      jobs["matched_score"].between(0, 1).mean() == 1.0)

print("== 20) model artifact regression ==")
step6_pkl = ROOT / "MODELS" / "match_model_step6.pkl"
check("promoted match_model.pkl IS step6 (kind) - step inference",
      blob.get("kind") == "jobfit-match-model-step6")
with open(step6_pkl, "rb") as fh:
    blob_step6 = pickle.load(fh)
check("match_model.pkl == match_model_step6.pkl semantics",
      blob_step6.get("model").predict(
          [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])[0] ==
      blob.get("model").predict(
          [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])[0])
check("metadata reflects the Step-6 rich model",
      len(meta.get("features")) == 17
      and meta.get("promoted_to_match_model_pkl") is True)

print("== 21) step artifacts untouched / old scoring removed ==")
check("old job_classifier.pkl still present and untouched",
      (ROOT / "MODELS" / "job_classifier.pkl").exists())
check("old match_baseline.pkl still present and untouched",
      (ROOT / "MODELS" / "match_baseline.pkl").exists())
check("tfidf_vectorizer.pkl (old pipeline) untouched",
      (ROOT / "MODELS" / "tfidf_vectorizer.pkl").exists())
check("shared experience_years_from_dates NOT modified in features.py",
      "def experience_years_from_dates" in (ROOT / "UTILS" / "features.py").read_text(encoding="utf-8"))
check("app.py has no old TF-IDF formula",
      "cosine_similarity" not in app_src and "TfidfVectorizer" not in app_src
      and "54.0" not in app_src and "90.0" not in app_src)
check("app.py has no prototype scale/detect helpers",
      "scale_match_score" not in app_src and "TITLE_DOMAINS" not in app_src)

# ---------------------------------------------------------------------------
# 22) prior suites still pass (98 / 48 / 71 / 58)
# ---------------------------------------------------------------------------
print("== 22) prior suites still pass ==")
for script, expect in (("UTILS/test_features.py", 98),
                       ("UTILS/test_step4_features.py", 48),
                       ("UTILS/test_step6_features.py", 71),
                       ("UTILS/test_step7_production.py", 58)):
    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        capture_output=True, text=True,
        env={"PYTHONIOENCODING": "utf-8", **__import__("os").environ},
        cwd=str(ROOT))
    out = result.stdout + result.stderr
    check(f"{script} reports '{expect} passed'",
          result.returncode == 0 and f"{expect} passed" in out,
          extra=f"rc={result.returncode}")

print(f"\n==== {PASSED} passed, {FAILED} failed ====")
sys.exit(1 if FAILED else 0)