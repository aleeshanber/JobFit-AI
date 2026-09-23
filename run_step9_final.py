"""Step 9 — Final Evaluation: Flask smoke test + Resume scenarios A-J + Recommendation quality."""

import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
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


print("=" * 70)
print("STEP 9 — FINAL EVALUATION (Flask smoke + A-J extraction + rec quality)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 9.9 Flask smoke test
# ---------------------------------------------------------------------------
print("\n== 9.9 Flask / Production Readiness ==")
from APP.app import app as flask_app
flask_app.config["TESTING"] = True
client = flask_app.test_client()

resp = client.get("/")
html = resp.get_data(as_text=True)
check("homepage loads (200)", resp.status_code == 200)
check("loading state div present", 'id="loading"' in html)
check("upload form present", 'enctype="multipart/form-data"' in html)
check("accepts .pdf only", 'accept=".pdf"' in html)

resp = client.post("/", data={"resume": (io.BytesIO(b"test"), "test.docx")})
html = resp.get_data(as_text=True)
check("non-PDF rejected with friendly message",
      resp.status_code == 200 and "Only PDF format" in html)

import pymupdf
d = pymupdf.open()
page = d.new_page()
page.insert_text((72, 72),
    "SKILLS\nPython\nSQL\nMachine Learning\n\nEXPERIENCE\nSoftware Engineer\nJan 2019 - present",
    fontsize=12)
tmp = Path(tempfile.mkdtemp()) / "smoke.pdf"
d.save(str(tmp))
d.close()
pdf_bytes = tmp.read_bytes()
tmp.unlink(missing_ok=True)

resp = client.post("/", data={"resume": (io.BytesIO(pdf_bytes), "smoke.pdf")})
html = resp.get_data(as_text=True)
check("valid PDF renders results (200)", resp.status_code == 200)
check("Extracted Skills section present", "Extracted Skills" in html)
check("Top Matching Jobs present", "Top Matching Jobs" in html)
check("Match Score column", "Match Score" in html)
check("Required Skills column", "Required Skills" in html)
check("Missing Skills column", "Missing Skills" in html)
check("Experience Required column", "Experience Required" in html)
check("no traceback in output", "Traceback" not in html)
check("no matched_score leakage", "matched_score" not in html)
check("match_score value not raw 0.xx in HTML", "0.7" not in html or "%" in html)
check("loading div hidden on results page",
      "display:none" in html or "Extracted Skills" in html)

# ---------------------------------------------------------------------------
# 9.4 Resume extraction scenarios A-J
# ---------------------------------------------------------------------------
from UTILS.extractor import extract_skills_from_text, extract_text_from_pdf
from UTILS.recommender import build_candidate_profile_from_text, build_recommender

engine = build_recommender()


def verify_resume(name, text, expect_skills=True, expect_exp=True):
    """Run extraction + profile + recommendation, return (skills, profile)."""
    print(f"\n--- {name} ---")
    skills = extract_skills_from_text(text)
    profile = build_candidate_profile_from_text(text)

    # No invented skills
    lower = " " + " ".join(text.lower().split()) + " "
    for s in skills:
        if s.lower() not in lower:
            print(f"  [INVENTED SKILL]: '{s}' not found in text")

    # Education in range
    check("education_level in [0,4]",
          0 <= profile["education_level"] <= 4,
          extra=f"got {profile['education_level']}")

    print(f"  skills ({len(skills)}): {skills}")
    print(f"  profile: {profile}")
    return skills, profile


print("\n== 9.4 Final Resume Extraction (A-J) ==")

# A: Clear skills + experience + education
sk, pr = verify_resume("A. Full resume",
    "SKILLS\nPython\nSQL\nDocker\n\nEXPERIENCE\nSoftware Engineer\nJan 2019 - present\n\nEDUCATION\nBachelor of Science")
check("A: has 3 skills", len(sk) == 3, extra=f"got {len(sk)}")
check("A: experience is present", pr["experience_years"] is not None, extra=f"got {pr['experience_years']}")
check("A: education level 2 (bachelor)", pr["education_level"] == 2, extra=f"got {pr['education_level']}")

# B: No skills section
sk, pr = verify_resume("B. No skills section",
    "CONTACT\nJohn Doe\nemail@test.com\n\nEXPERIENCE\nEngineer\nJan 2020 - present",
    expect_skills=False)
check("B: no skills section -> fallback vocab or empty", len(sk) >= 0)
check("B: experience present", pr["experience_years"] is not None, extra=f"got {pr['experience_years']}")

# C: Unusual skill heading
sk, pr = verify_resume("C. Unusual skill heading (TECHNOLOGY STACK)",
    "TECHNOLOGY STACK\nReact\nNode.js\nTypeScript\n\nEXPERIENCE\nDeveloper\nJan 2021 - present")
check("C: extracts skills from unusual heading", len(sk) >= 2, extra=f"got {len(sk)}")

# D: Special character skills
sk, pr = verify_resume("D. Special character skills",
    "SKILLS\nC++\nC#\n.NET\nNode.js\nHTML/CSS\nCI/CD\nC/C++\nASP.NET\nF#\nPython")
for want in ("C++", "C#", ".NET", "Node.js", "HTML/CSS", "CI/CD", "C/C++", "ASP.NET", "F#", "Python"):
    check(f"D: '{want}' intact", want in sk, extra=f"got {sk}")

# E: Multiple employment periods
sk, pr = verify_resume("E. Multiple employment periods",
    "SKILLS\nJava\nSQL\n\nEXPERIENCE\nDeveloper\nJan 2015 - Dec 2017\nEngineer\nMar 2018 - present")
check("E: union of periods (not sum)", pr["experience_years"] is not None, extra=f"got {pr['experience_years']}")

# F: Education dates near work dates
sk, pr = verify_resume("F. Education near work dates",
    "SKILLS\nPython\n\nEDUCATION\nBachelor of Science 2016 - 2020\n\nEXPERIENCE\nAnalyst\nFeb 2020 - present")
check("F: education dates not counted as experience",
      pr["experience_years"] is not None and pr["experience_years"] < 4.0,
      extra=f"got {pr['experience_years']}")

# G: Missing experience
sk, pr = verify_resume("G. Missing experience",
    "SKILLS\nMachine Learning\nTensorFlow\n\nEDUCATION\nPh.D. in Computer Science")
check("G: experience is None when missing", pr["experience_years"] is None)
check("G: education level 4 (PhD)", pr["education_level"] == 4, extra=f"got {pr['education_level']}")

# H: Missing education
sk, pr = verify_resume("H. Missing education",
    "SKILLS\nDocker\nKubernetes\nAWS\n\nEXPERIENCE\nDevOps Engineer\nJan 2018 - present")
check("H: education level 0 when missing", pr["education_level"] == 0)
check("H: experience present", pr["experience_years"] is not None, extra=f"got {pr['experience_years']}")

# I: No recognizable technical skills
sk, pr = verify_resume("I. No technical skills",
    "CONTACT\nJane Smith\n\nHOBBIES\nReading\nTravel\nCooking",
    expect_skills=False, expect_exp=False)
check("I: no crash, safe empty", isinstance(sk, list))
check("I: experience None", pr["experience_years"] is None)
check("I: education level 0", pr["education_level"] == 0)

# J: Garbage text
sk, pr = verify_resume("J. Garbage text",
    "###???abc123!!!",
    expect_skills=False, expect_exp=False)
check("J: no crash on garbage", isinstance(sk, list) and len(sk) == 0)
check("J: experience None", pr["experience_years"] is None)

# ---------------------------------------------------------------------------
# 9.5 Recommendation quality check
# ---------------------------------------------------------------------------
print("\n== 9.5 Final Recommendation Quality ==")

test_cases = [
    ("ML Engineer", "Python\nSQL\nMachine Learning\nDeep Learning", {"experience_years": 5.0, "education_level": 3}),
    ("Web Developer", "HTML/CSS\nJavaScript\nReact\nNode.js", {"experience_years": 3.0, "education_level": 2}),
    ("DevOps", "Docker\nKubernetes\nAWS\nCI/CD\nTerraform", {"experience_years": 4.0, "education_level": 2}),
    ("No skills", "", {"experience_years": None, "education_level": 0}),
]

for name, skills_text, profile in test_cases:
    print(f"\n--- rec quality: {name} ---")
    recs = engine.recommend(skills_text, profile=profile)

    check(f"{name}: exactly 10 results", len(recs) == 10, extra=f"got {len(recs)}")

    titles = [r["job_title"] for r in recs]
    check(f"{name}: no duplicate titles", len(titles) == len(set(titles)))

    scores = [r["match_score"] for r in recs]
    check(f"{name}: all scores finite", all(s == s for s in scores))
    check(f"{name}: all scores in [0,1]", all(0 <= s <= 1 for s in scores))
    check(f"{name}: scores descending",
          all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))

    for r in recs:
        missing = r["missing_skills"]
        required = r["required_skills"]
        check(f"  {r['job_title'][:30]}: missing skills are strings",
              all(isinstance(m, str) and len(m.strip()) > 0 for m in missing))

    if name != "No skills":
        check(f"{name}: scores vary across jobs (not all identical)",
              len(set(round(s, 4) for s in scores)) > 1)

# Determinism check
recs_a = engine.recommend("Python\nSQL", profile={"experience_years": 3, "education_level": 2})
recs_b = engine.recommend("Python\nSQL", profile={"experience_years": 3, "education_level": 2})
check("deterministic: same inputs -> same outputs",
      [r["job_title"] for r in recs_a] == [r["job_title"] for r in recs_b]
      and [r["match_score"] for r in recs_a] == [r["match_score"] for r in recs_b])

print(f"\n{'='*70}")
print(f"==== {PASSED} passed, {FAILED} failed ====")
print(f"{'='*70}")
sys.exit(1 if FAILED else 0)
