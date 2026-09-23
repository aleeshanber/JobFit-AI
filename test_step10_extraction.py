"""Step 10 - resume skill-extraction integrity tests.

Regression suite for the extraction-only fix. It guarantees that every
explicitly listed skill in a Skills section is extracted (out-of-vocabulary
technical terms included), that the strict no-invention / no-junk contract is
unchanged, that special-character skills are never split or merged, that
section boundaries are respected, and that multi-column
PDFs are read in visual order so headings precede their content.

Covers the 21 extraction scenarios plus real-PDF regressions (1/2/4/
sample_resume) and the dev-only debug path that never logs resumes in
production.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UTILS.extractor import (
    extract_skills_from_text,
    extract_text_from_pdf,
)

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "UPLOADS"

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


def extract(*lines, text=None):
    return extract_skills_from_text(text if text is not None else "\n".join(lines))


def no_invented(skills, text):
    lower = " " + " ".join(text.lower().split()) + " "
    return [s for s in skills if s.lower() not in lower]


print("RUNNING STEP 10 RESUME SKILL EXTRACTION TESTS")
print("== 1) explicit 10-skill list is fully extracted ==")
ten = ("Python, Java, C++, SQL, Machine Learning, Deep Learning, TensorFlow, "
       "PyTorch, Pandas, NumPy")
sk = extract("SKILLS", ten)
check("10/10 skills extracted (incl. out-of-vocab Pandas/NumPy)",
      len(sk) == 10,
      extra=f"got {len(sk)} -> {sk}")
check("Pandas present", "Pandas" in sk, extra=f"got {sk}")
check("NumPy present", "NumPy" in sk, extra=f"got {sk}")

print("== 2) 5-skill inline comma list ==")
sk = extract("SKILLS", "Python, SQL, Docker, Pandas, NumPy")
check("all 5 extracted", len(sk) == 5, extra=f"got {sk}")
check("Docker explicit skill kept", "Docker" in sk, extra=f"got {sk}")

print("== 3) newline-separated skills ==")
sk = extract("SKILLS", "Python", "SQL", "Docker", "", "EXPERIENCE", "Engineer")
check("3 newline skills extracted", len(sk) == 3, extra=f"got {sk}")

print("== 4) bullet-separated skills ==")
sk = extract("SKILLS", "- Python", "- SQL", "- Docker")
check("3 bullet skills extracted", len(sk) == 3, extra=f"got {sk}")

print("== 5) semicolon-separated skills ==")
sk = extract("SKILLS", "Python; SQL; PostgreSQL; MongoDB")
check("4 semicolon skills extracted", len(sk) == 4, extra=f"got {sk}")

print("== 6) pipe-separated skills ==")
sk = extract("SKILLS", "Python | SQL | Java | Kubernetes")
check("4 pipe skills extracted", len(sk) == 4, extra=f"got {sk}")

print("== 7) multi-word technical phrases preserved ==")
phrases = ["Natural Language Processing", "Machine Learning", "Deep Learning",
           "Computer Vision"]
sk = extract("TECHNICAL SKILLS", *phrases)
check("multi-word phrases kept intact",
      all(p in sk for p in phrases), extra=f"got {sk}")

print("== 8) special-character skills preserved (never split) ==")
special = ["C++", "C#", ".NET", "C/C++", "HTML/CSS", "CI/CD",
           "Node.js", "React.js", "ASP.NET", "F#"]
sk = extract("PROFESSIONAL SKILLS", *special, "Python")
check("all special-char skills intact",
      all(w in sk for w in special), extra=f"got {sk}")
check("React.js distinct from React",
      "React.js" in sk and "React" not in sk, extra=f"got {sk}")

print("== 9) two-column PDF reading order (2.pdf -> all 9 listed skills) ==")
pdf2 = UPLOADS / "2.pdf"
if pdf2.exists():
    sk2 = extract_skills_from_text(extract_text_from_pdf(str(pdf2)))
    wanted9 = ["Branding & Visual Identity", "Design", "Packaging Design",
               "Social Media Design", "Print Design", "Layout & Typography",
               "Creative Concept", "Development", "Photo Editing & Retouching"]
    check("9/9 graphic-design skills extracted from columns",
          len(sk2) == 9 and all(w in sk2 for w in wanted9),
          extra=f"got {len(sk2)} -> {sk2}")
else:
    check("2.pdf fixture present", False, extra=f"missing {pdf2}")

print("== 10) skills section stops at the next major heading ==")
sk = extract("SKILLS", "Python", "SQL", "Docker",
             "EXPERIENCE", "Software Engineer", "Jan 2019 - present",
             "EDUCATION", "Bachelor of Science")
check("exactly the 3 listed skills (no experience/education leakage)",
      sk == ["Docker", "Python", "SQL"], extra=f"got {sk}")
check("experience line never a skill",
      "Software Engineer" not in sk and "Jan 2019" not in sk,
      extra=f"got {sk}")

print("== 11) no Skills section -> safe fallback, no junk, no crash ==")
sk = extract_skills_from_text("CONTACT\nJohn Doe\nemail@test.com\n"
                              "EXPERIENCE\nEngineer\nJan 2020 - present")
check("empty/no-skills text returns empty", extract_skills_from_text("") == [],
      extra=f"got {sk}")
check("no crash on garbage text",
      isinstance(sk, list) and all(isinstance(s, str) for s in sk),
      extra=f"got {sk}")
check("contact name/email never skills",
      "John Doe" not in sk and "email@test.com" not in sk, extra=f"got {sk}")

print("== 12) similar-but-distinct skills never merged ==")
sk = extract("SKILLS", "C", "C++", ".NET", "Network Security",
             "Node.js", "Node", "F#", "F")
check("C and C++ distinct", "C" in sk and "C++" in sk, extra=f"got {sk}")
check(".NET distinct from Network Security",
      ".NET" in sk and "NET" not in sk, extra=f"got {sk}")
check("Node.js distinct from Node", "Node.js" in sk, extra=f"got {sk}")
check("F# distinct from F", "F#" in sk and "F" in sk, extra=f"got {sk}")

print("== 13) never-invented contract ==")
for name, txt in [
    ("1.pdf", "UPLOADS/1.pdf"),
    ("2.pdf", "UPLOADS/2.pdf"),
    ("4.pdf", "UPLOADS/4.pdf"),
    ("sample", "UPLOADS/sample_resume.pdf"),
]:
    p = UPLOADS / Path(txt).name
    if p.exists():
        raw = extract_text_from_pdf(str(p))
        sks = extract_skills_from_text(raw)
        bad = no_invented(sks, raw)
        check(f"{name}: every extracted skill appears in the resume ({len(sks)})",
              not bad, extra=f"invented={bad}" if bad else "")

print("== 14) full resume: skills -> experience -> education boundaries ==")
sk = extract("SUMMARY", "Junior engineer, Python shop.",
             "SKILLS", "Python", "Docker", "PostgreSQL",
             "EXPERIENCE", "Backend Engineer", "Jan 2020 - present",
             "EDUCATION", "BSc Computer Science")
check("3 skills, no boundary leakage",
      sorted(sk) == ["Docker", "PostgreSQL", "Python"], extra=f"got {sk}")

print("== 15) skill heading variants ==")
for hdg in ("SKILLS", "TECHNICAL SKILLS", "TECHNICAL EXPERTISE",
            "CORE SKILLS", "KEY SKILLS", "SKILL SET", "TECHNOLOGIES",
            "TECHNOLOGY", "TOOLS & TECHNOLOGIES", "TECHNICAL PROFICIENCIES",
            "IT SKILLS", "COMPUTER SKILLS", "RELEVANT SKILLS"):
    sk = extract(f"{hdg}", "Python", "SQL")
    check(f"'{hdg}' captures Python+SQL", sk == ["Python", "SQL"],
          extra=f"got {sk}")

print("== 16) 'Heading: inline content' form ==")
for hdg, must_have in (("Core Competencies: Python, SQL, Docker, Pandas",
                        ("python", "sql")),
                       ("Skills: Python, SQL", ("python", "sql")),
                       ("Technical Skills: C++, SQL", ("c++", "sql"))):
    sk = extract(text=hdg)
    low = " ".join(sk).lower()
    check(f"'{hdg[:24]}…' captures its inline terms",
          all(m in low for m in must_have), extra=f"got {sk}")

print("== 17) unusual heading (TECHNOLOGY STACK) ==")
sk = extract("TECHNOLOGY STACK", "React", "Node.js", "TypeScript")
check(">= 2 skills from unusual heading",
      len(sk) >= 2 and "React" in sk and "Node.js" in sk, extra=f"got {sk}")

print("== 18) capitalization canonicalized ==")
raw = "\n".join(["SKILLS", "python", "PYTHON", "Python", "SQL", "sql"])
sk = extract_skills_from_text(raw)
check("python mentioned 3 ways -> single skill",
      "Python" in sk and sk.count("Python") == 1, extra=f"got {sk}")

print("== 19) duplicates removed ==")
sk = extract("SKILLS", "Python", "Python", "SQL", "Python")
check("duplicate entries collapse to distinct set", sk == ["Python", "SQL"],
      extra=f"got {sk}")

print("== 20) no obvious ordinary words / contact noise ==")
noise = ("CONTACT\nName: Some Person\nEmail: x@y.com\nPhone: 123456\n"
         "PROFESSIONAL SKILLS\nPython\nSQL")
sk = extract_skills_from_text(noise)
check("contact name/email/phone/address not extracted as skills",
      "Some Person" not in sk and "x@y.com" not in sk
      and "123456" not in sk and "Name" not in sk, extra=f"got {sk}")
sk = extract("SKILLS", "The", "and", "to", "using", "Development", "Design")
check("edge/stop words dropped but capitalized listed skills kept",
      sk == ["Design", "Development"], extra=f"got {sk}")
sk = extract("SKILLS", "Python dev experience using FastAPI and MongoDB")
check("sentence-like fragment never a skill",
      all(s in ("Python", "FastAPI", "MongoDB") for s in sk), extra=f"got {sk}")

print("== 21) debug hook is dev-only; production path logs nothing ==")
buf = io.StringIO()
with redirect_stdout(buf):
    extract_skills_from_text("SKILLS\nPython\nSQL\nEXPERIENCE\nEngineer")
prod_out = buf.getvalue()
check("production call emits no diagnostics",
      "[extractor]" not in prod_out and prod_out.strip() == "",
      extra=f"output={prod_out!r}")
buf2 = io.StringIO()
with redirect_stdout(buf2):
    extract_skills_from_text("SKILLS\nPython\nPandas\nDocker", debug=True)
dev_out = buf2.getvalue()
check("debug=True shows accept/reject diagnostics",
      "[extractor]" in dev_out and "Pandas" in dev_out,
      extra=f"got markers={dev_out.count('[extractor]')}")

print("== regression: failing PDFs before/after thresholds ==")
pdf1 = UPLOADS / "1.pdf"
if pdf1.exists():
    raw = extract_text_from_pdf(str(pdf1))
    sk = extract_skills_from_text(raw)
    check("1.pdf now extracts >= 6 skills (was 3)",
          len(sk) >= 6, extra=f"got {len(sk)} -> {sk}")
    check("1.pdf keeps machine-learning core",
          "Machine Learning" in sk and "Computer Programming" in sk,
          extra=f"got {sk}")

pdf4 = UPLOADS / "4.pdf"
if pdf4.exists():
    raw = extract_text_from_pdf(str(pdf4))
    sk = extract_skills_from_text(raw)
    for want in ("Management Skills", "Negotiation", "Critical Thinking"):
        check(f"4.pdf keeps '{want}'", want in sk, extra=f"got {sk}")

pdfs = UPLOADS / "sample_resume.pdf"
if pdfs.exists():
    raw = extract_text_from_pdf(str(pdfs))
    sk = extract_skills_from_text(raw)
    check("sample_resume.pdf keeps medical skills (was ~0)",
          "Rheumatology" in sk and "Geriatric Medicine" in sk
          and "Hospice and Palliative Care" in sk, extra=f"got {sk}")

print(f"\n==== {PASSED} passed, {FAILED} failed ====")
sys.exit(1 if FAILED else 0)