"""UTILS/test_features.py — validation for the shared preprocessing rules
(project Step 2). Run:  python UTILS/test_features.py

Covers the special-character technical skills, delimiter integrity, casing,
whitespace/BOM handling, single-char skills, word-limit behavior, and the
shared job-document builder used by training and inference.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UTILS.features import (
    canonical_skill_key,
    clean_skill_item,
    format_skill_name,
    normalize_text,
    parse_skills,
    split_skill_items,
    build_job_document,
)

PASSED = 0
FAILED = 0


def check(label, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}")


def section(title):
    print(f"\n== {title} ==")


section("Special-character skills survive every layer")
for name in [
    "C++", "C#", ".NET", "ASP.NET", "C/C++", "CI/CD", "HTML/CSS",
    "Node.js", "F#", "Python", "SQL", "Machine Learning", "Deep Learning",
    "Natural Language Processing", "Computer Vision",
    "JavaScript", "TypeScript", "Scikit-learn",
]:
    check(f"format_skill_name preserves {name!r}", format_skill_name(name) == name)
    check(f"canonical key stable for {name!r}", canonical_skill_key(name).lower() != "")
    check(f"parse_skills keeps {name!r}", parse_skills(name) == [name])

section("C++ must not become C, C# must not become C")
check("canonical('C++') != canonical('C')", canonical_skill_key("C++") != canonical_skill_key("C"))
check("canonical('C#') != canonical('C')", canonical_skill_key("C#") != canonical_skill_key("C"))
check("canonical('.NET') != canonical('NET')", canonical_skill_key(".NET") != canonical_skill_key("NET"))
check("canonical('ASP.NET') != canonical('ASPNET')", canonical_skill_key("ASP.NET") != canonical_skill_key("ASPNET"))
check("canonical('CI/CD') != canonical('CICD')", canonical_skill_key("CI/CD") != canonical_skill_key("CICD"))
check("canonical('HTML/CSS') != canonical('HTMLCSS')", canonical_skill_key("HTML/CSS") != canonical_skill_key("HTMLCSS"))

section("Comma-separated skills stay separate & recoverable")
four = parse_skills("Python, SQL, C++, C#")
check("'Python, SQL, C++, C#' -> 4 skills", four == ["Python", "SQL", "C++", "C#"])
messed = parse_skills("Python,   SQL ,C++  ,  C#")
check("messy spacing still recovers 4 skills", messed == ["Python", "SQL", "C++", "C#"])

section("Delimiter integrity (no / : - . splitting of names)")
check("C/C++ is ONE skill", parse_skills("C/C++") == ["C/C++"])
check("HTML/CSS is ONE skill", parse_skills("HTML/CSS") == ["HTML/CSS"])
check("CI/CD is ONE skill", parse_skills("CI/CD") == ["CI/CD"])
check("'.NET' is ONE skill (not 'NET')", parse_skills(".NET") == [".NET"])
check("split_skill_items does not split 'C/C++'", split_skill_items("C/C++") == ["C/C++"])
check("hyphenated name not split", split_skill_items("K-Nearest Neighbors") == ["K-Nearest Neighbors"])
check("newline list splits", split_skill_items("Python\nSQL\nJava") == ["Python", "SQL", "Java"])

section("Whitespace / BOM normalization")
check("normalize_text trims and collapses", normalize_text("  Python   ") == "Python")
check("normalize_text collapses 'Python,   SQL'", normalize_text("Python,   SQL") == "Python, SQL")
check("BOM stripped", normalize_text("\ufeffPython") == "Python")
check("empty stays empty", normalize_text("   ") == "")

section("Case handling (case-insensitive dedupe, readable output)")
check("parse_skills dedupes Python/python/PYTHON", parse_skills("Python, python, PYTHON") == ["Python"])
check("format_skill_name('python') readable", format_skill_name("python") == "Python")
check("format_skill_name('c++') readable", format_skill_name("c++") == "C++")
check("format_skill_name('node.js') readable", format_skill_name("node.js") == "Node.js")
check("connector 'or' stays lowercase ('R or Java')",
      format_skill_name("R or Java") == "R or Java")
check("no synonym: 'ML' not expanded", format_skill_name("ML") == "ML")
check("no synonym: 'NLP' not expanded", format_skill_name("NLP") == "NLP")

section("Single-character technical skills are retained")
check("'R' survives parse_skills", parse_skills("R") == ["R"])
check("'C' survives parse_skills", parse_skills("C") == ["C"])
check("'R' survives clean_skill_item", clean_skill_item("R") == "R")
check("'C' survives clean_skill_item", clean_skill_item("C") == "C")

section("Word/token limits (legitimate multi-word skills preserved)")
for phrase in [
    "Machine Learning", "Deep Learning", "Natural Language Processing",
    "Computer Vision", "Data Science", "Microsoft SQL Server",
    "Amazon Web Services", "Predictive Analytics",
]:
    check(f"parse_skills keeps {phrase!r}", parse_skills(phrase) == [phrase])
check("7-word sentence fragment is dropped",
      parse_skills("Design and implement scalable cloud systems with many teams") == [])
check("oversized word count dropped consistently with cleaner+extractor",
      clean_skill_item("a b c d e f g h i j k l") is None)

section("Training/inference shared job document")
doc = build_job_document("Data Engineer", "Python, SQL", "Build pipelines", "3 years", "BS")
check("job doc contains title", "Data Engineer" in doc)
check("job doc doubles required skills", doc.count("Python, SQL") == 2)
check("job doc has no double spaces", "  " not in doc)
doc2 = build_job_document("  Data   Engineer ", " Python,  SQL ", " Build   pipelines ", "  3 years ", " BS ")
check("job doc is identical for raw vs messy inputs", doc == doc2)


print(f"\n==== {PASSED} passed, {FAILED} failed ====")
sys.exit(1 if FAILED else 0)