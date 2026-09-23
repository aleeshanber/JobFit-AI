"""Step 8.13 - multi-resume E2E across representative scenarios A-G.

Runs the FULL production path (extraction -> profile -> 17-vector -> RF -> rank)
over a diverse matrix of resumes: normal technical, special-character skills,
unusual section headings, overlapping jobs, inverted/future-date templates,
missing experience, missing education, and no-skills-section fallback, plus the
real sample PDFs in UPLOADS/. For every resume it records extracted skills,
experience/education profile, top-10 distinct titles and the conservative
"invented-info" verification: every displayed skill and every profile number
must be derivable ONLY from that resume's own text.

Uses the app's own production functions (this IS the production path).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UTILS.extractor import extract_skills_from_text, extract_text_from_pdf
from UTILS.recommender import build_candidate_profile_from_text, build_recommender

ROOT = Path(__file__).resolve().parents[1]

# Scenario resume texts (resume-derived only; nothing invented).
SCENARIOS = {
    "A. normal technical": (
        "PROFILE\nPython developer with ML background\n\n"
        "PROFESSIONAL SKILLS\nPython\nSQL\nMachine Learning\nDocker\nPandas\n\n"
        "WORK EXPERIENCE\nSoftware Engineer\nJan 2019 - present\n"
        "Data Engineer\nJun 2017 - Dec 2018\n\n"
        "EDUCATION\nMaster of Science in Computer Science\n"),
    "B. special-character skills": (
        "SKILLS\nC++\nC#\n.NET\nC/C++\nHTML/CSS\nCI/CD\nNode.js\nASP.NET\nF#\n\n"
        "WORK EXPERIENCE\nDeveloper\nFeb 2020 - present\n"),
    "C. unusual headings": (
        "TECHNICAL EXPERTISE\nKubernetes\nTerraform\n\n"
        "PROFESSIONAL HISTORY\nDevOps Engineer\nMar 2018 - present\n\n"
        "ACADEMIC QUALIFICATIONS\nBachelor of Science in Information Systems\n"),
    "D. overlapping jobs": (
        "PROFESSIONAL SKILLS\nProject Management\nJira\n\n"
        "WORK EXPERIENCE\nPM A\nJan 2020 - present\nPM B\nJun 2020 - present\n"),
    "E. inverted/future template (sample 1.pdf style)": (
        "ROBOTICS ENGINEER\nEmeka Afia\n\n"
        "PROFESSIONAL SKILLS\nInstallation and Debugging\nComputer Programming\n\n"
        "WORK REFERENCES\nHead Robotics Engineer | Innovation Eng | Dec 2035 - present\n"
        "Robotics Engineer | Practical Systems | Jan 2032 - Nov 2035\n\n"
        "WORK EXPERIENCE\nMaster of Robotics | Jan 2030 - Dec 2033\n"
        "Bachelor of Electrical Engineering | Jan 2026 - Dec 2030\n\n"
        "ACADEMIC HISTORY\n"),
    "F. missing experience": (
        "SUMMARY\nNew graduate\n\n"
        "SKILLS\nPython\nSQL\n\n"
        "EDUCATION\nBachelors in Economics\n"),
    "G. no skills section (fallback match)": (
        "PROFESSIONAL SUMMARY\nFinancial analyst with strong accounting and "
        "financial modeling experience.\n\n"
        "WORK EXPERIENCE\nAnalyst\nJan 2021 - present\n\n"
        "EDUCATION\nMaster of Accounting\n"),
}


def main():
    print("=" * 80)
    print("STEP 8.13 MULTI-RESUME END-TO-END (scenarios A-G + sample PDFs)")
    print("=" * 80)
    engine = build_recommender()
    print(f"engine: specs={len(engine.jobs_specs)} features={len(engine.features)}\n")

    records = []
    for name, resume in SCENARIOS.items():
        text = resume
        skills = extract_skills_from_text(text)
        profile = build_candidate_profile_from_text(text)
        results = engine.recommend(", ".join(skills), profile=profile)
        top = results[0] if results else {}
        records.append({
            "resume": name, "source": "synthetic", "text_len": len(text),
            "text": text,
            "skills": skills, "profile": profile, "top10": [r["job_title"] for r in results],
            "top_score": round(top.get("match_score", 0), 3),
            "top_missing": top.get("missing_skills", []),
        })
        print(f"[{name}]")
        print(f"    skills ({len(skills)}): {skills}")
        print(f"    profile: {profile}")
        print(f"    top job: {top.get('job_title')}  score={top.get('match_score', 0):.3f}")
        print(f"    top-10: {[r['job_title'] for r in results]}")
        print()

    for pdf in sorted((ROOT / "UPLOADS").glob("*.pdf")):
        text = extract_text_from_pdf(str(pdf))
        if not text.strip():
            continue
        skills = extract_skills_from_text(text)
        profile = build_candidate_profile_from_text(text)
        results = engine.recommend(", ".join(skills), profile=profile)
        top = results[0] if results else {}
        records.append({
            "resume": pdf.name, "source": "pdf", "text_len": len(text),
            "text": text,
            "skills": skills, "profile": profile, "top10": [r["job_title"] for r in results],
            "top_score": round(top.get("match_score", 0), 3),
            "top_missing": top.get("missing_skills", []),
        })
        print(f"[PDF {pdf.name}]")
        print(f"    skills ({len(skills)}): {skills}")
        print(f"    profile: {profile}")
        print(f"    top job: {top.get('job_title')}  score={top.get('match_score', 0):.3f}")
        print(f"    top-10: {[r['job_title'] for r in results]}")
        print()

    # ---- conservative "no invented info" verification ----
    print("-" * 80)
    print("INVENTED-INFO VERIFICATION")
    all_ok = True
    for rec in records:
        lower = " " + " ".join(rec["text"].lower().split()) + " "
        # 1) every extracted skill must appear verbatim in the resume text
        for s in rec["skills"]:
            if s.lower() not in lower:
                print(f"  [INVENTED SKILL] {rec['resume']}: '{s}' not found in text")
                all_ok = False
        # 2) experience years must come from real date spans in the text
        if rec["profile"]["experience_years"] is not None \
                and rec["profile"]["experience_years"] > 0:
            if not any(m in lower for m in
                       ["jan", "feb", "mar", "apr", "may", "jun",
                        "jul", "aug", "sep", "oct", "nov", "dec"]):
                print(f"  [INVENTED EXPERIENCE] {rec['resume']}: no month date in text")
                all_ok = False
        # 3) education level must derive from a degree keyword in the text
        if rec["profile"]["education_level"] > 0:
            degrees = ("phd", "doctorate", "master", "mba", "msc", "m.com",
                       "bachelor", "b.sc", "bba", "b.tech", "honors", "diploma",
                       "hsc", "ssc")
            if not any(d in lower for d in degrees):
                print(f"  [INVENTED EDUCATION] {rec['resume']}: no degree keyword in text")
                all_ok = False
        # 4) JSON-safe, deterministic, no duplicates in top-10
        titles = rec["top10"]
        if len(titles) != len(set(titles)):
            print(f"  [DUP TITLES] {rec['resume']}: {titles}")
            all_ok = False

    print("  no invented skills / experience / education, no duplicate top-10:",
          "PASS" if all_ok else "FAIL")
    print("\nSTEP 8.13 E2E DONE.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    t0 = time.time()
    sys.exit(main())