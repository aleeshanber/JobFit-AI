"""Step 7.13 - end-to-end production flow on a real sample PDF.

PDF -> extracted text -> extracted skills -> candidate profile -> 17-feature
vectors -> trained RF prediction -> ranking -> top-10 distinct titles ->
required/missing skills -> Flask/API response. Prints a concise diagnostic
(skills, jobs scored, top-10 titles, model scores, missing skills). The app's
own functions are used so this IS the production path.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UTILS.extractor import extract_skills_from_text, extract_text_from_pdf
from UTILS.recommender import RICH_MATCH_FEATURE_NAMES, build_candidate_profile_from_text, build_recommender

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "UPLOADS" / "1.pdf"


def main():
    print("=" * 78)
    print("STEP 7.13 END-TO-END (sample PDF: UPLOADS/1.pdf)")
    print("=" * 78)

    engine = build_recommender()
    print(f"engine: specs={len(engine.jobs_specs)} "
          f"features={len(engine.features)} "
          f"tfidf_docs={sum(1 for s in engine.jobs_specs if s['required_skills'].strip())}")

    # 1. PDF -> text
    text = extract_text_from_pdf(str(PDF))
    print(f"[1] extracted text: {len(text)} chars")

    # 2. text -> skills
    skills = extract_skills_from_text(text)
    print(f"[2] extracted skills ({len(skills)}): {skills}")
    if not skills:
        print("    E2E ABORTED: no skills extracted")
        return 1

    # 3. candidate profile from the resume text only
    profile = build_candidate_profile_from_text(text)
    print(f"[3] candidate profile (resume-only): {profile}")

    # 4-6. 17-feature vectors -> RF prediction -> ranking
    candidate_skills_text = ", ".join(skills)
    results = engine.recommend(candidate_skills_text, profile=profile)
    print(f"[4] jobs scored: {len(engine.jobs_specs)} (all job specs)")
    print(f"[5-6] top-{len(results)} DISTINCT titles by model prediction:")

    for i, rec in enumerate(results, 1):
        missing = ", ".join(rec["missing_skills"]) if rec["missing_skills"] else "None specified"
        print(f"    {i:2d}. {rec['job_title']:<55} score={rec['match_score']:.3f} "
              f"({rec['match_score'] * 100:.1f}%)")

    print("[7] required + missing skills for the top job:")
    top = results[0]
    print(f"    required: {top['required_skills']}")
    print(f"    missing : {top['missing_skills']}")

    # feature-vector demonstration (one pair)
    vec = engine.features_for(candidate_skills_text, engine.jobs_specs[0])
    assert len(vec) == 17 == len(RICH_MATCH_FEATURE_NAMES)
    print(f"[v] sample 17-vector for top job: length={len(vec)}; "
          f"exp={vec[8]:.2f} exp_gap={vec[9]:.2f} edu={vec[10]:.0f} "
          f"tfidf_sim={vec[15]:.3f}")

    # 8. Flask/API response path (production renderer)
    from APP.app import app as flask_app

    app = flask_app
    app.config["TESTING"] = True
    with app.test_client() as client:
        resp = client.get("/")
        assert resp.status_code == 200
        with open(PDF, "rb") as fh:
            post = client.post("/", data={"resume": (fh, "1.pdf")},
                               content_type="multipart/form-data")
        assert post.status_code == 200, post.status_code
        html = post.get_data(as_text=True)
        if '<div class="error-box">' in html:
            import re as _re

            _m = _re.search(r'<div class="error-box">(.*?)</div>', html, _re.S)
            raise AssertionError(f"E2E rendered an error: "
                                 f"{_m.group(1).strip() if _m else 'unknown'}")
        assert "Analyze Resume" in html and "Match Score" in html
        assert top["job_title"] in html
        print(f"[8] Flask production response OK (title present, no error box)")

    print("\nE2E PASSED - full production flow verified.")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    sys.exit(main())