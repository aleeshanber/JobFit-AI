import re
import sys
import uuid
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from UTILS.extractor import (
    extract_skills_from_text,
    extract_text_from_pdf,
)
from UTILS.features import format_skill_name
from UTILS.recommender import (
    RecommendationEngine,
    RecommendationEngineError,
    build_candidate_profile_from_text,
    build_recommender,
)

NOT_SPECIFIED = "Not Specified in Job Posting"


def _format_title_case(skill_text):
    """Return a comma-separated skill string with proper display casing.

    Uses format_skill_name for each token so technical terms like C++, .NET,
    Node.js, ASP.NET are rendered correctly.  Empty/None input returns
    NOT_SPECIFIED.
    """
    if not skill_text or not str(skill_text).strip():
        return NOT_SPECIFIED
    tokens = re.split(r"[,;|\n]+", str(skill_text))
    cleaned = [format_skill_name(t) for t in tokens if t.strip()]
    return ", ".join(cleaned) if cleaned else NOT_SPECIFIED


def _compute_missing_skills(required_skills_text, candidate_skills):
    """Skills required by the job but absent from the candidate's resume.

    Matching uses canonical (lowercased, whitespace-collapsed) keys for
    exact comparison, then falls back to content-word containment so that
    fragmented candidate skills (e.g. ``OLT`` + ``ONU`` on separate lines)
    still satisfy a multi-word required skill (``OLT and ONU``).

    Returns a list of display-cased skill names.  Empty list when the job
    has no requirements or every required skill is covered.
    """
    if not required_skills_text or not str(required_skills_text).strip():
        return []

    req_tokens = re.split(r"[,;|\n]+", str(required_skills_text))
    req_skills = {t.strip().lower() for t in req_tokens if t.strip()}

    cand_keys = {s.lower() for s in candidate_skills}

    # Fast path: everything matched by exact canonical key.
    still_missing = req_skills - cand_keys
    if not still_missing:
        return []

    # Slow path: build a set of all content words from candidate skills
    # (connector words like 'and', 'or', 'the' are excluded).
    _CONNECTORS = frozenset({
        "and", "or", "the", "of", "in", "on", "at", "for", "to", "with",
        "a", "an", "is", "are", "was", "were", "be", "been", "by", "from",
        "into", "over", "between",
    })
    cand_words = set()
    for s in candidate_skills:
        for w in s.lower().split():
            if w not in _CONNECTORS:
                cand_words.add(w)

    def _covered(req_lower):
        req_words = req_lower.split()
        content = [w for w in req_words if w not in _CONNECTORS]
        if not content:
            return True
        return all(w in cand_words for w in content)

    missing = sorted(r for r in still_missing if not _covered(r))
    return [format_skill_name(s) for s in missing]


def _job_search_links(skills):
    """Return a list of (label, url) tuples for real job search links.

    Builds the query from the candidate's extracted skills.  Falls back to a
    generic query when the skills list is empty or None.
    """
    if skills:
        query = ", ".join(skills) + " jobs"
    else:
        query = "jobs"
    q = urllib.parse.quote_plus(query)
    return [
        ("LinkedIn Jobs", f"https://www.linkedin.com/jobs/search/?keywords={q}"),
        ("Indeed",        f"https://www.indeed.com/jobs?q={q}"),
        ("Google Search", f"https://www.google.com/search?q={q}"),
    ]


app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "TEMPLATES"),
    static_folder=str(PROJECT_ROOT / "STATIC"),
)

UPLOAD_FOLDER = PROJECT_ROOT / "UPLOADS"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload

# The trained model + shared feature pipeline are loaded ONCE at startup and
# reused for every uploaded resume. A failed load is captured here and shown as
# a clean error instead of a stack trace (Step 7.2 / 7.10).
_ENGINE = None
_ENGINE_ERROR = None
_NO_MODEL_MESSAGE = (
    "The recommendation engine is not available right now. Please ensure "
    "MODELS/match_model.pkl + MODELS/match_model_metadata.json and "
    "DATA/cleaned_jobs.csv exist, then restart."
)


def _ensure_engine():
    """Return the singleton engine, or show the friendly startup error."""
    global _ENGINE, _ENGINE_ERROR
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_ERROR is None:
        try:
            _ENGINE = build_recommender()
        except RecommendationEngineError as e:
            _ENGINE_ERROR = str(e)
            print(f"Engine startup failure (hidden from UI): {e}")
        except Exception as e:  # defensive: never leak a traceback to the UI
            _ENGINE_ERROR = f"Unexpected engine startup failure: {type(e).__name__}"
            print(f"Engine startup failure (hidden from UI): {e}")
    return _ENGINE


def clean_upload_filepath(filename):
    """Return a safe, unique, absolute filepath inside UPLOAD_FOLDER."""
    safe_name = secure_filename(filename) or "resume"
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    filepath = (UPLOAD_FOLDER / unique_name).resolve()
    if not filepath.is_relative_to(UPLOAD_FOLDER.resolve()):
        raise ValueError("Invalid upload path")
    return filepath


def _reject_payload(file):
    """Return an error message and True when the upload is unusable."""
    if "resume" not in request.files:
        return "Please select a PDF file first!", True
    if not file or file.filename.strip() == "":
        return "Please select a PDF file first!", True
    if not file.filename.lower().endswith(".pdf"):
        return "Only PDF format is supported!", True
    return None, False


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("resume")
        message, bad = _reject_payload(file)
        if bad:
            return render_template("index.html", error=message)

        filepath = None
        try:
            filepath = clean_upload_filepath(file.filename)
            file.save(str(filepath))

            engine = _ensure_engine()
            if engine is None:
                return render_template("index.html", error=_NO_MODEL_MESSAGE)

            extracted_text = extract_text_from_pdf(str(filepath))
            if not extracted_text or not extracted_text.strip():
                return render_template(
                    "index.html",
                    error="Could not read text from this PDF. Please check file format.",
                )

            skills = extract_skills_from_text(extracted_text)
            if not skills:
                return render_template(
                    "index.html",
                    error=(
                        "No matching skills were detected in this resume. "
                        "Please upload a resume with a clear Skills section."
                    ),
                )

            print(f"--> Extracted Raw PDF Text Length: {len(extracted_text)}")
            print(f"--> Extracted Skills Count: {len(skills)}")

            # Candidate-side information comes ONLY from the uploaded resume
            # (skill text + conservative experience/education extraction).
            candidate_skills_text = ", ".join(skills)
            profile = build_candidate_profile_from_text(extracted_text)
            print(f"--> Candidate profile: {profile}")

            results = engine.recommend(candidate_skills_text, profile=profile)
            tables = []
            for rec in results:
                raw_required = rec["required_skills"]
                formatted_required = _format_title_case(raw_required)
                missing = _compute_missing_skills(raw_required, skills)
                tables.append({
                    "job_title": rec["job_title"],
                    "required_skills": formatted_required,
                    "experience_required": rec["experience_required"],
                    "match_score": rec["match_score"],
                    "missing_skills": missing,
                    "search_links": _job_search_links(skills),
                })

            if not tables:
                return render_template(
                    "index.html", error=(
                        "No recommendations could be generated for this resume."
                    )
                )

            return render_template("index.html", skills=skills, tables=tables)

        except Exception as e:
            print(f"Error processing PDF: {e}")
            message = "File processing error. Please try a different PDF."
            return render_template("index.html", error=message)

        finally:
            if filepath and filepath.exists():
                try:
                    filepath.unlink()
                except Exception as e:
                    print(f"Cleanup error removing {filepath}: {e}")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)