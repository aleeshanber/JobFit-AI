from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"
CSV_PATH = DATA_DIR / "cleaned_jobs.csv"

# Cached artifacts built once at app startup / first use.
_JOBS_DF = None
_VECTORIZER = None
_JOB_VECTORS = None


class DatasetNotFoundError(RuntimeError):
    """Raised when the cleaned jobs dataset cannot be loaded."""


def _build_cache():
    """Load + vectorize the jobs dataset once and cache the results."""
    global _JOBS_DF, _VECTORIZER, _JOB_VECTORS

    if not CSV_PATH.exists():
        raise DatasetNotFoundError(f"Dataset not found: {CSV_PATH}")

    jobs_df = pd.read_csv(CSV_PATH)

    for col in ["job_title", "required_skills", "experience_required"]:
        jobs_df[col] = jobs_df[col].fillna("")

    jobs_df["combined_text"] = (
        jobs_df["job_title"]
        + " "
        + jobs_df["required_skills"]
        + " "
        + jobs_df["experience_required"]
    )

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(jobs_df["combined_text"].tolist())

    _JOBS_DF = jobs_df
    _VECTORIZER = vectorizer
    _JOB_VECTORS = matrix

    return _JOBS_DF, _VECTORIZER, _JOB_VECTORS


def _ensure_cache():
    if _JOBS_DF is None or _VECTORIZER is None or _JOB_VECTORS is None:
        return _build_cache()
    return _JOBS_DF, _VECTORIZER, _JOB_VECTORS


def match_resume_with_jobs(user_skills, top_n=10):
    """Match user skills against the cached jobs dataset.

    user_skills : list[str] – extracted skill terms
    top_n       : int       – number of top matches to return
    """
    jobs_df, vectorizer, job_vectors = _ensure_cache()

    user_skills_str = " ".join(user_skills)
    user_vector = vectorizer.transform([user_skills_str])

    similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

    jobs_df["match_score"] = (similarity_scores * 100).round(2)

    ranked = jobs_df.sort_values(by="match_score", ascending=False)

    # Collect the top scoring DISTINCT job titles (a skillset may repeat
    # under multiple titles, so scan down the ranked list until we have
    # top_n unique titles).
    seen_titles = set()
    keep_idx = []
    for idx, title in ranked["job_title"].items():
        if title not in seen_titles:
            seen_titles.add(title)
            keep_idx.append(idx)
        if len(keep_idx) == top_n:
            break

    top_matches = ranked.loc[keep_idx].reset_index(drop=True)

    return top_matches[
        ["job_title", "required_skills", "experience_required", "match_score"]
    ]
