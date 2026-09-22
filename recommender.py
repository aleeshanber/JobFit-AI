"""UTILS/recommender.py — production recommendation engine (Step 7).

Loads the trained Step-6 matching model once, validates it against the artifact
metadata, rebuilds the SAME shared TF-IDF skill space used during training (from
the job-requirement corpus), and scores every job specification with the exact
shared 17-feature builder from UTILS/features.py.

Production rules enforced here:
  * The model blob + metadata are validated (kind = jobfit-match-model-step6,
    17 features in RICH_MATCH_FEATURE_NAMES order) or the engine refuses to
    start with a clear error — never silent mis-recommendations.
  * No fabricated scoring: the prediction IS the match score (0..1). No
    cosine, no hard-coded domain/title gates, no 54 + 90*similarity.
  * The candidate profile is built ONLY from the uploaded resume text using
    conservative extraction (section-scoped tenure dates + shared education
    keyword rules). Missing information -> the same safe-zero behavior used at
    training (None experience / 0 education flow into zeros).
  * Never reads matched_score / target / job_title as features. Job title is
    used only to display the job being scored.
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from UTILS.extractor import MAJOR_HEADINGS, _normalize_heading
from UTILS.features import (
    RICH_MATCH_FEATURE_NAMES,
    CURRENT_REFERENCE_DATE,
    build_skills_tfidf,
    candidate_job_match_features_rich,
    canonical_skill_key,
    education_level_from_text,
    format_skill_name,
    parse_resume_dates,
    parse_skill_items,
    validate_match_features_rich,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PKL = ROOT / "MODELS" / "match_model.pkl"
MODEL_META_JSON = ROOT / "MODELS" / "match_model_metadata.json"
JOBS_CSV = ROOT / "DATA" / "cleaned_jobs.csv"

KIND_EXPECTED = "jobfit-match-model-step6"
TOP_N = 10

# Experience / education heading groups used for section-scoped resume
# extraction (conservative: only content under these headings is used, and
# nothing is parsed when no such section exists).
EXPERIENCE_HEADINGS = frozenset({
    "experience", "work experience", "professional experience",
    "employment", "employment history", "work history", "career history",
    "professional history", "work record",
})
EDUCATION_HEADINGS = frozenset({
    "education", "academic background", "education and training",
    "qualifications", "professional qualifications", "academic qualifications",
    "academic history", "education history", "educational background",
    "educational history", "educational qualifications",
    "academic record", "academics", "academic education",
})

# Job columns read from the cleaned dataset. The dataset has a matched_score
# column, but it is NEVER read here — only requirement CONTENT is used.
_SPEC_COLUMNS = ("job_title", "required_skills", "experience_required",
                 "education_required")


class RecommendationEngineError(RuntimeError):
    """Raised when the production model/artifacts fail validation."""


class RecommendationEngine:
    """Loaded model + shared vectorizer + job catalog, ready to score."""

    def __init__(self, model, features, jobs_specs, tfidf):
        self.model = model
        self.features = list(features)
        self.jobs_specs = jobs_specs
        self.tfidf = tfidf

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @property
    def feature_names(self):
        return list(self.features)

    def validate_predictable(self, vector):
        """Assert a vector satisfies the Step-6 17-feature contract."""
        validate_match_features_rich(vector)
        return np.asarray(vector, dtype=float).reshape(1, -1)

    # ------------------------------------------------------------------
    # profile extraction (inference only, conservative, safe zeros)
    # ------------------------------------------------------------------

    def features_for(self, candidate_skills_text, spec):
        """Build the exact 17-feature vector for one (candidate, job) pair.

        profile is intentionally not passed here: candidate experience/education
        are the profile derived from the resume at the caller.
        """
        return candidate_job_match_features_rich(
            candidate_skills_text,
            spec["required_skills"],
            job_experience_required=spec["experience_required"],
            job_education_required=spec["education_required"],
            skills_tfidf=self.tfidf,
        )

    def score_row(self, candidate_skills_text, profile, spec):
        """Predict the 0..1 relevance score for one (candidate, job) row.

        The raw RF prediction is gated by sqrt(jaccard) so that zero skill
        overlap produces a zero score.  The model was trained on data where
        95 % of rows have zero overlap yet carry high target values, causing
        it to predict ~0.7 even for completely unrelated jobs.  The gate is
        principled: jaccard = |C ∩ R| / |C ∪ R| is already a feature of the
        model, and sqrt scaling gives a smooth, non-linear penalty that
        preserves the model's ordering for overlapping pairs.
        """
        vector = candidate_job_match_features_rich(
            candidate_skills_text,
            spec["required_skills"],
            profile=profile,
            job_experience_required=spec["experience_required"],
            job_education_required=spec["education_required"],
            skills_tfidf=self.tfidf,
        )
        validate_match_features_rich(vector)
        pred = float(np.asarray(self.model.predict(
            np.asarray(vector, dtype=float).reshape(1, -1))).reshape(-1)[0])
        # Gate: jaccard (index 1) is 0 when no skill overlap exists.
        # sqrt gives a smooth curve: 0->0, 0.25->0.5, 1.0->1.0
        jaccard = float(vector[1])
        pred *= jaccard ** 0.5
        return pred

    # ------------------------------------------------------------------
    # ranking
    # ------------------------------------------------------------------

    def recommend(self, candidate_skills_text, profile=None, top_n=TOP_N):
        """Score every job spec, rank by prediction, return top_n DISTINCT titles.

        Each result: job_title, match_score (raw 0..1 prediction),
        required_skills (raw job text), missing_skills (readable names),
        experience_required (display compat).
        """
        profile = profile or {}
        ranked = []
        for spec in self.jobs_specs:
            score = self.score_row(candidate_skills_text, profile, spec)
            ranked.append({
                "job_title": spec["job_title"],
                "match_score": score,
                "required_skills": spec["required_skills"],
                "experience_required": spec["experience_required"],
                "missing_skills": missing_skills(
                    spec["required_skills"], candidate_skills_text),
            })
        ranked.sort(key=lambda r: r["match_score"], reverse=True)

        seen = set()
        results = []
        for rec in ranked:
            if rec["job_title"] in seen:
                continue
            seen.add(rec["job_title"])
            results.append(rec)
            if len(results) >= top_n:
                break
        return results


def missing_skills(required_skills_text, candidate_skills_text):
    """Readable required-skill names missing from the candidate's skills.

    Comparison uses canonical_skill_key (case/symbol preserving: C++ != C,
    .NET != NET); the display returns format_skill_name casing of the REQUIRED
    skill, never invented names. Empty requirement -> [].

    After exact canonical matching, a content-word containment check catches
    fragmented candidate skills (e.g. ``OLT`` + ``ONU`` on separate resume
    lines still covering ``OLT and ONU``).
    """
    cand_keys = {canonical_skill_key(t)
                 for t in parse_skill_items(candidate_skills_text)}
    # Build a set of all content words from candidate skills
    _CONNECTORS = frozenset({
        "and", "or", "the", "of", "in", "on", "at", "for", "to", "with",
        "a", "an", "is", "are", "was", "were", "be", "been", "by", "from",
        "into", "over", "between",
    })
    cand_words = set()
    for t in parse_skill_items(candidate_skills_text):
        for w in canonical_skill_key(t).split():
            if w not in _CONNECTORS:
                cand_words.add(w)

    seen = set()
    out = []
    for token in parse_skill_items(required_skills_text):
        key = canonical_skill_key(token)
        if key in cand_keys or key in seen:
            continue
        # Content-word containment check for fragmented skills
        req_words = key.split()
        content = [w for w in req_words if w not in _CONNECTORS]
        if content and all(w in cand_words for w in content):
            continue
        seen.add(key)
        out.append(format_skill_name(token))
    return out


# ---------------------------------------------------------------------------
# Date / education extraction from free resume text
# ---------------------------------------------------------------------------


# All known region headings (major resume sections + the experience/education
# groups) used to bound each extraction region.
_ALL_REGION_HEADINGS = (
    set(MAJOR_HEADINGS) | set(EXPERIENCE_HEADINGS) | set(EDUCATION_HEADINGS))


def _region_text(text, start_headings, stop_headings):
    """Return the text under the first heading in start_headings.

    Stops at the next heading in stop_headings (all other major headings).
    Returns "" when no such section exists — the caller then falls back to
    safe unknown behavior, never to fabricated information.
    """
    lines = text.splitlines()
    captured = []
    inside = False
    for raw in lines:
        norm = _normalize_heading(raw)
        if inside:
            if norm in stop_headings:
                break
            captured.append(raw)
            continue
        if norm in start_headings:
            inside = True
    return "\n".join(captured)


_DATE_TOKEN_RE = re.compile(
    r"(?i)\b(current|present|ongoing|now)\b"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b"
    r"|\b(\d{1,2})/(20\d{2})\b")


def _date_tokens(region_text):
    """Ordered month-level date tokens (or Current-family words) in a region."""
    tokens = []
    for line in region_text.splitlines():
        for m in _DATE_TOKEN_RE.finditer(line):
            if m.group(1):
                tokens.append("Current")
            else:
                tokens.append(m.group(0))
    return tokens


def _month_index(value):
    """(year, month) -> absolute month count (0-based month)."""
    return value[0] * 12 + (value[1] - 1)


def _experience_years_union(region_text):
    """Overlap-safe tenure span (years) from an Experience section.

    Rules (Step 8.3):
      * Tokens are paired as consecutive (start, end) ranges and each range is
        clamped to CURRENT_REFERENCE_DATE (future end dates do not inflate).
      * Calendar overlaps between ranges are MERGED (a union), so simultaneous
        jobs never double-count the same calendar period.
      * Invalid/masked dates (None from the shared parser) are ignored.
      * Ranges that start after CURRENT_REFERENCE_DATE (fake/template future
        jobs) are dropped entirely.
      * Returns None (safe zero for the feature pipeline) when nothing valid.
    The shared date parser and reference anchor are reused unchanged.
    """
    tokens = _date_tokens(region_text)
    if not tokens:
        return None

    anchor = _month_index(CURRENT_REFERENCE_DATE)
    ranges = []
    for start_tok, end_tok in zip(tokens, tokens[1:]):
        parsed = parse_resume_dates(f"[{start_tok!r}, {end_tok!r}]")
        s = parsed[0] if len(parsed) > 0 else None
        e = parsed[1] if len(parsed) > 1 else None
        if s is None:
            continue
        if e is None:
            e = CURRENT_REFERENCE_DATE
        s_m, e_m = _month_index(s), min(_month_index(e), anchor)
        if s_m >= anchor or e_m <= s_m:  # fully future / invalid / empty
            continue
        ranges.append((s_m, e_m))

    if not ranges:
        return None

    # Union of overlapping calendar ranges.
    ranges.sort()
    merged = [ranges[0]]
    for s_m, e_m in ranges[1:]:
        last_s, last_e = merged[-1]
        if s_m <= last_e:
            merged[-1] = (last_s, max(last_e, e_m))
        else:
            merged.append((s_m, e_m))
    total_months = sum(e_m - s_m for s_m, e_m in merged)
    return total_months / 12.0


def build_candidate_profile_from_text(text):
    """Extract {'experience_years', 'education_level'} from resume text.

    Conservative + resume-faithful:
      * experience: union of date ranges inside an Experience/Employment
        section only, using the shared month parser + CURRENT_REFERENCE_DATE;
        None when there is no section or no valid range -> safe zero at
        feature time. Overlapping jobs never double-count the period.
      * education: shared education_level_from_text on the Education region
        only; whole-resume degree-keyword fallback; 0 (unknown) when nothing.
    Never uses training-data profiles and never invents values.
    """
    exp_region = _region_text(text, EXPERIENCE_HEADINGS,
                              _ALL_REGION_HEADINGS - EXPERIENCE_HEADINGS)
    exp = _experience_years_union(exp_region)

    edu_region = _region_text(text, EDUCATION_HEADINGS,
                              _ALL_REGION_HEADINGS - EDUCATION_HEADINGS)
    if edu_region.strip():
        edu = education_level_from_text(edu_region)
    else:
        # No recognisable education section: fall back to degree keywords
        # anywhere in the resume text (specific, low false-positive; still
        # resume-derived, never invented). 0 when nothing found.
        edu = education_level_from_text(text)

    return {"experience_years": exp, "education_level": int(edu)}


# ---------------------------------------------------------------------------
# Artifact loading + validation
# ---------------------------------------------------------------------------


def validate_artifact_dicts(blob, meta):
    """Raise RecommendationEngineError when the trained artifact is unusable."""
    if not isinstance(blob, dict):
        raise RecommendationEngineError(
            "Model file is not a valid jobfit artifact.")
    if blob.get("kind") != KIND_EXPECTED:
        raise RecommendationEngineError(
            f"Model kind mismatch: expected {KIND_EXPECTED!r}, "
            f"got {blob.get('kind')!r}. Refusing to produce incorrect "
            "recommendations.")
    meta_features = meta.get("features")
    if not isinstance(meta_features, list) or len(meta_features) != len(
            RICH_MATCH_FEATURE_NAMES):
        raise RecommendationEngineError(
            f"Model uses {len(meta_features) if isinstance(meta_features, list) else '?'} "
            f"features, expected {len(RICH_MATCH_FEATURE_NAMES)} (Step-6 rich model).")
    if list(meta_features) != list(RICH_MATCH_FEATURE_NAMES):
        raise RecommendationEngineError(
            "Feature names/order do not match RICH_MATCH_FEATURE_NAMES.")
    blob_features = blob.get("features")
    if not isinstance(blob_features, list) or list(blob_features) != list(
            RICH_MATCH_FEATURE_NAMES):
        raise RecommendationEngineError(
            "Model blob feature names do not match RICH_MATCH_FEATURE_NAMES.")
    if not hasattr(blob.get("model"), "predict"):
        raise RecommendationEngineError("Model artifact has no .predict method.")
    return True


def _job_specs(df):
    """One representative spec per title (first non-empty required-skills row)."""
    first_req = {}
    fallback = {}
    for _, r in df.iterrows():
        fallback.setdefault(r["job_title"], r)
        if r["job_title"] not in first_req and str(r["required_skills"]).strip():
            first_req[r["job_title"]] = r
    specs = []
    for title in df["job_title"].unique():
        row = first_req.get(title)
        if row is None:
            row = fallback[title]
        specs.append({col: ("" if pd.isna(row[col]) else str(row[col]))
                      for col in _SPEC_COLUMNS})
    return specs


def build_recommender(model_pkl=MODEL_PKL, meta_json=MODEL_META_JSON,
                      jobs_csv=JOBS_CSV):
    """Load and validate the Step-6 model + shared corpus, return the engine.

    Raises RecommendationEngineError for every unusable condition so the caller
    can surface a clean user-facing error instead of a stack trace.
    """
    if not Path(jobs_csv).exists():
        raise RecommendationEngineError(
            "Job dataset missing (DATA/cleaned_jobs.csv).")
    if not Path(model_pkl).exists():
        raise RecommendationEngineError(
            "Model artifact missing (MODELS/match_model.pkl).")
    if not Path(meta_json).exists():
        raise RecommendationEngineError(
            "Model metadata missing (MODELS/match_model_metadata.json).")

    df = pd.read_csv(jobs_csv, keep_default_na=False, na_values=[],
                     usecols=lambda c: c in _SPEC_COLUMNS)
    specs = _job_specs(df)
    corpus_docs = [s["required_skills"] for s in specs
                   if str(s["required_skills"]).strip()]
    tfidf = build_skills_tfidf(corpus_docs)   # same space as training

    with open(model_pkl, "rb") as fh:
        blob = pickle.load(fh)
    with open(meta_json, encoding="utf-8") as fh:
        meta = json.load(fh)

    validate_artifact_dicts(blob, meta)
    return RecommendationEngine(blob["model"], blob["features"], specs, tfidf)


# Guard against accidental import-time heavy work (engine is built explicitly).
if __name__ == "__main__":
    import sys

    try:
        engine = build_recommender()
        docs = sum(1 for s in engine.jobs_specs if s["required_skills"].strip())
        print(f"engine ready: specs={len(engine.jobs_specs)} "
              f"tfidf_docs={docs} features={len(engine.features)}")
        sys.exit(0)
    except RecommendationEngineError as e:
        print(f"ENGINE ERROR: {e}")
        sys.exit(1)