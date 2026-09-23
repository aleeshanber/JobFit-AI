import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

# Shared preprocessing: symbol-preserving skill parsing, display casing and
# the canonical (case-insensitive) keys used for duplicate detection.
from UTILS.features import canonical_skill_key, parse_skills, skills_to_text

DATA_PATH = BASE_DIR.parent / "DATA" / "resume_data.csv"
OUT_PATH = BASE_DIR.parent / "DATA" / "cleaned_jobs.csv"

# ---------------------------------------------------------------------------
# Column selection (EXACT names only — no fuzzy/partial matching).
#
# The raw CSV has a UTF-8 BOM on a mid-file column name ("<BOM>job_position_name"),
# so names are compared after stripping a leading BOM. No candidate is ever
# matched on a partially-normalized substring; if an exact name is missing the
# process fails loudly instead of silently picking the wrong column.
# ---------------------------------------------------------------------------

# output field -> exact raw column name(s) it must be sourced from.
OUTPUT_COLUMNS = {
    "job_title": "job_position_name",              # job title (label)
    "required_skills": "skills_required",          # skills a JOB requires
    "responsibilities": "responsibilities.1",      # job responsibilities
    "experience_required": "experiencere_requirement",
    "education_required": "educationaL_requirements",
    "candidate_skills": "skills",                  # CANDIDATE/resume skills
    "matched_score": "matched_score",              # resume-job relevance 0..1
}


def _strip_bom(name):
    return str(name).lstrip("\ufeff") if name is not None else name


def _select_column(df, output_field):
    """Return the raw df column providing `output_field`, exact-match only.

    Raises KeyError (never a fuzzy fallback) when the expected column is absent.
    """
    expected = OUTPUT_COLUMNS[output_field]
    for col in df.columns:
        if _strip_bom(col) == expected:
            return col
    available = ["%r" % _strip_bom(c) for c in df.columns]
    raise KeyError(
        f"Exact source column {expected!r} not found for {output_field!r}. "
        f"Available columns: {available}"
    )


# ---------------------------------------------------------------------------
# Skill parsing.
#
# Delegated to the shared utility (UTILS/features.py) so data cleaning,
# resume extraction and training/inference all use the identical rules:
#   - delimiters are structural list separators only (comma/newline/semicolon/
#     pipe/bullets); "/" ":" "-" "." never split technical names,
#   - meaningful symbols (+, #, ., /, -) are preserved,
#   - the same word cap and junk gate apply everywhere,
#   - casing is normalized via canonical_skill_key(); display via
#     format_skill_name() (both shape-preserving).
#   parse_skills(value)  -> list[str]  unique, display-cased skills
#   skills_to_text(value)-> str        comma-joined, recoverable list
# ---------------------------------------------------------------------------


def main():
    df = pd.read_csv(DATA_PATH)

    print("=== Selecting source columns (exact name match only) ===")
    sources = {}
    for out_field in OUTPUT_COLUMNS:
        source = _select_column(df, out_field)
        sources[out_field] = source
        print(f"  {out_field:<20} <- {source!r}")

    # Reject rows without a job title (no label to learn against).
    title_raw = sources["job_title"]
    before_total = len(df)
    titled = df[df[title_raw].notna()].copy()
    titled[title_raw] = titled[title_raw].astype(str).str.strip()
    titled = titled[titled[title_raw] != ""]
    print(f"\nRows: {before_total} raw -> {len(titled)} with a non-empty job title "
          f"({before_total - len(titled)} removed: missing title)")

    cleaned = pd.DataFrame(
        {
            "job_title": titled[title_raw].str.strip(),
            "required_skills": titled[sources["required_skills"]].apply(skills_to_text),
            "responsibilities": titled[sources["responsibilities"]].fillna(""),
            "experience_required": titled[sources["experience_required"]].fillna(""),
            "education_required": titled[sources["education_required"]].fillna(""),
            "candidate_skills": titled[sources["candidate_skills"]].apply(skills_to_text),
            "matched_score": pd.to_numeric(
                titled[sources["matched_score"]], errors="coerce"
            ),
        }
    )

    cleaned["responsibilities"] = cleaned["responsibilities"].astype(str)
    cleaned["experience_required"] = cleaned["experience_required"].astype(str)
    cleaned["education_required"] = cleaned["education_required"].astype(str)
    # Row-aligned copy (kept in raw row order) for printing faithful examples.
    aligned = cleaned.copy()

    # Deduplication: remove ONLY rows that are fully identical across every
    # meaningful field (job title, job requirements, candidate skills and the
    # matched relevance score). Skill strings are compared through the shared
    # canonical (lowercase, shape-preserving) key, so "Python" vs "python" is a
    # duplicate while rows differing by any skill, or belonging to different
    # job titles, are always preserved.
    cleaned["_key_req"] = cleaned["required_skills"].apply(canonical_skill_key)
    cleaned["_key_cand"] = cleaned["candidate_skills"].apply(canonical_skill_key)
    dup_subset = [
        "job_title",
        "_key_req",
        "responsibilities",
        "experience_required",
        "education_required",
        "_key_cand",
        "matched_score",
    ]
    before_dedup = len(cleaned)
    n_dups = int(cleaned.duplicated(subset=dup_subset, keep=False).sum())
    cleaned = cleaned.drop_duplicates(subset=dup_subset, keep="first").drop(
        columns=["_key_req", "_key_cand"]
    )

    cleaned = cleaned.sort_values(
        ["job_title", "candidate_skills"], kind="stable"
    ).reset_index(drop=True)

    removed_dedup = before_dedup - len(cleaned)

    cleaned.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"\n=== Cleaning summary ===")
    print(f"Rows after dedup : {len(cleaned)} (removed {removed_dedup} exact full-record "
          f"duplicates; {n_dups} rows were part of a duplicate group)")
    print(f"Unique job titles: {cleaned['job_title'].nunique()} (before {titled[title_raw].nunique()})")
    print(f"matched_score preserved (non-null): {int(cleaned['matched_score'].notna().sum())} "
          f"| range {cleaned['matched_score'].min():.2f} - {cleaned['matched_score'].max():.2f}")
    print(f"candidate_skills non-empty rows : {int((cleaned['candidate_skills'] != '').sum())}")
    print(f"required_skills (JOB) non-empty : {int((cleaned['required_skills'] != '').sum())} "
          f"(5 job titles carry no skills_required in the source data)")
    print(f"Distinct (job_title, required_skills) pairs kept: "
          f"{cleaned[['job_title', 'required_skills']].drop_duplicates().shape[0]}")

    print("\n=== Before/after examples (same raw rows, row-aligned) ===")
    for i in range(min(3, len(aligned))):
        raw_skills = str(titled[sources["required_skills"]].iloc[i] or "")[:80]
        cand = str(titled[sources["candidate_skills"]].iloc[i] or "")[:80]
        print(f"  RAW  title={aligned['job_title'].iloc[i]!r} job_skills={raw_skills!r}")
        print(f"       candidate={cand!r}")
        print(f"  CLEAN job_req={aligned['required_skills'].iloc[i]!r}")
        print(f"       cand   ={aligned['candidate_skills'].iloc[i]!r}")

    print(f"\nSUCCESS: saved to {OUT_PATH}")


if __name__ == "__main__":
    main()