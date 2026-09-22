"""UTILS/features.py — single source of truth for shared text/skill preprocessing.

Every pipeline step (data cleaning, resume skill extraction, training,
inference/recommendation) must apply the SAME rules by importing from here:

  - whitespace / BOM handling        normalize_text()
  - skill list delimiters            SKILL_DELIMITER_RE / split_skill_items()
  - symbol-preserving token cleaning clean_skill_item()
  - word / junk limits               MAX_SKILL_WORDS, MAX_SKILL_CHARS, is_valid_skill()
  - canonical (case-insensitive) keys canonical_skill_key()
  - readable display casing          format_skill_name()
  - sentence-fragment gate           is_junk_fragment()
  - job document builder             build_job_document()

Design rules (project Step 2):
  * Delimiters are ONLY structural list separators: comma, newline, semicolon,
    pipe and bullet glyphs (plus the full-width comma).  Slashes, colons,
    hyphens and dots are NEVER list delimiters because they form technical
    names ("HTML/CSS", "CI/CD", "C/C++", ".NET", "Node.js", "Scikit-learn").
  * Meaningful symbols (+ # . / - etc.) are never stripped from a token.
    Only surrounding quote/whitespace/formatting noise is removed, and only a
    trailing period is treated as sentence-termination noise (".NET" stays
    ".NET", "Scikit-learn." becomes "Scikit-learn").
  * Casing is normalized for COMPARISON via canonical_skill_key() (lowercase)
    while format_skill_name() produces a readable, shape-PRESERVING display.
    No transformation inserts or removes punctuation/spaces, so the lowercased
    token identity is unchanged ("PowerBi" stays "PowerBi"; no synonym
    mapping, which is deferred).
  * Word caps and junk lists are shared so training == inference.
"""
import ast
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Whitespace / BOM
# ---------------------------------------------------------------------------


def normalize_text(value):
    """Strip BOM, leading/trailing whitespace and collapse repeated whitespace.

    Never changes the meaning of a skill: punctuation and symbols inside the
    string are left untouched.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    text = str(value).replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text.strip())
    return text


# ---------------------------------------------------------------------------
# Skill list delimiters
# ---------------------------------------------------------------------------

# Structural separators for a skill LIST. "/", ":", "-", "." are intentionally
# NOT here so technical names are never split.
SKILL_DELIMITERS_RE = re.compile(r"[,;\n|\uff0c•●◦▪▫‣►◇◆■□»«]+")


def split_skill_items(text, extra_pattern=None):
    """Split a skill value into item parts on structural delimiters only.

    extra_pattern : optional regex for resume-layout-only noise (2+ spaces,
    colon-labelled lines, em/en-dashes, star separators, ...). It is applied
    ONLY when the caller (resume section text) needs it; the canonical
    delimiter contract is SKILL_DELIMITERS_RE for every other pipeline.
    Slashes/hyphens-in-words are never split.
    """
    if text is None:
        return []
    parts = [p for p in SKILL_DELIMITERS_RE.split(str(text)) if p and p.strip()]
    if extra_pattern is None or not parts:
        return parts
    out = []
    for part in parts:
        out.extend(q for q in extra_pattern.split(part) if q and q.strip())
    return out


def split_skill_words(token):
    """Split a token into its component words without losing symbols.

    '+', '#', '.', "'", '-' stay inside words (C++, C#, .NET, Node.js).
    """
    return [w for w in re.split(r"[^a-z0-9+#.'\-]", token.lower()) if w]


# ---------------------------------------------------------------------------
# Limits (shared everywhere)
# ---------------------------------------------------------------------------


# Longest word-count for a single skill phrase. Phrases longer than this are
# treated as sentence fragments, not skills. Same for data cleaning and resume
# extraction so legitimate multi-word skills ("Natural Language Processing",
# "Microsoft SQL Server") survive and the two sides can never disagree.
MAX_SKILL_WORDS = 6

# Upper bound on a token's total length (resume/vocab hygiene). Single-char
# skills such as "R" and "C" are deliberately allowed (no lower bound).
MAX_SKILL_CHARS = 50

# Clearly-invalid noise tokens removed everywhere. Single-character technical
# skills ("R", "C") are NOT here and are never removed.
JUNK_SKILLS = {
    "nan", "none", "n/a", "na", "etc", "etc.", "et al",
    "and", "or", "of", "in", "to", "a", "an", "for", "the", "with",
    "ad", "ads", "com", "e-mail", "email", "emails", "phone", "page",
    "therapy", "hospice", "fax", "address", "website", "me",
    "http", "https", "www", "url", "tel",
}


def is_valid_skill(token):
    """Return True when a token is a meaningful skill (not junk / not numeric)."""
    if token is None:
        return False
    t = normalize_text(token)
    if not t:
        return False
    if t.lower() in JUNK_SKILLS:
        return False
    # Pure numbers (years, values) are not skills. "R", "C", "3D", "2D"
    # survive because they are not numeric-only tokens.
    try:
        float(t)
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Sentence-fragment gate (shared with resume extraction)
# ---------------------------------------------------------------------------

SENTENCE_VERB_LEADS = frozenset({
    "develop", "develops", "developing", "test", "tests", "testing",
    "design", "designs", "designing", "create", "creates", "creating",
    "manage", "manages", "managing", "lead", "leads", "leading",
    "prepare", "prepares", "preparing", "conduct", "conducts",
    "conducting", "evaluate", "evaluates", "evaluating", "analyze",
    "analyzes", "analyzing", "assist", "assists", "assisting",
    "maintain", "maintains", "maintaining", "provide", "provides",
    "providing", "handle", "handles", "handling", "work", "works",
    "working", "collaborate", "collaborates", "collaborating",
    "communicate", "communicates", "communicating", "coordinate",
    "coordinates", "coordinating", "deliver", "delivers", "delivering",
    "build", "builds", "building", "earn", "earns", "earning",
    "implement", "implements", "implementing", "support", "supports",
    "supporting", "supervise", "supervises", "supervising", "monitor",
    "monitors", "monitoring", "train", "trains", "present", "presents",
    "presenting", "write", "writes", "writing", "review", "reviews",
    "reviewing", "plan", "plans", "organize", "organizes", "organizing",
    "perform", "performs", "performing", "improve", "improves",
    "improving", "oversee", "oversees", "overseeing", "ensure",
    "ensures", "ensuring", "assess", "assesses", "assessing",
    "report", "reports", "reporting",
})

# Connector words that shouldn't start or end a skill phrase (leftover from
# line-wrapped fragments like "Systems Analysis and").
FRAGMENT_EDGE_WORDS = frozenset({
    "and", "or", "of", "to", "in", "the", "with", "a", "an", "for",
    "at", "on", "by", "from", "into", "as",
})

MONTH_YEAR_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}\b"
)


def is_junk_fragment(words, lower):
    """Heuristic gate for sentence-like noise vs. a real skill phrase.

    Shared by data cleaning and resume extraction so the same junk rule is
    applied on both sides.
    """
    if MONTH_YEAR_RE.search(lower):
        return True
    if words and (words[0] in FRAGMENT_EDGE_WORDS or words[-1] in FRAGMENT_EDGE_WORDS):
        return True
    if len(words) >= 4 and words[0] in SENTENCE_VERB_LEADS:
        return True
    return False


# ---------------------------------------------------------------------------
# Token cleaning (shared)
# ---------------------------------------------------------------------------

# Surrounding formatting noise removed from the START/END of a candidate item.
# '+' '#' '.' '/' '-' '_' are deliberately absent so "C++", "C#", ".NET",
# "HTML/CSS", "Node.js" and hyphenated names survive untouched.
SAFE_STRIP_CHARS = " \t'\"`[]{}()*<>\\|@$%^&~=»,;:!?•●◦▪▫‣►◇◆■□»«"


def clean_skill_item(item):
    """Clean a single skill item; return the token or None when not skill-like.

    Preserves meaningful symbols; strips only surrounding noise and a single
    trailing period. Applies the shared word cap and the shared junk gates.
    """
    raw = normalize_text(item)
    if not raw:
        return None
    cleaned = raw.strip(SAFE_STRIP_CHARS).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Trailing '.' is sentence-termination noise ("Scikit-learn."). Leading or
    # middle dots are never touched, so ".NET" and "Node.js" stay intact.
    cleaned = cleaned.rstrip(".").strip()
    if not cleaned:
        return None
    if not is_valid_skill(cleaned):
        return None
    words = split_skill_words(cleaned)
    if not words or len(words) > MAX_SKILL_WORDS:
        return None
    if is_junk_fragment(words, cleaned.lower()):
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Casing
# ---------------------------------------------------------------------------

CONNECTOR_WORDS = frozenset({
    "and", "or", "of", "in", "to", "with", "for", "a", "an", "the",
    "on", "at", "by", "from", "as", "vs", "per",
})


def canonical_skill_key(skill):
    """Lowercase, whitespace-collapsed key for case-insensitive comparisons.

    Symbols are preserved exactly, so "C++" != "C" and ".NET" != "NET".
    """
    return normalize_text(skill).lower()


# Acronyms to preserve in uppercase when display-casing a matched skill.
ACRONYMS = {
    "sql", "html", "css", "api", "api's", "http", "https", "json", "xml",
    "aws", "gcp", "azure", "mysql", "nosql", "php", "js", "rest", "soap",
    "ui", "ux", "ai", "ml", "nlp", "cv", "sass", "scss", "seo", "devops",
    "cs", "sde", "saas", "paas", "iaas", "sdk", "jdbc", "odbc", "rdf",
    "vba", "bsd", "smtp", "pop", "imap", "ssh", "ssl", "tls", "dns", "dhcp",
    "tcp", "udp", "ip", "vpn", "lan", "wan", "gpu", "cpu", "ram", "hpc",
    "crm", "erp", "bi", "etl", "oltp", "olap", "dwh", "cd", "ci", "qa",
    "qa/qc", "hr", "ceo", "cto", "cio", "cfo", "b2b", "b2c", "kpi", "roi",
    "bert", "ocr", "spark", "hive", "hdfs", "yarn", "pyspark",
}

# Developer language symbols that must never be mangled into "C".
DEVELOPER_TOKENS = {
    "c": "C", "c++": "C++", "c#": "C#", "f#": "F#",
    "a#": "A#", "d#": "D#", "g#": "G#",
    ".net": ".NET", "asp": "ASP", "asp.net": "ASP.NET",
    "asp.net core": "ASP.NET Core", "vb.net": "VB.NET",
}

# Brand/product names with exact display casing. Every mapping here preserves
# the token's character shape (no added/removed spaces or punctuation), so the
# canonical lowercase identity is unchanged between training and inference.
BRAND_CASING = {
    "azure": "Azure", "mysql": "MySQL", "nosql": "NoSQL", "mongodb": "MongoDB",
    "sqlite": "SQLite", "postgresql": "PostgreSQL", "redis": "Redis",
    "numpy": "NumPy", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
    "tensorflow.js": "TensorFlow.js", "scikit-learn": "Scikit-learn",
    "scikit learn": "Scikit Learn", "matplotlib": "Matplotlib",
    "seaborn": "Seaborn", "plotly": "Plotly", "fastapi": "FastAPI",
    "opencv": "OpenCV", "django": "Django", "flask": "Flask",
    "pandas": "Pandas", "keras": "Keras", "jupyter": "Jupyter",
    "pyspark": "PySpark", "kafka": "Kafka", "spark": "Spark",
    "powerbi": "PowerBi", "power bi": "Power BI", "tableau": "Tableau",
    "github": "GitHub", "gitlab": "GitLab", "d3.js": "D3.js",
    "node.js": "Node.js", "react": "React", "react.js": "React.js",
    "vue.js": "Vue.js", "angular": "Angular", "kubernetes": "Kubernetes",
    "docker": "Docker", "jira": "Jira", "maven": "Maven", "gradle": "Gradle",
    "hadoop": "Hadoop", "big data": "Big Data", "data science": "Data Science",
    "machine learning": "Machine Learning", "deep learning": "Deep Learning",
    "natural language processing": "Natural Language Processing",
    "nlp": "NLP", "computer vision": "Computer Vision",
    "time series analysis": "Time Series Analysis",
    "statistical analysis": "Statistical Analysis",
    "data visualization": "Data Visualization",
    "data analysis": "Data Analysis", "data analytics": "Data Analytics",
    "business analysis": "Business Analysis", "artificial intelligence": "Artificial Intelligence",
    "data mining": "Data Mining", "predictive modeling": "Predictive Modeling",
    "predictive analytics": "Predictive Analytics",
    "amazon web services": "Amazon Web Services", "aws": "AWS",
    "microsoft azure": "Microsoft Azure", "google cloud": "Google Cloud",
    # Languages (mixed-case display, shape-preserving)
    "javascript": "JavaScript", "typescript": "TypeScript", "java": "Java",
    "kotlin": "Kotlin", "swift": "Swift", "dart": "Dart", "ruby": "Ruby",
    "rust": "Rust", "go": "Go", "r": "R", "scala": "Scala",
}


def format_skill_name(skill):
    """Display-casing for a skill that preserves its character shape precisely.

    Words are re-cased (acronyms/dev/brand/connector aware) without inserting
    or removing any punctuation or whitespace. A lowercased token therefore
    always equals the original lowercased token:
        "C++"  -> "C++"   ".NET"  -> ".NET"   "HTML/CSS" -> "HTML/CSS"
    """
    if skill is None:
        return ""
    full = normalize_text(skill)
    if not full:
        return ""
    full_lower = full.lower()
    if full_lower in BRAND_CASING:
        return BRAND_CASING[full_lower]
    if full_lower in DEVELOPER_TOKENS:
        return DEVELOPER_TOKENS[full_lower]

    formatted = []
    for word in full.split():
        lower = word.lower()
        if lower in DEVELOPER_TOKENS:
            formatted.append(DEVELOPER_TOKENS[lower])
        elif lower in BRAND_CASING:
            formatted.append(BRAND_CASING[lower])
        elif lower in ACRONYMS:
            formatted.append(word.upper())
        elif lower in CONNECTOR_WORDS:
            formatted.append(lower)
        elif word.isupper():
            formatted.append(word.upper())
        else:
            formatted.append(lower.capitalize())
    return " ".join(formatted)


# ---------------------------------------------------------------------------
# Structured skill-cell parsing (shared by data cleaning, vocab loading and
# any future training/evaluation feature builder)
# ---------------------------------------------------------------------------


def parse_skill_items(value):
    """Parse a skills value into a list of clean, unique tokens (unedited case).

    - Works on the RAW cell text so structural delimiters (esp. newlines that
      separate job-required skills) survive until the split happens.
    - Handles Python/list literals used in the candidate column.
    - Preserves meaningful symbols; drops junk, numbers, oversize phrases.
    - Removes exact duplicates case-insensitively via canonical_skill_key().
    """
    if value is None:
        return []
    text = str(value).strip() if not isinstance(value, str) else value.strip()
    if not text:
        return []

    items = []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            items = [str(x).strip() for x in parsed if str(x).strip()]
        elif isinstance(parsed, str):
            items = [parsed]
    except (ValueError, SyntaxError):
        items = []

    if not items:
        items = split_skill_items(text)

    seen = set()
    out = []
    for item in items:
        token = clean_skill_item(item)
        if token is None:
            continue
        key = canonical_skill_key(token)
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out


def parse_skills(value):
    """Parse a skills value into readable, display-cased, unique skills."""
    seen = set()
    out = []
    for token in parse_skill_items(value):
        name = format_skill_name(token)
        key = canonical_skill_key(name)
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def skills_to_text(value):
    """Parse a skills value and render it as a comma-separated (recoverable) string."""
    return ", ".join(parse_skills(value))


# ---------------------------------------------------------------------------
# Candidate-job relevance features (NEW matching formulation, Step 3)
# ---------------------------------------------------------------------------
# The shared preprocessing above is the ONLY tokenization for both training and
# inference. These pair-level features compare a candidate's parsed skills to a
# job's required skills using canonical (shape-preserving, case-insensitive)
# tokens; a feature vector built here is identical whether constructed from the
# cleaned CSV row or from an uploaded resume text + job listing.



def candidate_skill_tokens(skills_text):
    """Canonical token set of a skills cell/value. Empty-aware."""
    if not skills_text:
        return set()
    return {canonical_skill_key(t) for t in parse_skill_items(skills_text)}


def candidate_job_match_features(candidate_skills, required_skills):
    """Pair-level overlap features for a (candidate, job) row.

    Returns a TUPLE with exactly the 8 features in MATCH_FEATURE_NAMES order
    (the locked training/inference contract). Order is fixed positionally and
    never depends on dict/item ordering.

    Definition (C = candidate token set, R = required token set):
        intersection   = len(C & R)
        overlap_count  = |C & R|
        jaccard        = |C & R| / |C union R|
        req_coverage   = |C & R| / |R|
        cand_precision = |C & R| / |C|
        candidate_count= |C|
        required_count = |R|
        candidate_has_skills   = 1 if |C| > 0 else 0
        job_has_requirements   = 1 if |R| > 0 else 0

    SAFE-ZERO BEHAVIOR: whenever a denominator set is empty the ratio is 0.0.
    An empty R means there is nothing to cover (req_coverage = 0.0), an empty C
    means nothing to be precise about (cand_precision = 0.0), and a fully empty
    union means no overlap basis (jaccard = 0.0). No cell is ever NaN or inf.

    Candidates and jobs are compared only through their skill content, never
    through the job title or any per-title job record; nothing here depends on
    ``matched_score``, so no target leakage is possible.
    """
    cand = candidate_skill_tokens(candidate_skills)
    req = candidate_skill_tokens(required_skills)
    overlap = len(cand & req)
    union = len(cand | req)
    return (
        float(overlap),
        float(overlap / union) if union else 0.0,
        float(overlap / len(req)) if req else 0.0,
        float(overlap / len(cand)) if cand else 0.0,
        float(len(cand)),
        float(len(req)),
        int(bool(cand)),
        int(bool(req)),
    )


def match_feature_dict(candidate_skills, required_skills):
    """Same vector as candidate_job_match_features, as an ordered dict (reporting)."""
    return dict(zip(MATCH_FEATURE_NAMES, candidate_job_match_features(candidate_skills, required_skills)))


def validate_match_features(vector):
    """Assert a feature vector satisfies the Step-4 contract.

    Raises ValueError if the vector is not exactly 8 numeric, finite values.
    """
    if not isinstance(vector, (tuple, list)) or len(vector) != len(MATCH_FEATURE_NAMES):
        raise ValueError(
            f"feature vector must have {len(MATCH_FEATURE_NAMES)} values, got {len(vector) if hasattr(vector, '__len__') else '?'}"
        )
    for i, v in enumerate(vector):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"feature {MATCH_FEATURE_NAMES[i]} is not numeric: {v!r}")
        if v != v:  # NaN
            raise ValueError(f"feature {MATCH_FEATURE_NAMES[i]} is NaN")
        if v in (float("inf"), float("-inf")):
            raise ValueError(f"feature {MATCH_FEATURE_NAMES[i]} is infinite")
    return vector


MATCH_FEATURE_NAMES = [
    "overlap_count",
    "jaccard",
    "req_coverage",
    "cand_precision",
    "candidate_count",
    "required_count",
    "candidate_has_skills",
    "job_has_requirements",
]


# ---------------------------------------------------------------------------
# Step 6 - richer candidate-side matching features (shared, single source)
# ---------------------------------------------------------------------------
# Everything below is the ONE implementation used for both training and future
# inference. The first 8 rich features are literally the locked Step-4 tuple
# (they are computed by candidate_job_match_features, never re-implemented).
# The remaining features add candidate-level signal (experience years, education
# level, skill-category breadth, category overlap, TF-IDF lexical similarity).
#
# LEAKAGE AUDIT (per feature): no feature uses matched_score, job_title, or the
# target; job-side content enters ONLY as requirement CONTENT (experience-min
# years / education level parsed from the requirement text, exactly like
# required_skills). Candidate-side values come from the resume itself, not from
# the answer. All parsers are deterministic string->number functions so
# training and inference cannot diverge.
# ---------------------------------------------------------------------------


# Anchor date for "Current"/"Present"/"Ongoing" work. Set to the latest
# observed end date (Oct 2023) present in DATA/resume_data.csv. Deterministic
# and NOT derived from matched_score or any job field.
CURRENT_REFERENCE_DATE = (2023, 10)

_MONTH_NAMES = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}
_DATE_TEXT_RE = re.compile(
    r"^\s*(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{4})", re.I)
_DATE_NUM_RE = re.compile(r"^\s*(\d{1,2})/(\d{4})\s*$")
_DATE_YEAR_RE = re.compile(r"^\s*(\d{4})\s*$")
_CURRENT_WORDS = {"current", "present", "ongoing", "now"}


def parse_resume_dates(list_literal):
    """Parse a resume date LIST-LITERAL cell into (year, month) tuples.

    Accepts the formats observed in the candidate column: "['May 2019']",
    "['2/2019', ...]", "['2014', 'N/A', None, 'Current']". Tokens that cannot
    be interpreted ('N/A', None, masked '20XX' years) become None entries and
    are skipped by callers. 'Current'/'Present'/'Ongoing' resolve to
    CURRENT_REFERENCE_DATE. Returns a list aligned to the source order.
    """
    if list_literal is None:
        return []
    text = normalize_text(list_literal)
    if not text or text == "[]":
        return []
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if it is None:
            out.append(None)
            continue
        s = str(it).strip()
        low = s.lower()
        if any(c in low for c in _CURRENT_WORDS):
            out.append(CURRENT_REFERENCE_DATE)
            continue
        if low in ("n/a", "na"):
            out.append(None)
            continue
        m = _DATE_TEXT_RE.match(s)
        if m:
            out.append((int(m.group(2)), _MONTH_NAMES[m.group(1).lower()[:3]]))
            continue
        m = _DATE_NUM_RE.match(s)
        if m:
            out.append((int(m.group(2)), int(m.group(1))))
            continue
        m = _DATE_YEAR_RE.match(s)
        if m:
            out.append((int(m.group(1)), 1))
            continue
        out.append(None)  # masked/unknown tokens ("20XX") -> skipped
    return out


def experience_years_from_dates(start_dates_literal, end_dates_literal):
    """Total professional experience in years from paired start/end date lists.

    Each start date aligns with the end date at the same index; a missing end
    ('Current', 'Present', 'Ongoing', absent) is anchored at
    CURRENT_REFERENCE_DATE. Overlapping/multi-role summers are reported as the
    raw summed span (a resume-faithful estimate). Returns None when no start
    date could be parsed.
    """
    starts = [d for d in parse_resume_dates(start_dates_literal) if d is not None]
    ends = parse_resume_dates(end_dates_literal)
    if not starts:
        return None
    total_months = 0.0
    for i, (sy, sm) in enumerate(starts):
        ey, em = (ends[i] if i < len(ends) and ends[i] is not None
                  else CURRENT_REFERENCE_DATE)
        total_months += max(0.0, (ey - sy) * 12 + (em - sm))
    return total_months / 12.0


# Education level ordering (ordinal, higher is more advanced). Keywords are
# matched as whole words. Education text comes from the resume's own degree
# fields at training and from resume parsing at inference.
_EDU_LEVEL_RULES = [
    (4, ("phd", "ph.d", "doctorate", "doctoral")),
    (3, ("master", "mba", "m.sc", "msc", "m.com", "mcom", "mbm",
         "m.eng", "meng", "m.phil", "ms ") ),
    (2, ("bachelor", "b.sc", "bsc", "bba", "b.com", "bcom", "b.eng",
         "beng", "b.e.", "be ", "honours", "honors", "b.a", "btech",
         "b.tech", "bs ", "b.s.")),
    (1, ("diploma", "diplomas", "hsc", "ssc", "intermediate", "tin")),
]


def education_level_from_text(*texts):
    """Highest education level 0..4 found in the resume/job text.

    0 = none/unknown, 1 = school/diploma, 2 = bachelor, 3 = master, 4 = PhD.
    """
    joined = " ".join(normalize_text(t) for t in texts if t).lower()
    if not joined:
        return 0
    best = 0
    for level, kws in _EDU_LEVEL_RULES:
        for kw in kws:
            if re.search(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", joined):
                best = max(best, level)
    return best


def job_experience_min_years(experience_required):
    """Minimum years a job actually requires, parsed from its requirement text.

    "At least 5 years" -> 5, "3 to 7 years" -> 3, "At least 1 year(s)" -> 1,
    "" -> None. This is job CONTENT (like required_skills), never the title.
    """
    text = normalize_text(experience_required)
    if not text:
        return None
    nums = [int(n) for n in re.findall(r"\d+", text)]
    return float(min(nums)) if nums else None


# Semantic categories for skills, used by the category-overlap features.
# Assignment is deterministic: a token goes to the FIRST category in this
# priority order that contains a matching keyword (single category per token).
# Simulation/CAD keywords are deliberately checked AFTER business/data keywords
# so "mechanical design" lands on design_cad only when no business term applies.
SKILL_CATEGORY_PRIORITY = [
    ("programming", ("python", "java", "javascript", "typescript", "c++", "c#",
                     "c/c++", ".net", "php", "kotlin", "swift", "ruby", "scala",
                     "perl", "haskell", "matlab", "pascal", "fortran", "assembl",
                     "visual basic", "vb", "f#", "objective-c", "dart", "lua", "c",
                     "golang", "node", "delphi", "erlang", "prolog")),
    ("database", ("sql", "database", "db2", "oracle", "mysql", "postgres",
                  "mongodb", "nosql", "sqlite", "redis", "hive", "hdfs",
                  "bigquery", "snowflake", "redshift", "ssis", "ssrs", "ssms",
                  "etl", "cassandra", "elasticsearch", "dynamodb")),
    ("web", ("html", "css", "react", "angular", "vue", "bootstrap", "jquery",
             "django", "flask", "frontend", "front end", "web design",
             "web development", "webgl", "wordpress", "next.js", "rest api",
             "restapi", "graphql", "api", "web services", "ui design", "ux",
             "tailwind", "full stack", "backend")),
    ("business", ("market", "sales", "marketing", "finance", "financial",
                  "account", "audit", "tax", "vat", "payroll", "invoice",
                  "budget", "hr ", "human resource", "recruit", "employee",
                  "negoti", "communication", "leadership", "lead ", "team",
                  "customer", "client", "supply chain", "procurement", "vendor",
                  "advertis", "brand", "crm", "compliance", "risk", "management",
                  "legal", "insurance", "banking", "real estate", "property",
                  "mortgage", "lending")),
    ("data_science", ("machine learning", "deep learning", "nlp",
                      "natural language processing", "computer vision",
                      "tensorflow", "pytorch", "keras", "sklearn", "scikit",
                      "pandas", "numpy", "spark", "statistic", "regression",
                      "classification", "clustering", "analytics", "analysis",
                      "modeling", "modelling", "prediction", "forecast",
                      "data science", "data mining", "data wrangling",
                      "data engineering", "visualization", "visualisation",
                      "tableau", "power bi", "llm", "gpt", "bert",
                      "artificial intelligence", "data analysis", "big data",
                      "recommendation", "anomaly", "time series")),
    ("infrastructure", ("aws", "azure", "gcp", "cloud", "docker", "kubernetes",
                        "linux", "ci/cd", "jenkins", "terraform", "ansible",
                        "git", "github", "network", "server", "vpn", "firewall",
                        "tcp/ip", "windows server", "unix", "bash", "shell",
                        "devops", "mainframe", "vmware", "virtualization",
                        "os/2", "active directory", "routing", "switching",
                        "security", "endpoint", "storage", "backup")),
    ("design_cad", ("cad", "autocad", "solidworks", "catia", "revit", "3d",
                    "design", "draft", "drawing", "blueprint", "photoshop",
                    "illustrator", "indesign", "sketch", "lathe", "cnc",
                    "weld", "machin", "cadence", "orcad", "proteus",
                    "simulink", "labview", "mechanical", "civil", "structural",
                    "architect", "fmea", "nastran", "patran", "ansys",
                    "abaqus", "hypermesh")),
    ("office_tools", ("excel", "powerpoint", "word", "ms office",
                      "microsoft office", "outlook", "sharepoint", "access",
                      "google docs", "gantt", "jira", "trello", "visio",
                      "typing", "pivot", "spreadsheet")),
]
SKILL_CATEGORY_NAMES = tuple(name for name, _ in SKILL_CATEGORY_PRIORITY)


def _token_matches_keyword(token, keyword):
    """Boundary-aware keyword match that never mangles symbol tokens.

    "c++", "c#", ".NET" only match their exact keyword (no "c" substring
    explosion); multi-char keywords also match inside tokens or as whole words.
    """
    t = canonical_skill_key(token)
    k = canonical_skill_key(keyword)
    if t == k:
        return True
    if len(k) < 2:
        # bare symbol/language tokens ("c", "r") must match exactly
        return False
    return (
        k in t
        or t.startswith(k + " ")
        or t.endswith(" " + k)
        or (" " + k + " " in t)
    )


def skill_categories(skills_text):
    """Set of semantic categories covered by a skills value (empty-aware)."""
    cats = set()
    for token in parse_skill_items(skills_text):
        for cat, kws in SKILL_CATEGORY_PRIORITY:
            if any(_token_matches_keyword(token, kw) for kw in kws):
                cats.add(cat)
                break
    return cats


def build_skills_tfidf(corpus_docs):
    """Fit the shared TF-IDF vectorizer on required-skills documents.

    The tokenizer uses parse_skill_items + canonical_skill_key so symbol names
    ("C++", ".NET", "HTML/CSS") stay single tokens and bigrams capture phrases.
    Passing the same required-skills documents makes training and inference use
    the exact same vector space.
    """
    tokenizer = lambda txt: [canonical_skill_key(t) for t in parse_skill_items(txt)]
    return TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True,
        tokenizer=tokenizer, token_pattern=None,
    ).fit([normalize_text(d) for d in corpus_docs])


def skills_tfidf_similarity(candidate_skills, required_skills, vectorizer=None):
    """Cosine similarity in the shared TF-IDF skill space (0..1).

    0 when either side is empty or no vectorizer is provided. Same vector space
    at training and inference (the vectorizer is fit on the job-requirement
    corpus, not on candidates). Inputs are transformed as their original
    delimiter-separated text so the shared tokenizer sees every skill item.
    """
    cand = normalize_text(candidate_skills)
    req = normalize_text(required_skills)
    if not cand or not req or vectorizer is None:
        return 0.0
    v = vectorizer.transform([cand, req])
    return float(cosine_similarity(v[0:1], v[1:2])[0][0])


def load_candidate_profiles(resume_csv):
    """Candidate profile {candidate_skills: dict} extracted from the raw resume
    spreadsheet. Reads ONLY candidate-side fields (skills, dates, degrees) and
    NEVER matched_score/job fields, so no target information can leak in.
    """
    import pandas as pd
    raw = pd.read_csv(resume_csv, keep_default_na=False, na_values=[])
    rendered = raw["skills"].map(lambda s: ", ".join(parse_skills(s)))
    profiles = {}
    for key, row in raw.assign(_r=rendered).drop_duplicates("_r").iterrows():
        csk = row["_r"]
        exp = experience_years_from_dates(row["start_dates"], row["end_dates"])
        edu = education_level_from_text(row["degree_names"], row["result_types"])
        profiles[csk] = {
            "experience_years": exp,       # float or None (nothing parseable)
            "education_level": edu,        # int 0..4
        }
    return profiles


RICH_MATCH_FEATURE_NAMES = [
    # 8 locked Step-4 features (identical values, not re-implemented)
    "overlap_count",
    "jaccard",
    "req_coverage",
    "cand_precision",
    "candidate_count",
    "required_count",
    "candidate_has_skills",
    "job_has_requirements",
    # 9 Step-6 richer candidate/job compatibility features
    "cand_exp_years",
    "exp_gap",
    "cand_edu_level",
    "edu_gap",
    "cand_cat_breadth",
    "cat_overlap",
    "cat_req_coverage",
    "skills_tfidf_sim",
    "has_profile",
]


def candidate_job_match_features_rich(
    candidate_skills,
    required_skills,
    profile=None,
    job_experience_required="",
    job_education_required="",
    skills_tfidf=None,
):
    """17-feature matching vector (locked Step-6 contract, fixed tuple order).

    profile : candidate dict {'experience_years': float|None,
                              'education_level': int} built from the resume.
              None/missing fields -> safe zeros (same code path at inference).
    job_experience_required / job_education_required : the job's own requirement
              CONTENT, parsed here with the shared parsers (never the title).
    skills_tfidf : the shared TF-IDF vectorizer (built once from the job
              requirement corpus); None -> skills_tfidf_sim = 0.0.

    SAFE ZEROS (same policy as the Step-4 contract): unknown candidate fields,
    missing job requirements and empty category sets all produce 0.0, never
    NaN/inf. Feature values never come from matched_score or job_title.
    """
    base = candidate_job_match_features(candidate_skills, required_skills)
    if profile is None:
        cand_exp, cand_edu, has_profile = 0.0, 0, 0
    else:
        cand_exp = float(profile.get("experience_years") or 0.0)
        cand_edu = int(profile.get("education_level") or 0)
        has_profile = int(bool(profile.get("experience_years") is not None
                               or profile.get("education_level")))

    job_min = job_experience_min_years(job_experience_required)
    job_edu = education_level_from_text(job_education_required)
    exp_gap = (cand_exp - job_min) if (job_min is not None and cand_exp > 0) else 0.0
    edu_gap = float(cand_edu - job_edu) if (job_edu > 0 and cand_edu > 0) else 0.0

    cats_c = skill_categories(candidate_skills)
    cats_r = skill_categories(required_skills)
    cat_overlap = len(cats_c & cats_r)
    sim = skills_tfidf_similarity(candidate_skills, required_skills, skills_tfidf)

    return base + (
        cand_exp,
        exp_gap,
        float(cand_edu),
        edu_gap,
        float(len(cats_c)),
        float(cat_overlap),
        float(cat_overlap / len(cats_r)) if cats_r else 0.0,
        sim,
        float(has_profile),
    )


def rich_match_feature_dict(candidate_skills, required_skills, **kwargs):
    """Same 17-vector as an ordered dict (reporting)."""
    return dict(zip(RICH_MATCH_FEATURE_NAMES,
                    candidate_job_match_features_rich(candidate_skills, required_skills, **kwargs)))


def validate_match_features_rich(vector):
    """Assert a vector satisfies the Step-6 contract (17 finite numerics)."""
    if not isinstance(vector, (tuple, list)) or len(vector) != len(RICH_MATCH_FEATURE_NAMES):
        raise ValueError(
            f"rich feature vector must have {len(RICH_MATCH_FEATURE_NAMES)} values, "
            f"got {len(vector) if hasattr(vector, '__len__') else '?'}"
        )
    for i, v in enumerate(vector):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"rich feature {RICH_MATCH_FEATURE_NAMES[i]} is not numeric: {v!r}")
        if v != v:
            raise ValueError(f"rich feature {RICH_MATCH_FEATURE_NAMES[i]} is NaN")
        if v in (float("inf"), float("-inf")):
            raise ValueError(f"rich feature {RICH_MATCH_FEATURE_NAMES[i]} is infinite")
    return vector


# ---------------------------------------------------------------------------
# Job document builder (training == inference)
# ---------------------------------------------------------------------------


def build_job_document(
    job_title,
    required_skills,
    responsibilities="",
    experience_required="",
    education_required="",
):
    """Build the exact text document used for a job row in BOTH training and
    inference. Required skills are doubled so a candidate's primary skills
    dominate over dataset jargon shared across roles.
    """
    title = normalize_text(job_title)
    req = normalize_text(required_skills)
    resp = normalize_text(responsibilities)
    exp = normalize_text(experience_required)
    edu = normalize_text(education_required)
    return f"{title} {req} {req} {resp} {exp} {edu}".strip()