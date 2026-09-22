import re
from pathlib import Path

import pandas as pd
import pymupdf

from UTILS.features import (
    ACRONYMS,
    FRAGMENT_EDGE_WORDS,
    JUNK_SKILLS,
    MAX_SKILL_CHARS,
    MAX_SKILL_WORDS,
    canonical_skill_key,
    clean_skill_item,
    format_skill_name,
    normalize_text,
    parse_skill_items,
    split_skill_items,
    split_skill_words,
)

STOP_WORDS = {
    "and", "or", "the", "in", "on", "at", "for", "to", "with", "of", "as",
    "by", "from", "into", "all", "any", "its",
    "a", "an", "is", "are", "was", "were", "be", "being", "been",
    "work", "working", "works", "skills", "skill", "effective",
    "experience", "experiences", "ability", "abilities", "knowledge",
    "required", "job", "candidate", "team", "good", "strong", "engineer",
    "engineering", "software", "senior", "junior", "management", "manager",
    "lead", "leadership", "design", "designer", "developer", "development",
    "professional", "career", "company", "role", "position",
    "responsibilities", "responsibility", "related", "personal", "person",
    "excellent", "communication", "interpersonal", "detail", "oriented",
    "skillset", "years", "year", "plus", "including", "etc", "based",
    "using", "use", "used", "within", "across", "through", "familiar",
    "proficient", "demonstrated", "proven", "solid", "basic", "hands",
    "self", "motivated", "teamwork", "analysis", "analyst", "analytics",
    "ms", "services", "service", "business", "support", "technical",
    "office", "tool", "tools", "language", "languages", "section",
    "sections", "header", "headers", "heading", "headings",
    "cases", "class", "com", "drive", "features", "file", "logo", "page",
    "pdf", "photo", "website", "tips", "tip", "recruiters",
    "resource", "resources", "template", "resume", "curriculum", "vitae",
    "vita", "created", "canva", "member", "members", "date", "area",
    "areas", "various", "several", "appropriate",
    "me", "ad", "ads", "email", "emails", "e-mail", "contact", "address",
    "addresses", "details", "phone", "fax",
    "therapy", "hospice",
}

# Single-word technical terms kept even when not present in the dataset
# vocabulary (their standalone words would otherwise read as generic noise).
# Single-char languages "R", "C" and "go" are included so those skills survive
# the vocabulary gate just like every other technical term.
TECHNICAL_SINGLE_WORDS = frozenset({
    "slam", "ros", "arduino", "embedded", "opencv", "fintech", "biotech",
    "nanotech", "aerospace", "biomedical", "cybersecurity", "telecom",
    "blockchain", "cryptography",
    "r", "c", "go",
})

# Common two-word technical phrasings that are explicit resume skills but are
# not present verbatim in the 28-title dataset vocabulary. Listing them lets the
# extraction keep the resume's own words (never invented) instead of dropping
# them on the two-capitalised-word noise rule, while contact names / sidebar
# noise ("Lily River", "Emeka Afia") still fail the shape check.
TECHNICAL_PHRASES = frozenset({
    "machine learning", "deep learning", "computer vision", "natural language",
    "language processing", "computer programming", "artificial intelligence",
    "data science", "data engineering", "data analysis", "data analytics",
    "data visualization", "business intelligence", "business analysis",
    "business analytics", "quality assurance", "quality control",
    "project management", "product management", "supply chain", "data mining",
    "web development", "web design", "front end", "back end", "full stack",
    "software engineering", "software development", "software testing",
    "system administration", "systems engineering", "network administration",
    "network engineering", "cloud computing", "cloud architecture",
    "devops engineering", "platform engineering", "information security",
    "cyber security", "digital marketing", "content marketing",
    "user experience", "user interface", "graphic design",
    "mechanical engineering", "electrical engineering", "civil engineering",
    "industrial engineering", "chemical engineering", "automotive engineering",
    "aerospace engineering", "robotics engineering", "robotic systems",
    "embedded systems", "control systems", "control engineering",
    "power systems", "power electronics", "signal processing",
    "image processing", "systems analysis", "business development",
})


DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"
CSV_PATH = DATA_DIR / "cleaned_jobs.csv"

_skills_cache = None


def dataset_available():
    """Return True if the cleaned jobs dataset exists on disk."""
    return CSV_PATH.exists()


def load_skills_from_dataset(use_cache=True):
    """Read dataset and extract clean skill terms from the required_skills column.

    use_cache : bool – cache the vocabulary after first load to avoid re-reading
    the CSV on every request.
    """
    global _skills_cache

    if use_cache and _skills_cache is not None:
        return _skills_cache

    skills_set = set()

    if not CSV_PATH.exists():
        print(f"Dataset not found: {CSV_PATH}")
        if use_cache:
            _skills_cache = []
        return []

    try:
        df = pd.read_csv(CSV_PATH)

        possible_cols = ["required_skills", "skills", "Job Requirement", "Key Skills"]
        target_col = None
        for col in possible_cols:
            if col in df.columns:
                target_col = col
                break

        if target_col:
            for item in df[target_col].fillna("").astype(str):
                # Shared preprocessing (same delimiters / symbol handling / word cap
                # as data cleaning and inference). "/" is never a split point, so
                # "HTML/CSS", "CI/CD" and "C/C++" remain single vocabulary entries.
                for token in parse_skill_items(item):
                    key = canonical_skill_key(token)
                    if key and len(key) <= MAX_SKILL_CHARS and key not in STOP_WORDS:
                        skills_set.add(key)

    except Exception as e:
        print(f"Dataset loading error: {e}")

    if use_cache:
        _skills_cache = list(skills_set)
        return _skills_cache
    return list(skills_set)


def _page_reading_order_text(page):
    """Return a page's text in visual reading order.

    Canva-style two-column resumes are stored in a content-stream order that
    interleaves the sidebar with the main column, so heading lines often land
    AFTER the content they label. Text blocks are grouped into columns by their
    horizontal centre and each column is emitted top-to-bottom (left column
    first), keeping every section heading in front of its own content so the
    section-aware extractor sees correct heading -> content boundaries.

    Single-column pages fall back to a plain top-to-bottom, left-to-right sort.
    """
    blocks = [b for b in page.get_text("blocks") if b[4] and b[4].strip()]
    if not blocks:
        return ""

    width = page.rect.width or 1.0
    mid = width / 2.0
    margin = max(15.0, width * 0.06)

    columns = {0: [], 1: []}
    for b in blocks:
        centre = (b[0] + b[2]) / 2.0
        if centre < mid - margin:
            columns[0].append(b)
        elif centre > mid + margin:
            columns[1].append(b)
        elif centre < mid:
            columns[0].append(b)
        else:
            columns[1].append(b)

    if len(columns[0]) >= 2 and len(columns[1]) >= 2:
        ordered = (
            sorted(columns[0], key=lambda b: (round(b[1], 1), b[0]))
            + sorted(columns[1], key=lambda b: (round(b[1], 1), b[0]))
        )
    else:
        ordered = list(blocks)

    return "\n".join(
        b[4].rstrip() for b in ordered if b[4] and b[4].strip()
    )


def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file using PyMuPDF.

    Text is read in visual reading order (see _page_reading_order_text) so
    section headings precede the content they label even in multi-column
    layouts.
    """
    text = ""
    try:
        doc = pymupdf.open(pdf_path)
        for page in doc:
            extracted = _page_reading_order_text(page)
            if extracted:
                text += extracted + "\n"
        doc.close()
    except Exception as e:
        print(f"PDF Read Error: {e}")

    return text


# ---------------------------------------------------------------------------
# Section-aware skill extraction (handles multi-column Canva resumes).
# ---------------------------------------------------------------------------

# Normalized heading names that introduce skill-related content. When a skills
# heading is found, only the content up to the next major heading is used.
SKILL_HEADINGS = {
    "skill", "skills", "key skills", "my skills", "skills summary",
    "skill set", "skill sets", "skillset",
    "technical skills", "relevant skills", "professional skills",
    "core skills", "core competencies", "core competency",
    "competencies", "competency", "professional competencies",
    "skills and competencies", "skills and expertise", "skills and tools",
    "skills and technologies", "technical skills and tools",
    "expertise", "areas of expertise", "area of expertise",
    "expertise areas", "technical expertise", "expertise and skills",
    "proficiencies", "proficiency areas", "technical proficiencies",
    "proficiencies and tools", "technologies", "technology",
    "tools and technologies", "technologies and tools",
    "technology stack", "tech stack", "it skills",
    "computer skills", "software skills",
}

# Normalized names of major resume sections. A skills section runs until the
# next major heading (Education, Experience, Projects, Volunteering, ...).
MAJOR_HEADINGS = {
    "summary", "professional summary", "career summary", "career objective",
    "objective", "profile", "professional profile", "about me", "overview",
    "qualifications summary", "strengths", "personal summary",
    "experience", "work experience", "professional experience",
    "employment", "employment history", "work history", "career history",
    "professional history", "work record",
    "education", "academic background", "education and training",
    "qualifications", "certifications", "certification", "training",
    "certificates", "certificates and licenses", "certificates and training",
    "licenses", "licences", "license", "licensing", "credentials",
    "licenses and certifications", "certifications and licenses",
    "academic qualifications", "educational qualifications",
    "academic education", "academic record", "academic history",
    "education history", "educational background", "educational history",
    "professional qualifications",
    "honors", "honours", "honors and awards", "awards and honors",
    "projects", "project work", "accomplishments", "achievements",
    "awards", "publications", "languages", "interests", "hobbies",
    "activities", "references", "work references", "volunteer", "volunteering",
    "volunteer experience", "volunteer work", "volunteer work and",
    "volunteer work and experience", "additional information", "personal details",
    "personal information", "contact", "contact details",
    "contact information", "declaration", "leadership",
    "leadership experience", "affiliations", "memberships", "portfolio",
}

# Boilerplate / CV-template guidance that must never be treated as resume
# content. A skills section is truncated at the first marker.
BOILERPLATE_SECTION_MARKERS = (
    "resource page",
    "tips for creating an effective cv",
    "what recruiters usually look for",
    "remember to add how long you have stayed",
    "when writing out your achievements",
    "you may also list specific software or tools",
    "it is ideal to follow this format",
)

# Resume-layout-only splits layered on top of the shared skill delimiters in
# UTILS/features.split_skill_items(): runs of 2+ spaces (multi-column Canva
# layouts), tabs, ellipsis, colon labels ("Languages: ..."), em/en-dashes,
# star separators, and single hyphens surrounded by spaces. Bare hyphens inside
# words are NEVER split ("K-Nearest Neighbors" stays intact), matching the
# shared no-punctuation-splitting contract.
RESUME_SPLIT_EXTRAS = re.compile(
    r"\s{2,}"
    r"|\t+"
    r"|…+"
    r"|:"
    r"|\s*[–—]\s*"
    r"|\s*[*]\s*"
    r"|\s+[-]\s+"
)


def _normalize_heading(line):
    """Normalize a heading line into its canonical lowercase word form."""
    s = line.strip().strip(":;. ")
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _is_header_like(line):
    """Heuristic to tell a short section header from a content line."""
    s = line.strip()
    if not s:
        return False
    # Commas/semicolons/pipes and long text belong to inline content after a
    # colon, so all checks apply to the heading prefix (the part before ":").
    # This keeps "Core Competencies: X, Y" a heading while content lines like
    # "Python, SQL, Java" are not.
    heading_part = s.split(":", 1)[0]
    if re.match(r"^[\s•●◦▪▫‣►◇◆■□»|\-–—*]+", heading_part):
        return False
    if s.endswith((".", "…", "!", "?")):
        return False
    if re.search(r"[,;|]", heading_part):
        return False
    if not heading_part[0].isalpha():
        return False
    if len(heading_part) > 40:
        return False
    if len(heading_part.split()) > 5:
        return False
    return True


def _classify_line(raw):
    """Classify a line as 'skills', 'major', 'boilerplate', or None."""
    line = raw.strip()
    if not line:
        return None

    norm = _normalize_heading(line)
    if not norm:
        return None

    if any(marker in norm for marker in BOILERPLATE_SECTION_MARKERS):
        return "boilerplate"

    if not _is_header_like(line):
        return None

    if norm in SKILL_HEADINGS:
        return "skills"
    if norm in MAJOR_HEADINGS:
        return "major"

    # "Heading: inline content" form (e.g. "Core Competencies: Data Analysis",
    # "Skills: Python, SQL") – classify via the heading prefix so the inline
    # content is captured instead of surfacing as a stray token.
    if ":" in line:
        prefix_norm = _normalize_heading(line.split(":", 1)[0])
        if prefix_norm in SKILL_HEADINGS:
            return "skills"
        if prefix_norm in MAJOR_HEADINGS:
            return "major"

    return None


def _cut_at_boilerplate(content_lines):
    """Drop everything from the first boilerplate marker onward.

    Keep whole lines so newlines stay available as delimiters. The marker is
    searched against whitespace-normalized accumulated text so phrases that
    wrap across lines are still caught.
    """
    prefix = ""
    kept = []
    for line in content_lines:
        prefix = re.sub(r"\s+", " ", (prefix + " " + line).strip())
        lower = prefix.lower()
        if any(marker in lower for marker in BOILERPLATE_SECTION_MARKERS):
            break
        kept.append(line)
    return kept


def _capture_skill_sections(text):
    """Find skill sections (heading -> next major heading / boilerplate).

    Returns a list of line-lists (one per skills heading) or None when no
    skills heading exists in the text.
    """
    lines = text.splitlines()

    headers = []
    for i, raw in enumerate(lines):
        kind = _classify_line(raw)
        if kind in ("skills", "major", "boilerplate"):
            headers.append((i, kind))

    if not any(kind == "skills" for _, kind in headers):
        return None

    sections = []
    for n, (h_idx, kind) in enumerate(headers):
        if kind != "skills":
            continue

        raw_header = lines[h_idx]
        inline = [raw_header.split(":", 1)[1]] if ":" in raw_header else []
        start = h_idx + 1
        end = headers[n + 1][0] if n + 1 < len(headers) else len(lines)

        section_lines = _cut_at_boilerplate(inline + lines[start:end])
        if section_lines:
            sections.append(section_lines)

    return sections or None


def _clean_token(part):
    """Clean a single resume candidate, returning None when it isn't skill-like.

    The shared gate (UTILS.features.clean_skill_item) applies the same symbol
    preservation, word cap and junk rules used by data cleaning; this function
    adds resume-text-only checks (emails, addresses, dates, headings, and
    stop-word-heavy fragments).
    """
    # Shared cleaning + word cap + junk-fragment gate.
    token = clean_skill_item(part)
    if token is None:
        return None

    if len(token) > MAX_SKILL_CHARS:
        return None

    lower = token.lower()

    # Email addresses and address lines (starts with a bare number) are not
    # skills.
    if "@" in token:
        return None

    words = split_skill_words(token)
    if not words:
        return None
    if words[0].isdigit() and len(words) >= 2:
        return None

    if all(w in STOP_WORDS for w in words):
        # This gate is only ever reached for text inside a detected skills
        # section, where a capitalized single item ("Design", "Development",
        # "Leadership") is an explicitly listed skill, not prose. Lowercase
        # words and true junk ("The", "and", "Email", "Phone") stay rejected.
        low = words[0]
        if token.islower() or low in FRAGMENT_EDGE_WORDS or low in JUNK_SKILLS:
            return None

    stop_count = sum(1 for w in words if w in STOP_WORDS)
    # Inside a skills section a phrase that is more than half connector/soft
    # words ("in and of its", "and with the ability") is prose noise, while
    # legitimate phrases with a few connectors ("Hospice and Palliative Care",
    # "Quality Assurance and Control") must survive. Months/dates/verb-leads/
    # edge-word fragments are already rejected by is_junk_fragment upstream.
    if len(words) >= 3 and stop_count * 2 > len(words):
        return None

    if lower in SKILL_HEADINGS or lower in MAJOR_HEADINGS:
        return None

    # Page-number / date artifacts left behind by template text.
    if re.search(r"(^|\s)(page|p)\s*\.?\s*\d+(\s|$)", lower):
        return None
    if re.fullmatch(r"\d{4}", lower) or re.fullmatch(r"\d{4}\s*[-–]\s*\d{4}", lower):
        return None

    # Year / date-span artifacts from job or volunteer entries.
    if re.match(r"^20\d\d\b", lower):
        return None

    return token


def _is_skill_shaped(token, vocabulary):
    """Vocabulary-aware shape gate for candidates inside a skills section.

    Heuristics alone cannot tell "Lily River" (contact sidebar noise) from
    "Computer Vision" (a skill), so candidates are tested against the dataset
    vocabulary and symbol/acronym/casing allowlists. Because this is only ever
    called on text already isolated inside a detected skills section (whose
    actual junk - emails, addresses, dates, fragments - is filtered upstream
    by _clean_token), explicitly itemised entries are trusted: a capitalized
    single word ("Pandas", "Docker", "Isolation", "Rheumatology") is an
    explicit list entry, and two-word phrases ("Packaging Design", "Critical
    Thinking") are kept rather than assumed to be proper-noun names.
    """
    lower = token.lower()
    pieces = [p for p in re.split(r"[^A-Za-z0-9+#.'\-]", token) if p]
    words = [p.lower() for p in pieces]
    if not words:
        return False
    if len(words) == 1:
        word = words[0]
        # Tokens containing explicit technical symbols (+, #, ., /) are
        # overwhelmingly real technical skill names (Node.js, .NET, C#, C/C++,
        # ASP.NET) — accept them unless they match known junk.
        has_symbol = any(c in word for c in "+#./")
        is_itemised = (
            pieces[0][:1].isupper()
            and word not in FRAGMENT_EDGE_WORDS
            and word not in JUNK_SKILLS
        )
        return (
            token.isupper()
            or word in vocabulary
            or word in ACRONYMS
            or word in TECHNICAL_SINGLE_WORDS
            or has_symbol
            or is_itemised
        )
    return True


def _tokenize_section(section_lines, debug=False):
    """Turn the raw lines of a skills section into clean candidate tokens.

    debug : dev-only diagnostic. When True, prints each candidate with its
    accept/reject decision. Never enabled by the production path, so real
    resumes are never logged.
    """
    vocabulary = set(load_skills_from_dataset())
    tokens = []
    for line in section_lines:
        for part in split_skill_items(line, RESUME_SPLIT_EXTRAS):
            token = _clean_token(part)
            shaped = bool(token) and _is_skill_shaped(token, vocabulary)
            if debug:
                if shaped:
                    print(f"[extractor]   accept {part!r} -> {token!r}")
                elif token:
                    print(f"[extractor]   reject-shape {part!r}")
                else:
                    print(f"[extractor]   reject-junk {part!r}")
            if shaped:
                tokens.append(token)
    return tokens


def _format_unique(tokens):
    """Apply casing rules, drop duplicates (case-insensitive), sort."""
    seen = set()
    formatted = []
    for token in tokens:
        name = format_skill_name(token)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        formatted.append(name)
    return sorted(formatted)


def _fallback_vocab_match(text):
    """Match the dataset vocabulary against text using word boundaries.

    Used when the resume has no recognizable skills section.
    """
    haystack = " " + re.sub(r"\s+", " ", text.lower()) + " "
    vocabulary = load_skills_from_dataset()
    found = set()

    for skill in vocabulary:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, haystack):
            found.add(format_skill_name(skill))

    return sorted(found)


def extract_skills_from_text(text, debug=False):
    """Extract a clean, unique list of skills from resume text.

    Primary path: locate "Skills / Relevant Skills / Expertise / Core
    Competencies / Technical Skills" sections, tokenize their content on
    bullets/commas/pipes/newlines, and filter template junk.
    Fallback: when no skills section exists, match the dataset vocabulary
    against the whole text with word boundaries.

    debug : dev-only diagnostic flag (see _tokenize_section). The production
    caller (APP/app.py) never sets it, so real resumes are never logged.
    """
    if not text or not text.strip():
        return []

    sections = _capture_skill_sections(text)

    if sections is None:
        return _fallback_vocab_match(text)

    tokens = []
    for section_lines in sections:
        tokens.extend(_tokenize_section(section_lines, debug=debug))

    extracted = _format_unique(tokens)

    # Almost nothing captured -> supplement with direct vocabulary matches.
    if len(extracted) < 2:
        vocab = _fallback_vocab_match(text)
        if vocab:
            extracted = _format_unique(extracted + vocab)

    return extracted