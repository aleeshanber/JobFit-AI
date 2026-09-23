from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from UTILS.features import build_job_document

RARE_CLASS = "__rare__"


def stratify_labels(series):
    """Return split-safe labels, grouping single-instance classes.

    Stratified splitting fails when a class has only one sample, so rare
    (count == 1) classes are merged into a placeholder label.
    """
    counts = series.value_counts()
    rare = set(counts.index[counts == 1])
    if not rare:
        return series, rare
    return series.map(lambda label: RARE_CLASS if label in rare else label), rare

DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"
MODELS_DIR = Path(__file__).resolve().parent.parent / "MODELS"
CSV_PATH = DATA_DIR / "cleaned_jobs.csv"

MODEL_PATH = MODELS_DIR / "job_classifier.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"


def find_column(df, *names):
    """Locate a column case-insensitively from candidate names."""
    lower_to_actual = {col.lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lower_to_actual:
            return lower_to_actual[name.lower()]
    return None


def load_dataset():
    """Load features and labels from the cleaned jobs dataset.

    Features are a combined text document per row:
    required_skills (doubled to weight its relevance for the resume-only
    inference path in app.py) + responsibilities + experience + education.
    """
    df = pd.read_csv(CSV_PATH)

    skills_col = find_column(df, "required_skills", "skills", "Job Requirement", "Key Skills")
    title_col = find_column(df, "Job Title", "job_title", "title", "Role")
    resp_col = find_column(df, "responsibilities", "responsibility")
    exp_col = find_column(df, "experience_required", "experience")
    edu_col = find_column(df, "education_required", "education")

    if skills_col is None or title_col is None:
        raise ValueError(
            f"Could not locate feature/label columns. Available: {list(df.columns)}"
        )

    data = df.copy()
    for col in [skills_col, resp_col, exp_col, edu_col]:
        if col is not None:
            data[col] = data[col].fillna("").astype(str)

    resp = data[resp_col] if resp_col else pd.Series("", index=data.index)
    exp = data[exp_col] if exp_col else pd.Series("", index=data.index)
    edu = data[edu_col] if edu_col else pd.Series("", index=data.index)

    # Same shared builder used by APP/app.py at inference time
    # (UTILS/features.build_job_document) so training features and prediction
    # features can never drift apart.
    combined = [
        build_job_document(title, skills, r, e, d)
        for title, skills, r, e, d in zip(
            data[title_col], data[skills_col], resp, exp, edu
        )
    ]

    return combined, data[title_col]


def main():
    X, y = load_dataset()
    print(f"Loaded {len(X)} samples from {CSV_PATH.name}")

    y_split, rare = stratify_labels(y)
    if rare:
        print(f"Grouped {len(rare)} single-instance class(es) for stratification")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_split, test_size=0.2, random_state=42, stratify=y_split
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    base_model = LogisticRegression(max_iter=1000, C=1.0)

    # CalibratedClassifierCV wraps the Logistic Regression so predict_proba()
    # outputs realistic, calibrated probability distributions across the
    # trained job title classes instead of raw LR scores.
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    model.fit(X_train_tfidf, y_train)

    X_test_tfidf = vectorizer.transform(X_test)
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()