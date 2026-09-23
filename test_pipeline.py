import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from UTILS.extractor import extract_skills_from_text

MODEL_PATH = PROJECT_ROOT / "MODELS" / "job_classifier.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "MODELS" / "tfidf_vectorizer.pkl"

SAMPLE_RESUME_TEXT = (
    "Alex Morgan\n"
    "Data Science Engineer\n"
    "Skills: Python, SQL, Machine Learning, Pandas, NumPy\n"
    "Education: MS in Data Science, 2019\n"
    "Experience: Building ML pipelines for 3 years\n"
)


def main():
    # Step 1: skill extraction (same code path as APP/app.py).
    skills = extract_skills_from_text(SAMPLE_RESUME_TEXT)
    print("--- Extracted Skills ---")
    print(skills)

    if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists()):
        print("Model artifacts missing. Run `python UTILS/train_model.py` first.")
        return

    # Step 2: the active ML prediction pipeline used by APP/app.py.
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    feature_vector = vectorizer.transform([" ".join(skills)])
    probabilities = model.predict_proba(feature_vector)[0]
    class_names = model.classes_

    ranked = sorted(
        range(len(class_names)),
        key=lambda i: probabilities[i],
        reverse=True,
    )[:10]
    total = sum(probabilities[index] for index in ranked) or 1.0

    print("\n--- Top 10 Predicted Jobs (ML Pipeline) ---")
    for index in ranked:
        pct = probabilities[index] / total * 100
        print(f"{class_names[index]:55s} {pct:5.1f}%")


if __name__ == "__main__":
    main()