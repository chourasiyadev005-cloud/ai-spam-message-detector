from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "spam.csv"
MODEL_PATH = ROOT / "spam_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"
REPORT_PATH = ROOT / "evaluation_report.txt"

def main():
    df = pd.read_csv(DATA_PATH)
    required = {"Category", "Message"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}, found {set(df.columns)}")

    df = df[["Category", "Message"]].dropna().copy()
    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
    df["Message"] = df["Message"].astype(str).str.strip()
    df = df[df["Category"].isin(["ham", "spam"]) & (df["Message"] != "")]
    # Exact duplicate messages can otherwise leak across train/test and inflate results.
    df = df.drop_duplicates(subset=["Message"], keep="first").reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["Message"], df["Category"],
        test_size=0.20, random_state=42, stratify=df["Category"]
    )

    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced")
    }
    results = {}
    fitted = {}

    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode",
                ngram_range=(1, 2), min_df=2, max_features=60000,
                sublinear_tf=True
            )),
            ("classifier", classifier)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_spam": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
            "recall_spam": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
            "f1_spam": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0))
        }
        fitted[name] = (pipeline, predictions)

    best_name = max(results, key=lambda name: results[name]["f1_spam"])
    best_model, best_predictions = fitted[best_name]
    joblib.dump(best_model, MODEL_PATH)

    metrics = {
        "dataset_name": "SMS Spam Collection (provided as spam.csv)",
        "task_note": "This is an SMS dataset, not a dedicated email dataset.",
        "rows_after_cleaning_and_deduplication": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {str(k): int(v) for k, v in df["Category"].value_counts().items()},
        "best_model": best_name,
        "models": results,
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": confusion_matrix(y_test, best_predictions, labels=["ham", "spam"]).tolist(),
        "classification_report": classification_report(y_test, best_predictions, labels=["ham", "spam"], zero_division=0, output_dict=True)
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = [
        "AI Spam Message Detector - Evaluation Report",
        "=" * 48,
        f"Dataset rows after cleaning/deduplication: {len(df)}",
        f"Training rows: {len(X_train)}",
        f"Test rows: {len(X_test)}",
        f"Selected model (highest spam F1): {best_name}",
        "",
        "Model comparison:"
    ]
    for name, values in results.items():
        report.append(
            f"{name}: accuracy={values['accuracy']:.4f}, "
            f"spam_precision={values['precision_spam']:.4f}, "
            f"spam_recall={values['recall_spam']:.4f}, "
            f"spam_f1={values['f1_spam']:.4f}"
        )
    report += ["", "Classification report:", classification_report(y_test, best_predictions, zero_division=0)]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved model: {MODEL_PATH.name}")
    print(f"Saved metrics: {METRICS_PATH.name}")

if __name__ == "__main__":
    main()
