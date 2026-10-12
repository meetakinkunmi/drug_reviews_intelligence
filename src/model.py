"""STEP 6 - Supervised sentiment classifier (TF-IDF + Logistic Regression).

Target = rating-based label (negative / neutral / positive). A simple, explainable model is the right
first model: you can read its coefficients to see which words drive each sentiment.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from config import Config

logger = logging.getLogger(__name__)


class SentimentClassifier:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self.metrics_: dict = {}

    @property
    def classes_(self):
        return list(self.pipeline.named_steps["clf"].classes_)

    def fit(self, df: pd.DataFrame, text_col: str = "combined_text",
            target_col: str = "sentiment_label") -> dict:
        X, y = df[text_col], df[target_col]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.config.test_size, stratify=y, random_state=self.config.random_state)

        baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        base_f1 = f1_score(y_te, baseline.predict(X_te), average="macro", zero_division=0)

        self.pipeline.fit(X_tr, y_tr)
        preds = self.pipeline.predict(X_te)
        labels = self.classes_
        cv = cross_val_score(self.pipeline, X, y, cv=5, scoring="f1_macro")

        self.metrics_ = {
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "accuracy": float(accuracy_score(y_te, preds)),
            "macro_f1": float(f1_score(y_te, preds, average="macro", zero_division=0)),
            "baseline_macro_f1": float(base_f1),
            "cv_macro_f1_mean": float(cv.mean()), "cv_macro_f1_std": float(cv.std()),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_te, preds, labels=labels).tolist(),
            "report": classification_report(y_te, preds, output_dict=True, zero_division=0),
        }
        logger.info("Model macro-F1 %.3f (baseline %.3f)", self.metrics_["macro_f1"], base_f1)
        # Refit on all data for the final deployed model
        self.pipeline.fit(X, y)
        return self.metrics_

    def predict(self, texts) -> np.ndarray:
        return self.pipeline.predict(list(texts))

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))

    def top_features(self, n: int = 15) -> pd.DataFrame:
        vec = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        names = vec.get_feature_names_out()
        rows = []
        for i, cls in enumerate(clf.classes_):
            for idx in clf.coef_[i].argsort()[::-1][:n]:
                rows.append({"class": cls, "term": names[idx], "weight": round(float(clf.coef_[i][idx]), 3)})
        return pd.DataFrame(rows)

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or self.config.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metrics": self.metrics_}, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None, config: Config | None = None) -> "SentimentClassifier":
        config = config or Config()
        obj = cls(config)
        payload = joblib.load(Path(path or config.model_path))
        obj.pipeline, obj.metrics_ = payload["pipeline"], payload.get("metrics", {})
        return obj
