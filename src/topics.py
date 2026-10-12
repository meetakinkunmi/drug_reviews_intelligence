"""STEP 4 - Topic modelling (TF-IDF + NMF). Works well on small/medium text sets and needs no GPU."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from config import Config


class TopicModeler:
    """One topic model for one text column."""

    def __init__(self, n_topics: int = 6, random_state: int = 42) -> None:
        self.vectorizer = TfidfVectorizer(max_df=0.8, min_df=3, ngram_range=(1, 2), max_features=5000)
        self.model = NMF(n_components=n_topics, init="nndsvda", random_state=random_state, max_iter=400)

    def fit(self, texts: pd.Series) -> "TopicModeler":
        texts = texts[texts.str.strip() != ""]
        self.model.fit(self.vectorizer.fit_transform(texts))
        return self

    def terms(self, n_terms: int = 10) -> pd.DataFrame:
        names = self.vectorizer.get_feature_names_out()
        rows = [{"topic": i, "top_terms": ", ".join(names[c.argsort()[::-1][:n_terms]])}
                for i, c in enumerate(self.model.components_)]
        return pd.DataFrame(rows)

    def labels(self, n_terms: int = 3) -> dict:
        names = self.vectorizer.get_feature_names_out()
        return {i: f"T{i}: " + ", ".join(names[c.argsort()[::-1][:n_terms]])
                for i, c in enumerate(self.model.components_)}

    def assign(self, texts: pd.Series) -> np.ndarray:
        weights = self.model.transform(self.vectorizer.transform(texts))
        ids = weights.argmax(axis=1)
        return np.where(weights.sum(axis=1) == 0, -1, ids)


class TopicAnalyzer:
    """Fits a TopicModeler for each review column (benefits, side effects, comments)."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.models: dict[str, TopicModeler] = {}

    def fit_annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.config.text_cols:
            clean = df[f"{col}_clean"]
            modeler = TopicModeler(self.config.n_topics, self.config.random_state).fit(clean)
            labels = modeler.labels()
            df[f"{col}_topic"] = [labels.get(i, "no topic") for i in modeler.assign(clean)]
            self.models[col] = modeler
        return df

    def tables(self) -> dict[str, pd.DataFrame]:
        return {f"topics_{col}": m.terms() for col, m in self.models.items()}
