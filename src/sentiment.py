"""STEP 2 - Sentiment analysis (VADER baseline + optional transformer)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import Config

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Scores review text and compares it with the rating-based label."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._vader = SentimentIntensityAnalyzer()
        self._hf = None  # lazy-loaded transformer

    # ---- VADER ----
    def vader_compound(self, text) -> float:
        if not isinstance(text, str) or not text.strip():
            return np.nan
        return self._vader.polarity_scores(text)["compound"]

    @staticmethod
    def label_from_score(score: float, threshold: float = 0.05) -> str:
        if pd.isna(score):
            return "neutral"
        if score >= threshold:
            return "positive"
        if score <= -threshold:
            return "negative"
        return "neutral"

    def add_vader(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.config.text_cols:
            df[f"{col}_vader"] = df[col].apply(self.vader_compound)
        df["combined_vader"] = df["combined_text"].apply(self.vader_compound)
        df["text_sentiment"] = df["combined_vader"].apply(self.label_from_score)
        # Disagreement between what the person rated and what they wrote (sarcasm, mis-clicks, VADER errors)
        df["sentiment_mismatch"] = (
            ((df["sentiment_label"] == "positive") & (df["text_sentiment"] == "negative"))
            | ((df["sentiment_label"] == "negative") & (df["text_sentiment"] == "positive"))
        )
        return df

    # ---- optional transformer ----
    def add_transformer(self, df: pd.DataFrame, text_col: str = "combined_text",
                        batch_size: int = 32) -> pd.DataFrame:
        """Needs `pip install transformers torch`. Slow on CPU; run it once and cache the CSV."""
        from transformers import pipeline  # imported lazily so the base install stays light

        if self._hf is None:
            logger.info("Loading transformer model %s", self.config.transformer_model)
            self._hf = pipeline("sentiment-analysis", model=self.config.transformer_model)
        preds = self._hf(df[text_col].tolist(), batch_size=batch_size, truncation=True, max_length=256)
        df = df.copy()
        df["transformer_label"] = [p["label"].lower() for p in preds]
        df["transformer_score"] = [p["score"] for p in preds]
        return df

    # ---- diagnostics ----
    @staticmethod
    def agreement_table(df: pd.DataFrame) -> pd.DataFrame:
        """Row-normalised crosstab: rating-based label vs VADER label."""
        return pd.crosstab(df["sentiment_label"], df["text_sentiment"], normalize="index").round(3)
