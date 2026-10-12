"""STEP 1 - Cleaning and feature engineering."""
from __future__ import annotations

import html
import logging
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import Config

logger = logging.getLogger(__name__)


class TextCleaner:
    """Stateless text helpers shared by training, the API and the dashboard."""

    _URL = re.compile(r"https?://\S+|www\.\S+")
    _TAG = re.compile(r"<[^>]+>")
    _SPACE = re.compile(r"\s+")
    _NON_ALPHA = re.compile(r"[^a-z\s']")
    STOPWORDS = set(ENGLISH_STOP_WORDS) | {
        "drug", "medication", "medicine", "taking", "took", "take", "taken",
        "day", "days", "doctor", "mg", "just", "like", "got", "get", "really",
    }

    @classmethod
    def light_clean(cls, text) -> str:
        """Keeps case and punctuation (VADER uses them). Removes HTML/URLs/extra spaces."""
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)
        text = cls._TAG.sub(" ", text)
        text = cls._URL.sub(" ", text)
        return cls._SPACE.sub(" ", text).strip()

    @classmethod
    def topic_clean(cls, text) -> str:
        """Lowercase, letters only, stop-words removed. Used for topics / keywords."""
        text = cls.light_clean(text).lower()
        text = cls._NON_ALPHA.sub(" ", text)
        tokens = [t.strip("'") for t in text.split()]
        return " ".join(t for t in tokens if len(t) > 2 and t not in cls.STOPWORDS)

    @classmethod
    def combine(cls, *parts) -> str:
        cleaned = [cls.light_clean(p) for p in parts]
        return " . ".join(p for p in cleaned if p)


class DataPreprocessor:
    """Turns the raw DataFrame into an analysis-ready DataFrame."""

    _CONDITION_PREFIX = re.compile(
        r"^(?:the )?(?:management|treatment|prevention|relief|control) of (?:the )?"
    )

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    # ---- public ----
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._standardise(df)
        df = self._filter(df)
        df = self._engineer(df)
        logger.info("Preprocessing finished: %d rows kept", len(df))
        return df.reset_index(drop=True)

    # ---- internals ----
    def _normalise_condition(self, value) -> str:
        value = str(value).lower().strip() if pd.notna(value) else ""
        value = re.sub(r"[^a-z0-9\s,'/-]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        value = self._CONDITION_PREFIX.sub("", value)
        return value or "unknown"

    def _standardise(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        for col in cfg.text_cols:
            df[col] = df[col].apply(TextCleaner.light_clean)
        for col in ("effectiveness", "sideEffects"):
            df[col] = df[col].astype(str).str.strip()
        df["drug"] = df["urlDrugName"].astype(str).str.strip().str.lower()
        df["condition"] = df["condition"].apply(self._normalise_condition)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        return df

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        n0 = len(df)
        df = df.dropna(subset=["rating"])
        df = df[df["drug"] != ""]
        df = df.drop_duplicates(
            subset=["drug", "rating", "effectiveness", "sideEffects", "condition", *cfg.text_cols]
        )
        words = df[cfg.text_cols].apply(lambda r: sum(len(str(v).split()) for v in r), axis=1)
        df = df[words >= cfg.min_words]
        logger.info("Dropped %d rows (missing rating, duplicates, near-empty text)", n0 - len(df))
        return df

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df["effectiveness_score"] = df["effectiveness"].map(cfg.effectiveness_order)
        df["side_effects_score"] = df["sideEffects"].map(cfg.side_effect_order)
        for col, score in (("effectiveness", "effectiveness_score"), ("sideEffects", "side_effects_score")):
            n_bad = int(df[score].isna().sum())
            if n_bad:
                logger.warning("%d rows have unrecognised '%s' categories: %s",
                               n_bad, col, df.loc[df[score].isna(), col].unique()[:5])

        df["combined_text"] = df.apply(
            lambda r: TextCleaner.combine(*[r[c] for c in cfg.text_cols]), axis=1
        )
        for col in cfg.text_cols:
            df[f"{col}_clean"] = df[col].apply(TextCleaner.topic_clean)
        df["combined_clean"] = df["combined_text"].apply(TextCleaner.topic_clean)
        df["word_count"] = df["combined_text"].str.split().str.len()

        df["sentiment_label"] = np.select(
            [df["rating"] <= cfg.negative_max, df["rating"] >= cfg.positive_min],
            ["negative", "positive"],
            default="neutral",
        )
        return df
