"""STEP 5 - Answers the five project objectives with tables (effectiveness, side effects,
conditions, benefits/side-effect aspects, trends). Works on any filtered DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from config import Config
from src.aspects import AspectExtractor


class InsightGenerator:
    def __init__(self, df: pd.DataFrame, config: Config | None = None) -> None:
        self.df = df
        self.config = config or Config()

    # ---- helpers ----
    def _group_summary(self, by: str) -> pd.DataFrame:
        g = (self.df.groupby(by)
             .agg(reviews=("rating", "size"),
                  mean_rating=("rating", "mean"),
                  mean_sentiment=("combined_vader", "mean"),
                  pct_positive=("sentiment_label", lambda s: (s == "positive").mean() * 100))
             .reset_index())
        g["share_pct"] = g["reviews"] / g["reviews"].sum() * 100
        return g

    # ---- Objective 1: key effectiveness ----
    def effectiveness_summary(self) -> pd.DataFrame:
        g = self._group_summary("effectiveness")
        g["order"] = g["effectiveness"].map(self.config.effectiveness_order)
        return g.sort_values("order", ascending=False).drop(columns="order").round(2)

    # ---- Objective 2: key side effects (severity categories) ----
    def side_effect_summary(self) -> pd.DataFrame:
        g = self._group_summary("sideEffects")
        g["order"] = g["sideEffects"].map(self.config.side_effect_order)
        return g.sort_values("order").drop(columns="order").round(2)

    # ---- Objective 3: key conditions ----
    def condition_summary(self, top_n: int = 20, min_reviews: int = 5) -> pd.DataFrame:
        g = self._group_summary("condition")
        g = g[g["reviews"] >= min_reviews]
        return g.sort_values("reviews", ascending=False).head(top_n).round(2)

    # ---- Objective 4: key benefits & side effects (free text) ----
    def aspect_summary(self, aspect_col: str) -> pd.DataFrame:
        return AspectExtractor.summarise(self.df, aspect_col)

    def top_phrases(self, ngram=(2, 2), top_k: int = 15, text_col: str = "combined_clean") -> pd.DataFrame:
        rows = []
        for label, sub in self.df.groupby("sentiment_label"):
            try:
                vec = CountVectorizer(ngram_range=ngram, min_df=3)
                counts = np.asarray(vec.fit_transform(sub[text_col]).sum(axis=0)).ravel()
            except ValueError:  # empty vocabulary
                continue
            names = vec.get_feature_names_out()
            for idx in counts.argsort()[::-1][:top_k]:
                rows.append({"sentiment": label, "phrase": names[idx], "count": int(counts[idx])})
        return pd.DataFrame(rows, columns=["sentiment", "phrase", "count"])

    # ---- Objective 5: trends (this dataset has NO date column, so trends are cross-sectional) ----
    def drug_summary(self, min_reviews: int = 10) -> pd.DataFrame:
        g = (self.df.groupby("drug")
             .agg(reviews=("rating", "size"),
                  mean_rating=("rating", "mean"),
                  mean_sentiment=("combined_vader", "mean"),
                  mean_effectiveness_score=("effectiveness_score", "mean"),
                  mean_side_effect_score=("side_effects_score", "mean"),
                  pct_positive=("sentiment_label", lambda s: (s == "positive").mean() * 100))
             .reset_index())
        g = g[g["reviews"] >= min_reviews].copy()
        # simple benefit-risk index: effectiveness minus side-effect burden (range -4 .. +4)
        g["benefit_risk_index"] = g["mean_effectiveness_score"] - g["mean_side_effect_score"]
        return g.sort_values("mean_rating", ascending=False).round(2)

    def effectiveness_by_side_effects(self) -> pd.DataFrame:
        pv = self.df.pivot_table(values="rating", index="effectiveness", columns="sideEffects", aggfunc="mean")
        rows = [k for k in sorted(self.config.effectiveness_order, key=self.config.effectiveness_order.get, reverse=True) if k in pv.index]
        cols = [k for k in sorted(self.config.side_effect_order, key=self.config.side_effect_order.get) if k in pv.columns]
        return pv.reindex(index=rows, columns=cols).round(2)

    def correlations(self) -> pd.DataFrame:
        cols = ["rating", "effectiveness_score", "side_effects_score", "combined_vader", "word_count"]
        return self.df[cols].corr(method="spearman").round(2)

    def sentiment_by_condition_drug(self, top_n: int = 15) -> pd.DataFrame:
        g = (self.df.groupby(["condition", "drug"])
             .agg(reviews=("rating", "size"), mean_rating=("rating", "mean"))
             .reset_index().sort_values("reviews", ascending=False).head(top_n))
        return g.round(2)

    # ---- export ----
    def all_tables(self) -> dict[str, pd.DataFrame]:
        corr = self.correlations().reset_index().rename(columns={"index": "variable"})
        return {
            "effectiveness_summary": self.effectiveness_summary(),
            "side_effect_summary": self.side_effect_summary(),
            "condition_summary": self.condition_summary(),
            "side_effect_aspects": self.aspect_summary("side_effect_aspects"),
            "benefit_aspects": self.aspect_summary("benefit_aspects"),
            "drug_summary": self.drug_summary(),
            "effectiveness_by_side_effects": self.effectiveness_by_side_effects().reset_index(),
            "correlations": corr,
            "top_bigrams_by_sentiment": self.top_phrases((2, 2)),
            "top_unigrams_by_sentiment": self.top_phrases((1, 1)),
            "condition_drug_pairs": self.sentiment_by_condition_drug(),
        }
