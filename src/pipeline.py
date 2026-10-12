"""STEP 9 - Orchestrator: runs every step in order and saves artifacts."""
from __future__ import annotations

import json
import logging

import pandas as pd

from config import Config
from src.aspects import BENEFIT_LEXICON, SIDE_EFFECT_LEXICON, AspectExtractor
from src.data_loader import DataLoader
from src.insights import InsightGenerator
from src.model import SentimentClassifier
from src.preprocessing import DataPreprocessor
from src.sentiment import SentimentAnalyzer
from src.topics import TopicAnalyzer

logger = logging.getLogger(__name__)


class DrugReviewPipeline:
    def __init__(self, config: Config | None = None, use_transformer: bool = False) -> None:
        self.config = config or Config()
        self.use_transformer = use_transformer

    def run(self) -> pd.DataFrame:
        cfg = self.config
        logger.info("1/7 Loading data");            df = DataLoader(cfg).load()
        logger.info("2/7 Preprocessing");           df = DataPreprocessor(cfg).run(df)
        logger.info("3/7 Sentiment scoring");       analyzer = SentimentAnalyzer(cfg); df = analyzer.add_vader(df)
        if self.use_transformer:
            df = analyzer.add_transformer(df)
        logger.info("4/7 Aspect extraction");       df = self._add_aspects(df)
        logger.info("5/7 Topic modelling");         topics = TopicAnalyzer(cfg); df = topics.fit_annotate(df)
        logger.info("6/7 Insight tables");          tables = InsightGenerator(df, cfg).all_tables()
        tables.update(topics.tables())
        logger.info("7/7 Training classifier");     clf = SentimentClassifier(cfg); metrics = clf.fit(df)
        tables["model_top_terms"] = clf.top_features()
        tables["rating_vs_text_sentiment"] = analyzer.agreement_table(df).reset_index()

        clf.save()
        self._save(df, tables, metrics)
        logger.info("Done. Artifacts in %s and %s", cfg.output_dir, cfg.model_dir)
        return df

    @staticmethod
    def _add_aspects(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["side_effect_aspects"] = AspectExtractor(SIDE_EFFECT_LEXICON, negation_aware=True).transform(df["sideEffectsReview"])
        df["benefit_aspects"] = AspectExtractor(BENEFIT_LEXICON).transform(df["benefitsReview"])
        return df

    def _save(self, df: pd.DataFrame, tables: dict, metrics: dict) -> None:
        cfg = self.config
        df.to_csv(cfg.processed_path, index=False)
        for name, table in tables.items():
            table.to_csv(cfg.tables_dir / f"{name}.csv", index=False)
        cfg.metrics_path.write_text(json.dumps(metrics, indent=2))

    # ---- loaders for the dashboard ----
    def artifacts_exist(self) -> bool:
        return self.config.processed_path.exists() and self.config.model_path.exists()

    @staticmethod
    def load_processed(config: Config | None = None) -> pd.DataFrame:
        config = config or Config()
        return pd.read_csv(config.processed_path, keep_default_na=False, na_values=[""],
                           dtype={"side_effect_aspects": str, "benefit_aspects": str})
