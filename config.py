"""Central configuration. Every class reads its settings from here."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # ---- paths ----
    raw_dir: Path = ROOT / "data" / "raw"
    output_dir: Path = ROOT / "outputs"
    model_dir: Path = ROOT / "models"

    # ---- columns ----
    required_cols: List[str] = field(default_factory=lambda: [
        "urlDrugName", "rating", "effectiveness", "sideEffects",
        "condition", "benefitsReview", "sideEffectsReview", "commentsReview",
    ])
    text_cols: List[str] = field(default_factory=lambda: [
        "benefitsReview", "sideEffectsReview", "commentsReview",
    ])

    # ---- ordinal encodings (higher effectiveness = better; higher side effect = worse) ----
    effectiveness_order: Dict[str, int] = field(default_factory=lambda: {
        "Ineffective": 1, "Marginally Effective": 2, "Moderately Effective": 3,
        "Considerably Effective": 4, "Highly Effective": 5,
    })
    side_effect_order: Dict[str, int] = field(default_factory=lambda: {
        "No Side Effects": 1, "Mild Side Effects": 2, "Moderate Side Effects": 3,
        "Severe Side Effects": 4, "Extremely Severe Side Effects": 5,
    })

    # ---- rating -> sentiment label (ratings are 1-10) ----
    negative_max: int = 4      # rating <= 4  -> negative
    positive_min: int = 7      # rating >= 7  -> positive ; 5-6 -> neutral

    # ---- modelling ----
    n_topics: int = 6
    min_words: int = 3
    test_size: float = 0.2
    random_state: int = 42
    transformer_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __post_init__(self) -> None:
        for p in (self.raw_dir, self.output_dir, self.model_dir, self.tables_dir):
            Path(p).mkdir(parents=True, exist_ok=True)

    @property
    def tables_dir(self) -> Path:
        return Path(self.output_dir) / "tables"

    @property
    def processed_path(self) -> Path:
        return Path(self.output_dir) / "processed_reviews.csv"

    @property
    def metrics_path(self) -> Path:
        return Path(self.output_dir) / "model_metrics.json"

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir) / "sentiment_model.joblib"
