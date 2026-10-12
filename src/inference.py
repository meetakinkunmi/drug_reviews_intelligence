"""STEP 7 - Serving logic shared by the FastAPI service and the Streamlit live predictor."""
from __future__ import annotations

from typing import Dict, List, Optional

from config import Config
from src.aspects import BENEFIT_LEXICON, SIDE_EFFECT_LEXICON, AspectExtractor
from src.model import SentimentClassifier
from src.preprocessing import TextCleaner
from src.sentiment import SentimentAnalyzer


class InferenceService:
    """Loads the trained model once and analyses new reviews."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.classifier = SentimentClassifier.load(self.config.model_path, self.config)
        self.sentiment = SentimentAnalyzer(self.config)
        self.side_effects = AspectExtractor(SIDE_EFFECT_LEXICON, negation_aware=True)
        self.benefits = AspectExtractor(BENEFIT_LEXICON)

    def analyze(self, benefits: str = "", side_effects: str = "", comments: str = "") -> Dict:
        combined = TextCleaner.combine(benefits, side_effects, comments)
        if not combined:
            raise ValueError("Provide at least one non-empty review field.")
        proba = self.classifier.predict_proba([combined])[0]
        classes = self.classifier.classes_
        compound: Optional[float] = self.sentiment.vader_compound(combined)
        return {
            "predicted_sentiment": classes[int(proba.argmax())],
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
            "vader_compound": None if compound != compound else round(float(compound), 4),
            "vader_label": self.sentiment.label_from_score(compound),
            "side_effects_detected": self.side_effects.extract(TextCleaner.light_clean(side_effects) or combined),
            "benefits_detected": self.benefits.extract(TextCleaner.light_clean(benefits) or combined),
        }

    def analyze_many(self, items: List[Dict]) -> List[Dict]:
        return [self.analyze(**it) for it in items]
