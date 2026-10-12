"""FastAPI service.  Run from the project root:  uvicorn api.app:app --reload"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import Config
from src.inference import InferenceService

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config()
    if not config.model_path.exists():
        raise RuntimeError("Model not found. Run `python main.py` first.")
    _state["service"] = InferenceService(config)
    yield
    _state.clear()


app = FastAPI(title="Drug Review Sentiment API", version="1.0.0", lifespan=lifespan)


class ReviewIn(BaseModel):
    benefitsReview: str = ""
    sideEffectsReview: str = ""
    commentsReview: str = ""


class PredictionOut(BaseModel):
    predicted_sentiment: str
    probabilities: Dict[str, float]
    vader_compound: Optional[float]
    vader_label: str
    side_effects_detected: List[str]
    benefits_detected: List[str]


def _analyze(review: ReviewIn) -> dict:
    try:
        return _state["service"].analyze(review.benefitsReview, review.sideEffectsReview, review.commentsReview)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "service" in _state}


@app.post("/predict", response_model=PredictionOut)
def predict(review: ReviewIn):
    return _analyze(review)


@app.post("/predict/batch", response_model=List[PredictionOut])
def predict_batch(reviews: List[ReviewIn]):
    if len(reviews) > 500:
        raise HTTPException(status_code=413, detail="Max 500 reviews per request.")
    return [_analyze(r) for r in reviews]
