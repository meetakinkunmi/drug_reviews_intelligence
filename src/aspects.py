"""STEP 3 - Aspect extraction: which side effects / benefits are mentioned?

Lexicons are STARTING POINTS. After reading the topic-model output and some sample reviews,
add or remove patterns. Each value is a list of regex fragments.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

import pandas as pd

SEP = "; "

SIDE_EFFECT_LEXICON: Dict[str, List[str]] = {
    "nausea": [r"nause\w*", r"queasy", r"sick to my stomach"],
    "vomiting": [r"vomit\w*", r"throw(?:ing)? up", r"threw up"],
    "headache": [r"headache\w*", r"migraine\w*"],
    "dizziness": [r"dizz\w*", r"light-?headed\w*", r"vertigo"],
    "fatigue": [r"fatigue\w*", r"tired\w*", r"exhaust\w*", r"letharg\w*", r"no energy"],
    "drowsiness": [r"drows\w*", r"sleepy", r"sedat\w*", r"groggy"],
    "insomnia": [r"insomnia", r"trouble sleeping", r"can'?t sleep", r"sleepless\w*"],
    "weight_gain": [r"weight gain", r"gain(?:ed)? (?:a lot of )?weight", r"put on weight", r"gained \d+ ?(?:lb|pound)\w*"],
    "weight_loss": [r"weight loss", r"lost weight", r"losing weight"],
    "appetite_change": [r"(?:loss|lack|decrease|decreased|increase|increased) (?:of )?appetite", r"no appetite"],
    "dry_mouth": [r"dry mouth", r"cottonmouth"],
    "dry_skin": [r"dry skin", r"dry lips", r"chapped lips", r"peeling", r"dryness"],
    "rash_itching": [r"rash\w*", r"hives", r"itch\w*"],
    "diarrhea": [r"diarrh\w*", r"loose stool\w*"],
    "constipation": [r"constipat\w*"],
    "stomach_pain": [r"stomach (?:pain|upset|ache|cramps?)", r"abdominal pain", r"heartburn", r"indigestion", r"bloat\w*", r"gastric"],
    "anxiety": [r"anxiety", r"anxious", r"jitter\w*", r"panic", r"nervous\w*", r"agitat\w*"],
    "mood_changes": [r"depress\w*", r"mood swings?", r"irritab\w*", r"emotional"],
    "sexual_dysfunction": [r"libido", r"sexual\w*", r"erectile", r"impoten\w*", r"orgasm\w*"],
    "muscle_joint_pain": [r"muscle (?:pain|ache|cramp|weakness)s?", r"joint pain", r"body ache\w*", r"cramp\w*"],
    "hair_loss": [r"hair loss", r"hair thinning", r"losing hair"],
    "sweating_hot_flashes": [r"sweat\w*", r"hot flash\w*", r"hot flush\w*"],
    "vision_problems": [r"blurr?e?d? vision", r"blurry", r"vision problems?"],
    "cough": [r"cough\w*"],
    "swelling": [r"swell\w*", r"swollen", r"edema"],
    "heart_palpitations": [r"palpitation\w*", r"racing heart", r"rapid heart ?beat", r"chest pain"],
    "numbness_tingling": [r"numb\w*", r"tingling"],
    "tremor": [r"tremor\w*", r"shaky", r"shaking"],
}

BENEFIT_LEXICON: Dict[str, List[str]] = {
    "pain_relief": [r"pain\w*", r"ache\w*", r"sore\w*", r"relie\w*"],
    "mood_improvement": [r"mood", r"happier", r"calmer", r"calm", r"stabiliz\w*", r"feel(?:ing)? better"],
    "anxiety_relief": [r"anxiety", r"anxious", r"panic", r"worry\w*"],
    "sleep_improvement": [r"sleep\w*", r"insomnia", r"slept"],
    "energy_focus": [r"energ\w*", r"alert\w*", r"focus\w*", r"concentrat\w*"],
    "blood_pressure_control": [r"blood pressure", r"\bbp\b", r"hypertension"],
    "cholesterol_control": [r"cholesterol", r"\bldl\b", r"triglyceride\w*"],
    "skin_clearing": [r"acne", r"skin", r"clear\w*", r"breakout\w*", r"pimple\w*"],
    "weight_loss": [r"weight loss", r"lost weight", r"losing weight", r"lose weight", r"lost \d+ ?(?:lb|pound)\w*"],
    "symptom_control": [r"symptom\w*", r"control\w*", r"manage\w*"],
    "infection_cleared": [r"infection\w*", r"cleared up", r"cured?", r"heal\w*"],
    "allergy_relief": [r"allerg\w*", r"sneez\w*", r"congest\w*", r"runny nose"],
    "digestive_relief": [r"stomach", r"acid", r"reflux", r"heartburn", r"digest\w*", r"\bgerd\b"],
    "headache_relief": [r"migraine\w*", r"headache\w*"],
}


class AspectExtractor:
    """Finds which aspects (from a lexicon) are mentioned in a text.

    negation_aware=True ignores matches like 'no nausea' / 'did not have headaches'
    (look-back of 3 words inside the same clause) - important for side effects.
    """

    NEGATIONS = {"no", "not", "never", "without", "none", "didn't", "don't", "doesn't",
                 "haven't", "hasn't", "zero", "nothing", "free"}
    _BOUNDARY = re.compile(r"[.,;:!?]|\bbut\b|\bhowever\b|\balthough\b")

    def __init__(self, lexicon: Dict[str, List[str]], negation_aware: bool = False) -> None:
        self.negation_aware = negation_aware
        self.patterns = {
            aspect: re.compile(r"\b(?:" + "|".join(frags) + r")\b", re.IGNORECASE)
            for aspect, frags in lexicon.items()
        }

    def _is_negated(self, text: str, start: int) -> bool:
        clause = self._BOUNDARY.split(text[:start])[-1]
        return any(tok in self.NEGATIONS for tok in clause.split()[-3:])

    def extract(self, text) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        text = text.lower().replace("\u2019", "'")
        found = []
        for aspect, pattern in self.patterns.items():
            for m in pattern.finditer(text):
                if self.negation_aware and self._is_negated(text, m.start()):
                    continue
                found.append(aspect)
                break
        return found

    def transform(self, texts: Iterable) -> pd.Series:
        """Returns a Series of 'aspect1; aspect2' strings (CSV-friendly)."""
        return pd.Series([SEP.join(self.extract(t)) for t in texts],
                         index=getattr(texts, "index", None))

    # ---- summaries ----
    @staticmethod
    def explode(df: pd.DataFrame, aspect_col: str) -> pd.DataFrame:
        tmp = df.copy()
        tmp[aspect_col] = tmp[aspect_col].fillna("").astype(str).str.split(SEP)
        tmp = tmp.explode(aspect_col)
        return tmp[tmp[aspect_col].str.strip() != ""]

    @classmethod
    def summarise(cls, df: pd.DataFrame, aspect_col: str) -> pd.DataFrame:
        exploded = cls.explode(df, aspect_col)
        if exploded.empty:
            return pd.DataFrame(columns=["aspect", "reviews", "share_pct", "mean_rating", "mean_sentiment"])
        out = (exploded.groupby(aspect_col)
               .agg(reviews=("rating", "size"), mean_rating=("rating", "mean"),
                    mean_sentiment=("combined_vader", "mean"))
               .reset_index().rename(columns={aspect_col: "aspect"}))
        out["share_pct"] = out["reviews"] / max(len(df), 1) * 100
        return out.sort_values("reviews", ascending=False).round(2)[
            ["aspect", "reviews", "share_pct", "mean_rating", "mean_sentiment"]]
