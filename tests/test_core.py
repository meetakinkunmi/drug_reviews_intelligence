from src.aspects import AspectExtractor, SIDE_EFFECT_LEXICON
from src.preprocessing import TextCleaner


def test_light_clean_removes_html_and_urls():
    assert TextCleaner.light_clean("Great <b>drug</b> see http://x.com  now") == "Great drug see now"


def test_light_clean_handles_non_string():
    assert TextCleaner.light_clean(None) == ""


def test_combine_skips_empty_parts():
    assert TextCleaner.combine("good", "", "bad") == "good . bad"


def test_topic_clean_drops_stopwords_and_symbols():
    assert TextCleaner.topic_clean("The Pain was GONE!!!") == "pain gone"


def test_negation_is_respected():
    ex = AspectExtractor(SIDE_EFFECT_LEXICON, negation_aware=True)
    found = ex.extract("No nausea but bad headaches")
    assert "headache" in found and "nausea" not in found


def test_aspects_without_negation_awareness():
    ex = AspectExtractor(SIDE_EFFECT_LEXICON, negation_aware=False)
    assert "nausea" in ex.extract("no nausea")
