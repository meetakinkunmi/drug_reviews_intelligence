# Drug Review Intelligence (UCI Drug Review)

An end-to-end NLP project that turns patient drug reviews into insight: sentiment, side effects, benefits, topics and effectiveness patterns, served through a REST API and an interactive dashboard.

> **Disclaimer:** This project analyses self-reported patient reviews for educational and analytical purposes. It is **not** medical advice and must not be used to make treatment decisions.

---

## Table of contents

1. [Problem statement](#problem-statement)
2. [Case study: drug reviews](#case-study-drug-reviews)
3. [Data](#data)
4. [Objectives](#objectives)
5. [Architecture](#architecture)
6. [Project structure](#project-structure)
7. [Setup](#setup)
8. [Run locally](#run-locally)
9. [Deploy with Docker](#deploy-with-docker)
10. [API reference](#api-reference)
11. [Results](#results)
12. [Tips for sentiment analysis](#tips-for-sentiment-analysis)
13. [Limitations](#limitations)
14. [Roadmap](#roadmap)
15. [Testing](#testing)
16. [Acknowledgements](#acknowledgements)

---

## Problem statement

Organisations in every sector collect large volumes of free-text feedback: product reviews, support tickets, survey answers, social media posts. This text holds valuable signals about what people liked, what went wrong and why, but it is unstructured. Nobody can read thousands of comments by hand, and star ratings alone hide the detail. A rating of 7 does not say *what* worked or *what* the person suffered from.

The general problem is: **how do we systematically turn unstructured opinion text into structured, trustworthy, explorable insight?** A practical solution needs to:

- clean and standardise messy text,
- measure sentiment and check it against any structured score,
- identify *what* people talk about (aspects and themes),
- summarise findings for decision-makers,
- and make the whole process reusable through an API and a dashboard.

## Case study: drug reviews

Healthcare is a high-value example. Patients describe on public review sites how well a medication worked, what side effects they experienced and what condition they were treating. This is real-world experience that clinical trials do not fully capture, but it is scattered across thousands of free-text reviews.

This project narrows the general problem to **patient reviews of prescription drugs**. Each review has a numeric rating (1 to 10), categorical labels for effectiveness and side-effect severity, the medical condition, and three free-text fields (benefits, side effects, comments). The project answers questions such as:

- How effective do patients say their medication is, and does that line up with their rating?
- Which side effects are mentioned most, and which ones go with unhappy patients?
- Which conditions and drugs are reviewed most, and how are they rated?
- Do people's words agree with their drug ratings?
- What themes recur across the benefits, side effects and comments?

## Data

| Item | Detail |
|---|---|
| Dataset | UCI **Drug Review Dataset (Drugs.com)** |
| Format | A single `.csv` file placed in `data/raw/` |
| Size | Over 4,000 reviews |
| Granularity | One row per patient review |

**Required columns** (exact names, case-sensitive):

| Column | Description |
|---|---|
| `urlDrugName` | Name of the drug |
| `rating` | Patient rating from 1 to 10 |
| `effectiveness` | One of: Ineffective, Marginally Effective, Moderately Effective, Considerably Effective, Highly Effective |
| `sideEffects` | One of: No Side Effects, Mild Side Effects, Moderate Side Effects, Severe Side Effects, Extremely Severe Side Effects |
| `condition` | Medical condition being treated |
| `benefitsReview` | Free text: perceived benefits |
| `sideEffectsReview` | Free text: side effects experienced |
| `commentsReview` | Free text: general comments |

**Data handling notes**

- Download the dataset from the UCI Machine Learning Repository and place the file in `data/raw/`. The data is not redistributed with this repository.
- The original UCI release ships as two `.tsv` files (train and test). This project is configured for a single comma-separated `.csv`. If you are using the original TSV files, change `glob("*.csv")` to `glob("*.tsv")` and `sep=","` to `sep="\t"` in `src/data_loader.py`.
- The dataset has **no review date**, so "trends" in this project are patterns across groups (drugs, conditions, severity levels), not changes over time.

## Objectives

1. **Key effectiveness:** summarise how patients rate effectiveness and how it relates to their overall rating.
2. **Key side effects:** summarise severity categories and detect the specific side effects described in the text, ignoring negated mentions such as "no nausea".
3. **Key conditions:** identify the most-reviewed conditions and how each is rated.
4. **Key benefits and aspects:** extract the benefits patients mention and the themes behind their reviews.
5. **Patterns across groups:** compare drugs with a benefit-risk index, a heatmap of effectiveness against side-effect severity, and correlations between variables.

Plus a supporting objective: **train an explainable sentiment classifier** and expose it for new reviews.

## Architecture

```
                      data/raw/*.csv
                            |
                            v
  +---------------------------------------------------------+
  |                  DrugReviewPipeline (main.py)           |
  |                                                         |
  |  0 DataLoader      read + validate columns              |
  |  1 Preprocessing   clean text, encode labels, filter    |
  |  2 Sentiment       VADER (+ optional transformer)       |
  |  3 Aspects         side-effect / benefit lexicons       |
  |  4 Topics          TF-IDF + NMF                         |
  |  5 Insights        11 summary tables                    |
  |  6 Model           TF-IDF + Logistic Regression         |
  +---------------------------------------------------------+
                |                              |
                v                              v
        outputs/ (CSV, JSON)         models/sentiment_model.joblib
                |                              |
                +--------------+---------------+
                               |
                 +-------------+--------------+
                 |                            |
        7 InferenceService (shared logic for new reviews)
                 |                            |
                 v                            v
        FastAPI  (api/app.py)      Streamlit  (dashboard/app.py)
          port 8000                    port 8501
```

**Design principles**

- **One configuration file.** Paths, columns, label encodings, thresholds and model settings live in `config.py`.
- **One class per stage.** Each step is a small, testable class in `src/`.
- **Shared inference logic.** The API and the dashboard both call `InferenceService`, so they always give identical answers, and new reviews are cleaned exactly like the training data.
- **Artifacts on disk.** The pipeline saves its results, so the API and dashboard start quickly without re-training.

| Stage | File | Technique |
|---|---|---|
| Load | `src/data_loader.py` | pandas, encoding fallback, column validation |
| Clean | `src/preprocessing.py` | regex cleaning, stop-word removal, ordinal encoding |
| Sentiment | `src/sentiment.py` | VADER lexicon, optional Hugging Face transformer |
| Aspects | `src/aspects.py` | regex lexicons with negation handling |
| Topics | `src/topics.py` | TF-IDF + NMF |
| Insights | `src/insights.py` | groupby summaries, pivots, Spearman correlation |
| Model | `src/model.py` | TF-IDF + Logistic Regression, cross-validation |
| Serving | `src/inference.py` | shared prediction service |
| Charts | `src/visuals.py` | Plotly |
| Orchestration | `src/pipeline.py`, `main.py` | pipeline runner and CLI |

## Project structure

```
.
├── config.py                 # central settings
├── main.py                   # run the full pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── conftest.py               # lets pytest find the project root
├── api/
│   └── app.py                # FastAPI service
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── aspects.py
│   ├── topics.py
│   ├── insights.py
│   ├── model.py
│   ├── inference.py
│   ├── visuals.py
│   └── pipeline.py
├── tests/
│   └── test_core.py
├── data/raw/                 # put your CSV here (not committed)
├── outputs/                  # generated tables and processed data
│   └── tables/
└── models/                   # generated trained model
```

## Setup
1. Download the **Drug Review Dataset** from the UCI repository; convert it to `.csv` file put it in `data/raw/`.
2. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python main.py`
5. `pytest`

## Run locally
1. **Start the Dashboard:**
   ```powershell
   streamlit run dashboard/app.py
   ```

2. **Start the API:**
   ```powershell
   uvicorn api.app:app --reload
   ```
   *The API documentation will be available at: http://localhost:8000/docs*

3. **Test the API Endpoint:**
   Open a separate PowerShell window and run the following command:
   ```powershell
   curl -X POST localhost:8000/predict -H "Content-Type: application/json" `
     -d '{\"benefitsReview\":\"Pain is much better\",\"sideEffectsReview\":\"mild nausea\",\"commentsReview\":\"works well\"}'
   ```

## Deploy
- **Docker (all services):** `docker compose up --build` → dashboard :8501, API :8000 (trainer runs first).
- **Streamlit Community Cloud (free):** push repo to GitHub, point the app to `dashboard/app.py`. Commit `data/raw/*.tsv` (or a sample); the dashboard runs the pipeline on first start.

**Notes for real deployments**

- The trainer re-runs the pipeline on every `docker compose up`. For production, train once and ship the saved model.
- The API has **no authentication or rate limiting**. Put it behind a reverse proxy or API gateway, add an API key or OAuth, and restrict CORS before exposing it to the internet.
- Add a `.dockerignore` (for `.venv/`, `outputs/`, `data/`) to keep the image small.
- Pin library versions in `requirements.txt`. A `.joblib` model saved with one scikit-learn version may not load in another.
- Typical hosting options: a container platform (Cloud Run, Azure Container Apps, AWS ECS/App Runner), Streamlit Community Cloud for the dashboard only, or a small VM running `docker compose`.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check and whether the model is loaded |
| POST | `/predict` | Analyse one review |
| POST | `/predict/batch` | Analyse up to 500 reviews |

**Request body** (all fields optional, but at least one must be non-empty):

```json
{
  "benefitsReview": "Helped my pain a lot",
  "sideEffectsReview": "Some nausea and headaches",
  "commentsReview": ""
}
```

**Response:**

```json
{
  "predicted_sentiment": "positive",
  "probabilities": {"negative": 0.21, "neutral": 0.14, "positive": 0.65},
  "vader_compound": -0.12,
  "vader_label": "negative",
  "side_effects_detected": ["nausea", "headache"],
  "benefits_detected": ["pain_relief"]
}
```

## Results

Rating-based labels are derived from the 1 to 10 rating: **negative** (1 to 4), **neutral** (5 to 6), **positive** (7 to 10). The classifier is trained to predict them from the review text.

| Metric | Value |
|---|---|
| Accuracy (hold-out) | 0.744 |
| Macro F1 (hold-out) | 0.552 |
| Majority-class baseline macro F1 | 0.269 |
| 5-fold CV macro F1 | 0.543 ± 0.012 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.59 | 0.67 | 0.63 | 180 |
| neutral | 0.33 | 0.13 | 0.19 | 85 |
| positive | 0.82 | 0.86 | 0.84 | 554 |

**Takeaways**

- The model roughly **doubles the baseline** and is stable across folds.
- **Positive and negative are separated well; neutral is weak.** Ratings of 5 and 6 don't produce distinctive language, and neutral is only about 10% of the data.
- VADER and ratings disagree often: only about 48% of high-rated reviews have positive-sounding text, because patients describe side effects and symptoms even when a drug worked. About 1,550 reviews have a rating and text that point in opposite directions.

## Tips for sentiment analysis

Lessons from building this project that apply to most sentiment tasks:

1. **Know what your label really means.** Here the target is the *star rating*, not what the text expresses. Rating and text can disagree (sarcasm, mis-clicks, side-effect descriptions on a successful treatment), which caps achievable accuracy.
2. **Always compare against a baseline.** A dummy model that predicts the majority class scored 0.269 macro F1. Without that yardstick, 74% accuracy looks better than it is.
3. **Use macro F1, not just accuracy, on imbalanced data.** Accuracy hides the neutral class failing. Macro F1 treats every class equally.
4. **Clean differently for different tools.** Rule-based tools like VADER use capitals and punctuation ("GREAT!!!"), so keep them. For topic models and keyword counts, lowercase, strip punctuation and remove stop words.
5. **Remove domain stop words.** Words like "drug", "medication" and "take" appear everywhere and drown out real themes.
6. **Handle negation.** "No nausea" is not a side effect. Look back a few words inside the same clause, and stop at punctuation and words like "but".
7. **Split large texts by section.** Scoring benefits and side effects separately shows that the same review can be positive about results and negative about tolerability.
8. **Treat lexicons as starting points.** Read sample reviews and mismatches, then add and remove patterns. Over-broad stems (such as `numb\w*` matching "number") cause false hits.
9. **Start simple and explainable.** TF-IDF + Logistic Regression lets you read the coefficients and verify the model learned sensible words. Add complexity only if it earns its keep.
10. **Handle the neutral class deliberately.** Options: widen or redefine the neutral band, merge neutral into the other classes, or switch to two classes.
11. **Fit on training data only.** Keep the vectorizer inside a scikit-learn `Pipeline`, so the test set never leaks into the vocabulary. Use `stratify` when splitting, and fix `random_state` for repeatable results.
12. **Check cross-validation spread.** A small standard deviation (here 0.012) shows the score is not a lucky split.
13. **Try a pretrained transformer for context.** Models like RoBERTa understand phrases such as "not bad at all", but a model trained on tweets may transfer imperfectly to medical language. Evaluate it on your own data, and cache its output because it is slow on CPU.
14. **Read the errors.** The confusion matrix and the mismatched reviews teach more than any single score.
15. **Treat predicted probabilities with care.** Low-confidence predictions, and any "neutral" prediction from this model, deserve caution.

## Limitations

- Labels come from ratings, not from human annotation of the text.
- The neutral class is poorly predicted.
- Aspect extraction is keyword-based. It does not understand context beyond a short negation window, and benefit patterns detect *mentions*, not proven benefit.
- Reviews are self-reported, unverified and subject to selection bias.
- No dates, so no true time trends.
- No dosage, age, sex or comorbidity information, so results cannot be adjusted for those factors.
- The API has no authentication.
- Correlations and averages describe patterns in reviews. They do not show that a drug works.

## Roadmap

- Compare the optional transformer against the TF-IDF model on a hand-labelled sample
- Redefine or remove the neutral class and re-evaluate
- Replace keyword lexicons with a trained named-entity or aspect-based model
- Add API tests with FastAPI's `TestClient` and pipeline smoke tests
- Add authentication, rate limiting and CI (tests on every push)
- Pin dependency versions and add a `.dockerignore`

## Acknowledgements

This project utilizes the **UCI Drug Review Dataset**, which was originally provided by the UCI Machine Learning Repository. 

* **Source Repository:** UCI Machine Learning Repository - Drug Review Dataset
* Kallumadi, S. & Grer, F. (2018). Drug Reviews (Druglib.com) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55G6J.
- Libraries: pandas, scikit-learn, VADER Sentiment, Plotly, FastAPI, Streamlit, Docker.