"""Streamlit dashboard.  Run from the project root:  streamlit run dashboard/app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from config import Config
from src.aspects import AspectExtractor
from src.insights import InsightGenerator
from src.inference import InferenceService
from src.pipeline import DrugReviewPipeline
from src.visuals import Plotter

st.set_page_config(page_title="Drug Review Intelligence", page_icon="💊", layout="wide")
cfg = Config()


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


# ---------- data / model loading (cached) ----------
@st.cache_resource(show_spinner="First run: executing the analysis pipeline (a few minutes)...")
def ensure_artifacts() -> bool:
    pipe = DrugReviewPipeline(cfg)
    if not pipe.artifacts_exist():
        pipe.run()
    return True


@st.cache_data
def load_data() -> pd.DataFrame:
    return DrugReviewPipeline.load_processed(cfg)


@st.cache_data
def load_topics(col: str) -> pd.DataFrame:
    return pd.read_csv(cfg.tables_dir / f"topics_{col}.csv")


@st.cache_data
def load_metrics() -> dict:
    return json.loads(cfg.metrics_path.read_text())


@st.cache_resource
def load_service() -> InferenceService:
    return InferenceService(cfg)


try:
    ensure_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

df = load_data()

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
sel_drugs = st.sidebar.multiselect("Drug", sorted(df["drug"].unique()))
sel_conditions = st.sidebar.multiselect("Condition (top 50)", df["condition"].value_counts().head(50).index.tolist())
sel_eff = st.sidebar.multiselect("Effectiveness", list(cfg.effectiveness_order))
rating_range = st.sidebar.slider("Rating", 1, 10, (1, 10))

mask = df["rating"].between(*rating_range)
if sel_drugs:
    mask &= df["drug"].isin(sel_drugs)
if sel_conditions:
    mask &= df["condition"].isin(sel_conditions)
if sel_eff:
    mask &= df["effectiveness"].isin(sel_eff)
fdf = df[mask]
if fdf.empty:
    st.warning("No reviews match the current filters.")
    st.stop()
ins = InsightGenerator(fdf, cfg)

st.title("💊 Drug Review Intelligence Dashboard")
st.caption("Self-reported patient reviews. For analysis only - not medical advice.")

tabs = st.tabs(["Overview", "1 · Effectiveness", "2 · Side effects", "3 · Conditions",
                "4 · Benefits", "5 · Trends & topics", "Drug explorer", "Live predictor"])

# ---------- Overview ----------
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Reviews", f"{len(fdf):,}")
    c[1].metric("Drugs", f"{fdf['drug'].nunique():,}")
    c[2].metric("Conditions", f"{fdf['condition'].nunique():,}")
    c[3].metric("Avg rating", f"{fdf['rating'].mean():.2f}")
    c[4].metric("% positive", f"{(fdf['sentiment_label'] == 'positive').mean() * 100:.1f}%")
    left, right = st.columns(2)
    with left:
        show(Plotter.rating_distribution(fdf))
    with right:
        sent = fdf["sentiment_label"].value_counts().rename_axis("label").reset_index(name="reviews")
        show(Plotter.bar(sent, "label", "reviews", "Rating-based sentiment label"))
    st.subheader("Does the text agree with the rating?")
    st.write(f"{int(fdf['sentiment_mismatch'].sum()):,} reviews have a rating and text (VADER) that point in opposite directions.")
    with st.expander("Show mismatched reviews"):
        st.dataframe(fdf.loc[fdf["sentiment_mismatch"], ["drug", "rating", "combined_vader", "combined_text"]].head(100),
                     hide_index=True)

# ---------- Objective 1 ----------
with tabs[1]:
    eff = ins.effectiveness_summary()
    a, b = st.columns(2)
    with a:
        show(Plotter.bar(eff, "effectiveness", "reviews", "Reviews by effectiveness level"))
    with b:
        show(Plotter.bar(eff, "effectiveness", "mean_rating", "Average rating by effectiveness level"))
    st.dataframe(eff, hide_index=True)
    st.subheader("Most common phrases by sentiment")
    st.dataframe(ins.top_phrases((2, 2)), hide_index=True)

# ---------- Objective 2 ----------
with tabs[2]:
    se = ins.side_effect_summary()
    a, b = st.columns(2)
    with a:
        show(Plotter.bar(se, "sideEffects", "reviews", "Reviews by side-effect severity"))
    with b:
        show(Plotter.bar(se, "sideEffects", "mean_rating", "Average rating by side-effect severity"))
    st.subheader("Specific side effects mentioned in the text")
    sea = ins.aspect_summary("side_effect_aspects")
    show(Plotter.hbar(sea.head(20), "reviews", "aspect", "Top side effects mentioned", color="mean_rating"))
    st.dataframe(sea, hide_index=True)

# ---------- Objective 3 ----------
with tabs[3]:
    top_n = st.slider("Number of conditions", 5, 40, 20)
    cond = ins.condition_summary(top_n=top_n, min_reviews=1 if len(fdf) < 500 else 5)
    show(Plotter.hbar(cond, "reviews", "condition", "Most reviewed conditions", color="mean_rating"))
    st.dataframe(cond, hide_index=True)
    st.subheader("Most reviewed condition-drug pairs")
    st.dataframe(ins.sentiment_by_condition_drug(), hide_index=True)

# ---------- Objective 4 ----------
with tabs[4]:
    ba = ins.aspect_summary("benefit_aspects")
    show(Plotter.hbar(ba, "reviews", "aspect", "Benefits mentioned in the text", color="mean_rating"))
    st.dataframe(ba, hide_index=True)
    st.caption("Aspect lists come from keyword lexicons in src/aspects.py - refine them after reading the topics.")

# ---------- Objective 5 ----------
with tabs[5]:
    st.info("The UCI dataset has no review date, so 'trends' here are patterns across effectiveness, side effects, "
            "conditions and drugs rather than changes over time.")
    pv = ins.effectiveness_by_side_effects()
    show(Plotter.heatmap(pv, "Mean rating: effectiveness x side-effect severity",
                         "Side effects", "Effectiveness", "Mean rating"))
    min_rev = st.slider("Minimum reviews per drug", 3, 50, 10)
    drugs = ins.drug_summary(min_reviews=min_rev)
    if drugs.empty:
        st.write("No drugs meet that threshold.")
    else:
        show(Plotter.benefit_risk(drugs))
        a, b = st.columns(2)
        a.subheader("Best rated drugs"); a.dataframe(drugs.head(10), hide_index=True)
        b.subheader("Lowest rated drugs"); b.dataframe(drugs.tail(10).iloc[::-1], hide_index=True)
    corr = ins.correlations()
    show(Plotter.heatmap(corr, "Spearman correlations", "", "", "rho"))
    st.subheader("Discovered topics")
    for col in cfg.text_cols:
        with st.expander(col):
            st.dataframe(load_topics(col), hide_index=True)
            share = fdf[f"{col}_topic"].value_counts().rename_axis("topic").reset_index(name="reviews")
            show(Plotter.hbar(share, "reviews", "topic", f"Topic share - {col}"))
    with st.expander("Model performance"):
        m = load_metrics()
        st.write(f"Macro-F1 **{m['macro_f1']:.3f}** vs majority-class baseline {m['baseline_macro_f1']:.3f}; "
                 f"5-fold CV {m['cv_macro_f1_mean']:.3f} ± {m['cv_macro_f1_std']:.3f}; accuracy {m['accuracy']:.3f}")
        st.dataframe(pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"]))

# ---------- Drug explorer ----------
with tabs[6]:
    drug = st.selectbox("Choose a drug", sorted(fdf["drug"].unique()))
    sub = fdf[fdf["drug"] == drug]
    c = st.columns(4)
    c[0].metric("Reviews", len(sub))
    c[1].metric("Avg rating", f"{sub['rating'].mean():.2f}")
    c[2].metric("Avg effectiveness (1-5)", f"{sub['effectiveness_score'].mean():.2f}")
    c[3].metric("Avg side-effect severity (1-5)", f"{sub['side_effects_score'].mean():.2f}")
    st.write("**Top conditions:**", ", ".join(sub["condition"].value_counts().head(5).index))
    sub_ins = InsightGenerator(sub, cfg)
    a, b = st.columns(2)
    a.write("Side effects mentioned"); a.dataframe(sub_ins.aspect_summary("side_effect_aspects").head(10), hide_index=True)
    b.write("Benefits mentioned"); b.dataframe(sub_ins.aspect_summary("benefit_aspects").head(10), hide_index=True)
    st.dataframe(sub[["rating", "effectiveness", "sideEffects", "condition", "combined_text"]].head(50), hide_index=True)

# ---------- Live predictor ----------
with tabs[7]:
    st.write("Paste a new review and the trained model will analyse it.")
    benefits = st.text_area("Benefits review")
    side = st.text_area("Side effects review")
    comments = st.text_area("Comments")
    if st.button("Analyse"):
        try:
            res = load_service().analyze(benefits, side, comments)
            st.success(f"Predicted sentiment: **{res['predicted_sentiment']}**")
            st.json(res)
        except ValueError as exc:
            st.warning(str(exc))
