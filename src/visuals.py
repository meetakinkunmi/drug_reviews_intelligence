"""STEP 8 - Plotly figure builders used by the dashboard (and handy in notebooks)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class Plotter:
    @staticmethod
    def rating_distribution(df: pd.DataFrame) -> go.Figure:
        counts = df["rating"].value_counts().sort_index().rename_axis("rating").reset_index(name="reviews")
        return px.bar(counts, x="rating", y="reviews", title="Rating distribution", text_auto=True)

    @staticmethod
    def bar(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> go.Figure:
        return px.bar(df, x=x, y=y, color=color, title=title, text_auto=".2s")

    @staticmethod
    def hbar(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> go.Figure:
        fig = px.bar(df, x=x, y=y, color=color, orientation="h", title=title,
                     color_continuous_scale="RdYlGn" if color else None)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return fig

    @staticmethod
    def heatmap(pivot: pd.DataFrame, title: str, x_label: str, y_label: str, color_label: str) -> go.Figure:
        fig = px.imshow(pivot, text_auto=".2f", aspect="auto", color_continuous_scale="RdYlGn",
                        labels={"x": x_label, "y": y_label, "color": color_label}, title=title)
        return fig

    @staticmethod
    def benefit_risk(drug_df: pd.DataFrame) -> go.Figure:
        return px.scatter(
            drug_df, x="mean_side_effect_score", y="mean_effectiveness_score", size="reviews",
            color="mean_rating", hover_name="drug", color_continuous_scale="RdYlGn",
            title="Benefit vs risk per drug (top-left = effective with few side effects)",
            labels={"mean_side_effect_score": "Avg side-effect severity (1-5)",
                    "mean_effectiveness_score": "Avg effectiveness (1-5)"})
