"""
Plotly charts for RAG Triad quality visualization.
"""
import plotly.graph_objects as go

_METRIC_LABELS = ["Faithfulness", "Answer Relevancy", "Context Precision"]
_METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision"]
_COLORS = ["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B", "#B279A2"]


def radar_chart(
    scores: list[dict],
    title: str = "RAG Triad Quality Radar",
) -> go.Figure:
    """
    Draw a spider/radar chart with one polygon per question.

    Each dict in `scores` must have:
      label, faithfulness, answer_relevancy, context_precision
    """
    categories = _METRIC_LABELS + [_METRIC_LABELS[0]]  # close the polygon

    fig = go.Figure()
    for i, item in enumerate(scores):
        values = [item[k] for k in _METRIC_KEYS] + [item[_METRIC_KEYS[0]]]
        color = _COLORS[i % len(_COLORS)]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                fillcolor=color,
                line=dict(color=color),
                opacity=0.5,
                name=item["label"][:45],
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".1f")
        ),
        showlegend=True,
        legend=dict(orientation="v", x=1.05),
        margin=dict(l=40, r=160, t=60, b=40),
    )
    return fig


def bar_chart(
    scores: list[dict],
    title: str = "Per-Question Metric Scores",
) -> go.Figure:
    """
    Grouped bar chart - one group per question, three bars per group.

    Each dict in `scores` must have:
      label, faithfulness, answer_relevancy, context_precision
    """
    labels = [item["label"][:35] for item in scores]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Faithfulness",
                x=labels,
                y=[item["faithfulness"] for item in scores],
                marker_color="#4C78A8",
            ),
            go.Bar(
                name="Answer Relevancy",
                x=labels,
                y=[item["answer_relevancy"] for item in scores],
                marker_color="#72B7B2",
            ),
            go.Bar(
                name="Context Precision",
                x=labels,
                y=[item["context_precision"] for item in scores],
                marker_color="#F58518",
            ),
        ]
    )

    fig.update_layout(
        barmode="group",
        title=dict(text=title, x=0.5),
        yaxis=dict(range=[0, 1.05], title="Score"),
        xaxis=dict(tickangle=-30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=100),
    )
    return fig


def average_radar(scores: list[dict], title: str = "Average RAG Triad Scores") -> go.Figure:
    """Single-polygon radar showing averaged scores across all questions."""
    if not scores:
        return go.Figure()
    avg = {k: sum(s[k] for s in scores) / len(scores) for k in _METRIC_KEYS}
    avg["label"] = f"Average (n={len(scores)})"
    return radar_chart([avg], title=title)
