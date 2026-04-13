"""Chart helpers for the analytics dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt


def make_violation_figure(distribution: dict[str, int]):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(distribution.keys()) or ["No violations"]
    values = list(distribution.values()) or [1]
    ax.bar(labels, values, color=["#b91c1c", "#f59e0b", "#22c55e", "#2563eb"][: len(labels)])
    ax.set_ylabel("Count")
    ax.set_title("Violation Type Distribution")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def make_trend_figure(series: list[tuple[str, float]], title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    if series:
        xs = list(range(len(series)))
        ys = [value for _, value in series]
        ax.plot(xs, ys, marker="o", linewidth=2, color="#1d4ed8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time Window Index")
    fig.tight_layout()
    return fig
