"""Chart helpers for the analytics dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt

# Modern color palette
COLOR_SAFE = "#10b981"
COLOR_WARNING = "#f59e0b"
COLOR_DANGER = "#ef4444"
COLOR_PRIMARY = "#3b82f6"
COLOR_SECONDARY = "#8b5cf6"
BG_COLOR = "#0f172a"
TEXT_COLOR = "#f1f5f9"


def _setup_plot_style():
    """Configure matplotlib for dark theme."""
    plt.rcParams["figure.facecolor"] = "#1e293b"
    plt.rcParams["axes.facecolor"] = "#0f172a"
    plt.rcParams["text.color"] = TEXT_COLOR
    plt.rcParams["axes.labelcolor"] = "#cbd5e1"
    plt.rcParams["xtick.color"] = "#cbd5e1"
    plt.rcParams["ytick.color"] = "#cbd5e1"
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["font.family"] = "sans-serif"


def make_violation_figure(distribution: dict[str, int]):
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    
    labels = list(distribution.keys()) or ["No violations"]
    values = list(distribution.values()) or [1]
    
    colors_list = [COLOR_DANGER, COLOR_WARNING, COLOR_PRIMARY, COLOR_SECONDARY][:len(labels)]
    
    bars = ax.bar(labels, values, color=colors_list, edgecolor="#334155", linewidth=1.5, alpha=0.85)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontweight="bold",
            fontsize=11,
        )
    
    ax.set_ylabel("Count", fontsize=12, fontweight="bold", color="#cbd5e1")
    ax.set_title("Violation Type Distribution", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=20)
    ax.tick_params(axis="x", rotation=15, labelsize=10)
    ax.grid(axis="y", alpha=0.1, linestyle="--", color="#cbd5e1")
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    return fig


def make_trend_figure(series: list[tuple[str, float]], title: str, ylabel: str):
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    
    if series:
        xs = list(range(len(series)))
        ys = [value for _, value in series]
        
        # Create gradient effect with filled area under line
        ax.fill_between(
            xs,
            ys,
            alpha=0.2,
            color=COLOR_PRIMARY,
        )
        
        # Main line
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=3,
            color=COLOR_PRIMARY,
            markersize=8,
            markerfacecolor=COLOR_PRIMARY,
            markeredgecolor="white",
            markeredgewidth=2,
        )
        
        # Add value labels on points
        for x, y in zip(xs, ys):
            ax.text(
                x,
                y,
                f"{y:.1f}",
                ha="center",
                va="bottom",
                color=TEXT_COLOR,
                fontweight="bold",
                fontsize=9,
            )
    
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=20)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold", color="#cbd5e1")
    ax.set_xlabel("Time Window", fontsize=12, fontweight="bold", color="#cbd5e1")
    ax.grid(True, alpha=0.1, linestyle="--", color="#cbd5e1")
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    return fig
