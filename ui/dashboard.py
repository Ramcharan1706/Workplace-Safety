"""Chart helpers for the analytics dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Modern color palette
COLOR_SAFE = "#10b981"
COLOR_WARNING = "#f59e0b"
COLOR_DANGER = "#ef4444"
COLOR_LIMITED = "#94a3b8"
COLOR_PRIMARY = "#3b82f6"
COLOR_SECONDARY = "#8b5cf6"
BG_COLOR = "#0f172a"
TEXT_COLOR = "#f1f5f9"
GRID_COLOR = "#334155"


def _setup_plot_style():
    """Configure matplotlib for dark theme."""
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = "#1e293b"
    plt.rcParams["axes.facecolor"] = "#0f172a"
    plt.rcParams["text.color"] = TEXT_COLOR
    plt.rcParams["axes.labelcolor"] = "#cbd5e1"
    plt.rcParams["xtick.color"] = "#cbd5e1"
    plt.rcParams["ytick.color"] = "#cbd5e1"
    plt.rcParams["axes.edgecolor"] = GRID_COLOR
    plt.rcParams["font.family"] = "sans-serif"


def _style_axes(ax):
    ax.set_facecolor("#0f172a")
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.1)
    ax.grid(axis="y", alpha=0.12, linestyle="--", color="#cbd5e1")
    ax.set_axisbelow(True)


def _annotate_bars(ax, bars) -> None:
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


def make_violation_figure(distribution: dict[str, int]):
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    labels = list(distribution.keys()) or ["No violations"]
    values = list(distribution.values()) or [1]

    palette = [COLOR_DANGER, COLOR_WARNING, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SAFE]
    colors_list = [palette[idx % len(palette)] for idx in range(len(labels))]

    bars = ax.bar(labels, values, color=colors_list, edgecolor=GRID_COLOR, linewidth=1.5, alpha=0.9)
    _annotate_bars(ax, bars)
    
    ax.set_ylabel("Count", fontsize=12, fontweight="bold", color="#cbd5e1")
    ax.set_title("Violation Type Distribution", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=20)
    ax.tick_params(axis="x", rotation=15, labelsize=10)
    _style_axes(ax)
    
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
    _style_axes(ax)
    
    fig.tight_layout()
    return fig


def make_status_figure(status_counts: dict[str, int]):
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=100)

    ordered_labels = ["Safe", "Limited", "Warning", "Unsafe", "Danger"]
    labels = [label for label in ordered_labels if status_counts.get(label, 0) > 0]
    values = [status_counts[label] for label in labels]

    if not values:
        labels = ["No data"]
        values = [1]

    colors = {
        "Safe": COLOR_SAFE,
        "Limited": COLOR_LIMITED,
        "Warning": COLOR_WARNING,
        "Unsafe": COLOR_DANGER,
        "Danger": COLOR_DANGER,
        "No data": COLOR_PRIMARY,
    }
    bars = ax.bar(labels, values, color=[colors.get(label, COLOR_PRIMARY) for label in labels], edgecolor=GRID_COLOR, linewidth=1.5, alpha=0.9)
    _annotate_bars(ax, bars)

    ax.set_title("Assessment Status Overview", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=20)
    ax.set_ylabel("Count", fontsize=12, fontweight="bold", color="#cbd5e1")
    ax.tick_params(axis="x", rotation=15, labelsize=10)
    _style_axes(ax)

    fig.tight_layout()
    return fig


def make_verdict_banner(verdict: str, reason: str):
    """Render a compact status banner for the session verdict."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 1.8), dpi=100)
    ax.axis("off")

    verdict_upper = verdict.strip() or "Inconclusive"
    color = {
        "Safe": COLOR_SAFE,
        "Limited": COLOR_LIMITED,
        "Warning": COLOR_WARNING,
        "Unsafe": COLOR_DANGER,
        "Danger": COLOR_DANGER,
    }.get(verdict_upper, COLOR_PRIMARY)

    patch = FancyBboxPatch(
        (0.02, 0.15),
        0.96,
        0.7,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=color,
        facecolor="#111827",
        alpha=0.95,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(0.05, 0.62, f"Session Verdict: {verdict_upper}", transform=ax.transAxes, fontsize=14, fontweight="bold", color=color, va="center")
    ax.text(0.05, 0.34, reason, transform=ax.transAxes, fontsize=10, color="#cbd5e1", va="center")

    fig.tight_layout()
    return fig
