"""
Shared plot style for DiGiTerra model visualizations.
Apply a softer, seaborn-like look to all matplotlib figures.
Import this module once at the start of any code path that creates plots.
"""
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(
        style="whitegrid",
        palette="muted",
        font_scale=1.05,
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.color": ".9",
            "grid.linestyle": "-",
            "axes.edgecolor": ".4",
            "axes.linewidth": 1,
            "font.family": ["sans-serif"],
            "xtick.color": ".3",
            "ytick.color": ".3",
        },
    )
except Exception:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")

# Softer defaults for scatter/plot elements when not using seaborn API
plt.rcParams.setdefault("scatter.edgecolors", "none")
plt.rcParams.setdefault("patch.edgecolor", "none")


def apply_plot_style():
    """Re-apply the shared plot style. Call at the start of plotting functions to ensure style is active."""
    try:
        import seaborn as sns
        sns.set_theme(
            style="whitegrid",
            palette="muted",
            font_scale=1.05,
            rc={
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "grid.color": ".9",
                "axes.edgecolor": ".4",
                "xtick.color": ".3",
                "ytick.color": ".3",
            },
        )
    except Exception:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            plt.style.use("ggplot")
