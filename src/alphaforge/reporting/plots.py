"""Publication-quality matplotlib charts for backtest and research reporting.

Style rules
-----------
* One layout engine per figure: ``constrained_layout=True`` everywhere,
  no ``tight_layout`` / ``subplots_adjust`` / ``bbox_inches='tight'``.
* Legends anchored OUTSIDE the axes when they cover data.
* Consistent institutional palette (navy / teal / amber / crimson).
* Every figure is saved at >= 150 DPI and also returned so callers can
  compose them into tearsheets or the web dashboard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# ---- shared style ----------------------------------------------------- #
COL_STRATEGY = "#0B3C5D"  # navy
COL_BENCH = "#B8B8B8"  # grey
COL_ACCENT = "#1D8A99"  # teal
COL_WARN = "#D98E04"  # amber
COL_BAD = "#A4202D"  # crimson
COL_GOOD = "#2E7D32"  # green

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#666666",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "savefig.dpi": 150,
    }
)


def _cum(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


# ---------------------------------------------------------------------- #
def plot_equity_curves(
    net: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve",
    log_scale: bool = False,
) -> plt.Figure:
    """Equity curve with drawdown shading underneath."""
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True, height_ratios=[3, 1], constrained_layout=True
    )
    ax.plot(
        _cum(net).index,
        _cum(net).values,
        color=COL_STRATEGY,
        lw=1.6,
        label="Strategy (net of costs)",
    )
    if benchmark is not None:
        bench_cum = _cum(benchmark.reindex(net.index))
        ax.plot(bench_cum.index, bench_cum.values, color=COL_BENCH, lw=1.2, label="Benchmark (SPY)")
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("Cumulative growth ($1 invested)")
    ax.set_title(title)
    ax.legend(loc="upper left")

    dd = _cum(net) / _cum(net).cummax() - 1
    ax2.fill_between(dd.index, dd.values, 0, color=COL_BAD, alpha=0.55, lw=0)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    return fig


def plot_rolling_sharpe(net: pd.Series, window: int = 126) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    roll = net.rolling(window).mean() / net.rolling(window).std(ddof=1) * np.sqrt(252)
    ax.plot(roll.index, roll.values, color=COL_ACCENT, lw=1.4)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.axhline(1, color=COL_WARN, lw=0.8, ls="--", label="Sharpe = 1")
    ax.set_ylabel(f"Rolling {window // 21}-month Sharpe")
    ax.set_xlabel("Date")
    ax.set_title("Rolling Sharpe Ratio")
    ax.legend(loc="upper left")
    return fig


def plot_monthly_heatmap(net: pd.Series) -> plt.Figure:
    m = (1 + net.fillna(0)).resample("ME").prod() - 1
    df = pd.DataFrame({"ret": m, "y": m.index.year, "m": m.index.month})
    pivot = df.pivot(index="y", columns="m", values="ret")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = months[: pivot.shape[1]]
    pivot["Year"] = pivot.mean(axis=1) * 1
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(pivot) + 1.6), constrained_layout=True)
    vmax = np.nanmax(np.abs(pivot.to_numpy(dtype=float)))
    im = ax.imshow(
        pivot.to_numpy(dtype=float) * 100,
        cmap="RdYlGn",
        vmin=-max(vmax * 100, 3),
        vmax=max(vmax * 100, 3),
        aspect="auto",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                ax.text(
                    j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=7.5, color="black"
                )
    ax.set_title("Monthly Net Returns (%)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="%")
    return fig


def plot_ic_series(ic: pd.Series, title: str = "Daily Rank IC", cum: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    if cum:
        ax.plot(ic.index, ic.fillna(0).cumsum(), color=COL_STRATEGY, lw=1.5)
        ax.set_ylabel("Cumulative IC")
        ax.set_title(f"{title} (cumulative)")
    else:
        ax.bar(ic.index, ic.values, width=1.5, color=COL_ACCENT, alpha=0.75)
        ax.axhline(ic.mean(), color=COL_WARN, lw=1.2, label=f"mean = {ic.mean():+.3f}")
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_ylabel("Rank IC")
        ax.set_title(title)
        ax.legend(loc="upper left")
    ax.set_xlabel("Date")
    return fig


def plot_ic_heatmap(ic_by_date: pd.DataFrame, title: str = "Model IC") -> plt.Figure:
    """Year x month heatmap of a model's daily IC."""
    ic = ic_by_date.iloc[:, 0] if isinstance(ic_by_date, pd.DataFrame) else ic_by_date
    df = pd.DataFrame({"ic": ic, "y": ic.index.year, "m": ic.index.month})
    pivot = df.pivot_table(index="y", columns="m", values="ic", aggfunc="mean")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ][: pivot.shape[1]]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(pivot) + 1.6), constrained_layout=True)
    vmax = np.nanmax(np.abs(pivot.to_numpy(dtype=float)))
    im = ax.imshow(
        pivot.to_numpy(dtype=float),
        cmap="RdYlGn",
        vmin=-max(vmax, 0.02),
        vmax=max(vmax, 0.02),
        aspect="auto",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5, color="black")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7)
    return fig


def plot_quantile_curves(
    qret: pd.DataFrame, title: str = "Quantile Portfolio Cumulative Returns"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    cmap = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(qret.columns)))
    for color, col in zip(cmap, qret.columns, strict=False):
        cum = _cum(qret[col])
        ax.plot(cum.index, cum.values, lw=1.2, color=color, label=str(col))
    ax.set_ylabel("Cumulative growth")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=len(qret.columns), fontsize=8)
    return fig


def plot_ic_decay(decay: pd.Series, name: str = "factor") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
    ax.plot(decay.index.astype(str), decay.values, marker="o", color=COL_STRATEGY, lw=1.6)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel("Forward horizon (trading days)")
    ax.set_ylabel("Mean rank IC")
    ax.set_title(f"IC Decay - {name}")
    return fig


def plot_feature_importance(
    imp: pd.Series, top: int = 15, title: str = "Feature Importance (gain)"
) -> plt.Figure:
    imp = imp.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(imp) + 1.2), constrained_layout=True)
    colors = [COL_ACCENT if v >= 0 else COL_BAD for v in imp.values]
    ax.barh(imp.index, imp.values, color=colors, height=0.65)
    ax.set_xlabel("Importance share")
    ax.set_title(title)
    return fig


def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix") -> plt.Figure:
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6.8), constrained_layout=True)
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
            )
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
