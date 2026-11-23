from __future__ import annotations

from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def _validate_numeric_column(df: pd.DataFrame, col: str, allow_constant: bool = False) -> pd.Series:
    """Validate that a column exists, can be coerced to numeric, is non-empty, and optionally non-constant."""
    if col not in df.columns:
        raise KeyError(f"Kolumna '{col}' nie istnieje w zbiorze danych.")

    # Coerce string artifacts (e.g. '?', 'N/A') to NaN so dirty numeric columns can still be plotted
    numeric_series = pd.to_numeric(df[col], errors="coerce")
    clean_series = numeric_series.dropna()

    if clean_series.empty:
        raise TypeError(f"Kolumna '{col}' nie zawiera danych numerycznych (jest tekstowa lub pusta).")
    if not allow_constant and clean_series.nunique() < 2:
        raise ValueError(f"Kolumna '{col}' ma wartość stałą (brak zmienności) – wykres jest bezsensowny.")

    return clean_series


def _validate_categorical_column(df: pd.DataFrame, col: str, max_categories: int = 25) -> pd.Series:
    """Validate that a column is suitable for categorical bar plotting."""
    if col not in df.columns:
        raise KeyError(f"Kolumna '{col}' nie istnieje w zbiorze danych.")

    clean_series = df[col].dropna()
    if clean_series.empty:
        raise ValueError(f"Kolumna '{col}' jest pusta.")
    if clean_series.nunique() > max_categories:
        raise ValueError(
            f"Kolumna '{col}' ma zbyt wiele unikalnych wartości ({clean_series.nunique()} > {max_categories}). "
            "Wybierz kolumnę kategoryczną lub użyj histogramu."
        )
    return clean_series


def build_chart(
        df: pd.DataFrame,
        chart_type: str,
        x_col: str,
        y_col: str | None = None,
        hue_col: str | None = None,
        title: str | None = None,
) -> Figure:
    """
    Build a matplotlib Figure based on chart type and validated columns.
    Supported chart_type values: 'histogram', 'scatter', 'box', 'bar'.
    """
    sns.set_theme(style="whitegrid")
    fig = Figure(figsize=(7.5, 4.5), dpi=100)
    ax = fig.add_subplot(111)

    if chart_type == "histogram":
        numeric_data = _validate_numeric_column(df, x_col, allow_constant=False)
        sns.histplot(x=numeric_data, kde=True, ax=ax, color="#2b7bba")
        ax.set_title(title or f"Rozkład zmiennej: {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Liczba obserwacji")

    elif chart_type == "scatter":
        if not y_col:
            raise ValueError("Wykres punktowy wymaga wybrania obu osi: X oraz Y.")
        if x_col == y_col:
            raise ValueError("Osie X oraz Y muszą wskazywać na różne kolumny.")

        x_num = _validate_numeric_column(df, x_col, allow_constant=False)
        y_num = _validate_numeric_column(df, y_col, allow_constant=False)

        plot_df = pd.DataFrame({x_col: x_num, y_col: y_num}).dropna()
        if len(plot_df) < 2:
            raise ValueError("Za mało kompletnych par (X, Y) do narysowania wykresu punktowego.")

        corr = plot_df[x_col].corr(plot_df[y_col])

        sns.regplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            ax=ax,
            scatter_kws={"alpha": 0.45, "s": 20, "color": "#2b7bba"},
            line_kws={"color": "#d95f02", "linewidth": 2},
        )
        ax.set_title(title or f"Zależność: {x_col} vs {y_col} (r = {corr:.3f})")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.text(
            0.03,
            0.94,
            f"Korelacja Pearsona r = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
        )

    elif chart_type == "box":
        if y_col and y_col != x_col:
            _validate_categorical_column(df, x_col, max_categories=15)
            y_num = _validate_numeric_column(df, y_col, allow_constant=False)
            sns.boxplot(x=df[x_col], y=y_num, hue=df[x_col], legend=False, ax=ax, palette="Set2")
            ax.set_title(title or f"Rozkład pudełkowy {y_col} względem {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.tick_params(axis="x", rotation=20)
        else:
            x_num = _validate_numeric_column(df, x_col, allow_constant=False)
            sns.boxplot(y=x_num, ax=ax, color="#66c2a5", width=0.4)
            ax.set_title(title or f"Wykres pudełkowy (IQR): {x_col}")
            ax.set_xlabel("")
            ax.set_ylabel(x_col)

    elif chart_type == "bar":
        _validate_categorical_column(df, x_col, max_categories=20)
        if y_col and y_col != x_col:
            _validate_numeric_column(df, y_col, allow_constant=True)
            sns.barplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=x_col,
                legend=False,
                ax=ax,
                estimator="mean",
                errorbar=None,
                palette="viridis",
            )
            ax.set_title(title or f"Średnia wartość {y_col} wg {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(f"Średnia ({y_col})")
        else:
            order = df[x_col].value_counts().index
            sns.countplot(
                data=df,
                x=x_col,
                hue=x_col,
                legend=False,
                order=order,
                ax=ax,
                palette="viridis",
            )

            ax.set_title(title or f"Liczebność kategorii: {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel("Liczba rekordów")
        ax.tick_params(axis="x", rotation=25)

    else:
        raise ValueError(f"Nieobsługiwany typ wykresu: '{chart_type}'.")

    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Export a matplotlib Figure to PNG, PDF, or SVG."""
    out_path = Path(path)
    if out_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("Obsługiwane formaty eksportu wykresu to: .png, .pdf, .svg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path
