from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class DatasetProfile:
    rows: int
    cols: int
    exact_duplicates: int
    column_summary: pd.DataFrame
    numeric_stats: pd.DataFrame


def build_profile(df: pd.DataFrame) -> DatasetProfile:
    """Tworzy pełny profil diagnostyczny zbioru danych."""
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique_count": df.nunique(dropna=True),
    })

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        stats = numeric_df.describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3)
        stats = stats.rename(columns={"50%": "median"})
    else:
        stats = pd.DataFrame()

    return DatasetProfile(
        rows=len(df),
        cols=len(df.columns),
        exact_duplicates=int(df.duplicated().sum()),
        column_summary=summary,
        numeric_stats=stats,
    )


def get_iqr_mask(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.Series:
    """Zwraca maskę logiczną (True = outlier) dla kolumny numerycznej metodą IQR."""
    if column not in df.columns:
        raise KeyError(f"Kolumna '{column}' nie istnieje w zbiorze.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Kolumna '{column}' nie jest numeryczna – nie można policzyć IQR.")

    series = df[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=df.index)

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper)


def get_out_of_range_mask(
    df: pd.DataFrame,
    column: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> pd.Series:
    """Zwraca maskę logiczną dla wartości wykraczających poza zadany przedział [min_val, max_val]."""
    if column not in df.columns:
        raise KeyError(f"Kolumna '{column}' nie istnieje w zbiorze.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Kolumna '{column}' musi być numeryczna do sprawdzania zakresu.")
    if min_val is None and max_val is None:
        raise ValueError("Należy podać przynajmniej jedną granicę: min_val lub max_val.")
    if min_val is not None and max_val is not None and min_val > max_val:
        raise ValueError(f"Wartość minimalna ({min_val}) nie może być większa od maksymalnej ({max_val}).")

    series = df[column]
    mask = pd.Series(False, index=df.index)
    if min_val is not None:
        mask = mask | (series < min_val)
    if max_val is not None:
        mask = mask | (series > max_val)
    return mask.fillna(False)
