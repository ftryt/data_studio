from __future__ import annotations

import pandas as pd
from src.profiling import get_iqr_mask, get_out_of_range_mask


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Usuwa pełne duplikaty wierszy."""
    cleaned = df.drop_duplicates().copy()
    affected = len(df) - len(cleaned)
    return cleaned, affected


def handle_missing_values(
    df: pd.DataFrame,
    column: str,
    strategy: str,
    fill_value: str | float | None = None,
) -> tuple[pd.DataFrame, int]:
    """
    Obsługuje braki danych w wybranej kolumnie.
    Dostępne strategie: 'drop_rows', 'mean', 'median', 'mode', 'constant'.
    """
    if column not in df.columns:
        raise KeyError(f"Kolumna '{column}' nie istnieje.")

    cleaned = df.copy()
    missing_mask = cleaned[column].isna()
    affected = int(missing_mask.sum())

    if affected == 0:
        return cleaned, 0

    if strategy == "drop_rows":
        cleaned = cleaned.dropna(subset=[column]).copy()
    elif strategy == "mean":
        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            raise TypeError(f"Strategia 'mean' wymaga kolumny numerycznej ('{column}').")
        cleaned[column] = cleaned[column].fillna(cleaned[column].mean())
    elif strategy == "median":
        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            raise TypeError(f"Strategia 'median' wymaga kolumny numerycznej ('{column}').")
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    elif strategy == "mode":
        modes = cleaned[column].mode(dropna=True)
        if modes.empty:
            raise ValueError(f"Nie można wyznaczyć dominanty dla pustej kolumny '{column}'.")
        cleaned[column] = cleaned[column].fillna(modes.iloc[0])
    elif strategy == "constant":
        if fill_value is None or str(fill_value).strip() == "":
            raise ValueError("Dla strategii 'constant' należy podać wartość wypełniającą.")
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            fill_value = float(fill_value)
        cleaned[column] = cleaned[column].fillna(fill_value)
    else:
        raise ValueError(f"Nieznana strategia obsługi braków: '{strategy}'.")

    return cleaned, affected


def handle_out_of_range(
    df: pd.DataFrame,
    column: str,
    min_val: float | None = None,
    max_val: float | None = None,
    action: str = "to_nan",
) -> tuple[pd.DataFrame, int]:
    """
    Obsługuje wartości spoza zakresu.
    Akcje: 'to_nan' (zamiana na brak), 'clip' (przycięcie do granic), 'drop_rows' (usunięcie wierszy).
    """
    mask = get_out_of_range_mask(df, column, min_val, max_val)
    affected = int(mask.sum())
    cleaned = df.copy()

    if affected == 0:
        return cleaned, 0

    if action == "to_nan":
        cleaned.loc[mask, column] = pd.NA
    elif action == "clip":
        cleaned[column] = cleaned[column].clip(lower=min_val, upper=max_val)
    elif action == "drop_rows":
        cleaned = cleaned.loc[~mask].copy()
    else:
        raise ValueError(f"Nieznana akcja dla wartości spoza zakresu: '{action}'.")

    return cleaned, affected


def handle_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    factor: float = 1.5,
    action: str = "flag",
) -> tuple[pd.DataFrame, int]:
    """
    Obsługuje obserwacje odstające wykryte metodą IQR.
    Akcje:
    - 'flag': dodaje/aktualizuje kolumnę logiczną '{column}_is_outlier' (nie usuwa danych)
    - 'clip': przycina wartości do granic [Q1 - factor*IQR, Q3 + factor*IQR] (winsoryzacja)
    - 'drop_rows': usuwa wiersze z outlierami
    """
    mask = get_iqr_mask(df, column, factor=factor)
    affected = int(mask.sum())
    cleaned = df.copy()

    if action == "flag":
        flag_col = f"{column}_is_outlier"
        cleaned[flag_col] = mask
    elif action == "clip":
        if affected > 0:
            q1 = cleaned[column].quantile(0.25)
            q3 = cleaned[column].quantile(0.75)
            iqr = q3 - q1
            cleaned[column] = cleaned[column].clip(
                lower=q1 - factor * iqr,
                upper=q3 + factor * iqr,
            )
    elif action == "drop_rows":
        cleaned = cleaned.loc[~mask].copy()
    else:
        raise ValueError(f"Nieznana akcja dla outlierów: '{action}'.")

    return cleaned, affected
