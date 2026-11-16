from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_csv_safe(path: str | Path) -> pd.DataFrame:
    """Wczytuje plik CSV z automatycznym wykrywaniem separatora i obsługą kodowania."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError("Wybrany plik CSV jest pusty.")

    encodings = ["utf-8", "cp1250", "latin1"]
    last_error: Exception | None = None

    for enc in encodings:
        try:
            # sep=None z silnikiem python automatycznie wykrywa ',' oraz ';'
            df = pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                encoding=enc,
                on_bad_lines="error",
            )
            if df.empty or len(df.columns) == 0:
                raise ValueError("Plik nie zawiera żadnych rekordów ani kolumn.")
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except pd.errors.ParserError as e:
            raise ValueError(
                f"Błąd struktury pliku CSV (wadliwy wiersz lub separator): {e}"
            ) from e
        except pd.errors.EmptyDataError as e:
            raise ValueError("Plik CSV nie zawiera danych do wczytania.") from e

    raise ValueError(
        f"Nie udało się odczytać pliku w obsługiwanych kodowaniach ({', '.join(encodings)})."
    ) from last_error


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Zapisuje DataFrame do CSV w formacie UTF-8 bez indeksu."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def save_text_report(content: str, path: str | Path) -> Path:
    """Zapisuje raport tekstowy (np. dziennik zmian) do pliku."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path
