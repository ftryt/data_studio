from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.io_manager import load_csv_safe
from src.profiling import DatasetProfile, build_profile
from src.cleaning import (
    remove_exact_duplicates,
    handle_missing_values,
    handle_out_of_range,
    handle_outliers_iqr, coerce_column_to_numeric,
)


@dataclass
class AuditEntry:
    timestamp: str
    operation: str
    column: str
    details: str
    rows_before: int
    rows_after: int
    affected_records: int

    def format_line(self) -> str:
        return (
            f"[{self.timestamp}] {self.operation} | Kolumna: {self.column} | "
            f"Szczegóły: {self.details} | Dotknięte rekordy: {self.affected_records} "
            f"(wiersze: {self.rows_before} -> {self.rows_after})"
        )


@dataclass
class DataSession:
    raw: pd.DataFrame
    source_name: str = "unknown.csv"
    current: pd.DataFrame = field(init=False)
    audit_log: list[AuditEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current = self.raw.copy(deep=True)

    @classmethod
    def from_csv(cls, path: str | Path) -> "DataSession":
        file_path = Path(path)
        df = load_csv_safe(file_path)
        return cls(raw=df, source_name=file_path.name)

    def _log(
        self,
        operation: str,
        column: str,
        details: str,
        rows_before: int,
        rows_after: int,
        affected: int,
    ) -> None:
        entry = AuditEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            operation=operation,
            column=column,
            details=details,
            rows_before=rows_before,
            rows_after=rows_after,
            affected_records=affected,
        )
        self.audit_log.append(entry)

    def get_profile(self, use_raw: bool = False) -> DatasetProfile:
        target = self.raw if use_raw else self.current
        return build_profile(target)

    def drop_duplicates(self) -> int:
        before = len(self.current)
        self.current, affected = remove_exact_duplicates(self.current)
        self._log("Usunięcie duplikatów", "WSZYSTKIE", "Pełne duplikaty wierszy", before, len(self.current), affected)
        return affected

    def clean_missing(self, column: str, strategy: str, fill_value: str | float | None = None) -> int:
        before = len(self.current)
        self.current, affected = handle_missing_values(self.current, column, strategy, fill_value)
        detail = f"strategia={strategy}" + (f", wartość={fill_value}" if fill_value is not None else "")
        self._log("Obsługa braków", column, detail, before, len(self.current), affected)
        return affected

    def apply_range_rule(
        self,
        column: str,
        min_val: float | None = None,
        max_val: float | None = None,
        action: str = "to_nan",
    ) -> int:
        before = len(self.current)
        self.current, affected = handle_out_of_range(self.current, column, min_val, max_val, action)
        detail = f"zakres=[{min_val}, {max_val}], akcja={action}"
        self._log("Reguła zakresu", column, detail, before, len(self.current), affected)
        return affected

    def convert_to_numeric(self, column: str) -> int:
        before = len(self.current)
        self.current, affected = coerce_column_to_numeric(self.current, column)
        self._log("Konwersja na typ liczbowy", column, "błędne teksty -> NaN", before, len(self.current), affected)
        return affected

    def process_outliers(self, column: str, factor: float = 1.5, action: str = "flag") -> int:
        before = len(self.current)
        self.current, affected = handle_outliers_iqr(self.current, column, factor, action)
        detail = f"IQR factor={factor}, akcja={action}"
        self._log("Obsługa outlierów", column, detail, before, len(self.current), affected)
        return affected

    def reset(self) -> None:
        before = len(self.current)
        self.current = self.raw.copy(deep=True)
        self._log("RESET SESJI", "WSZYSTKIE", "Przywrócono dane surowe", before, len(self.current), 0)

    def generate_report_text(self) -> str:
        raw_prof = self.get_profile(use_raw=True)
        cur_prof = self.get_profile(use_raw=False)

        lines = [
            "==========================================================",
            f"RAPORT JAKOŚCI I CZYSZCZENIA DANYCH: {self.source_name}",
            "==========================================================",
            f"Stan surowy:   {raw_prof.rows} wierszy, {raw_prof.cols} kolumn, duplikaty: {raw_prof.exact_duplicates}",
            f"Stan aktualny: {cur_prof.rows} wierszy, {cur_prof.cols} kolumn, duplikaty: {cur_prof.exact_duplicates}",
            "",
            "--- PROFIL KOLUMN (PO CZYSZCZENIU) ---",
            cur_prof.column_summary.to_string(),
            "",
            "--- DZIENNIK OPERACJI (AUDIT LOG) ---",
        ]
        if not self.audit_log:
            lines.append("Brak wykonanych operacji.")
        else:
            for idx, entry in enumerate(self.audit_log, start=1):
                lines.append(f"{idx}. {entry.format_line()}")

        return "\n".join(lines)
