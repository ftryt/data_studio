from pathlib import Path
from src.session import DataSession
from src.io_manager import save_csv, save_text_report


def main() -> None:
    input_path = Path("data/california_housing_dirty.csv")

    if not input_path.exists():
        print(f"No file in: {input_path.resolve()}")
        return

    session = DataSession.from_csv(input_path)

    print("=== PROFIL PRZED CZYSZCZENIEM ===")
    before = session.get_profile(use_raw=True)
    print(f"Wymiary: {before.rows} x {before.cols} | Duplikaty: {before.exact_duplicates}")
    print(before.column_summary)
    print("\nStatystyki numeryczne (przed):")
    print(before.numeric_stats)

    # --- PRZYKŁADOWY PIPELINE CZYSZCZENIA ---
    # Usuwamy pełne duplikaty
    session.drop_duplicates()

    # Reguły zakresu
    # Np. wiek domu nie może być ujemny ani > 120 lat; mediana dochodu > 0
    if "housing_median_age" in session.current.columns:
        session.apply_range_rule("housing_median_age", min_val=1, max_val=120, action="to_nan")
        session.clean_missing("housing_median_age", strategy="median")

    # Uzupełnienie braków w pozostałych kolumnach
    if "total_bedrooms" in session.current.columns:
        session.clean_missing("total_bedrooms", strategy="median")

    if "ocean_proximity" in session.current.columns:
        session.clean_missing("ocean_proximity", strategy="mode")

    # Outliery metodą IQR (najpierw oznaczamy flagą, żeby nie tracić danych bez uzasadnienia)
    if "median_income" in session.current.columns:
        session.process_outliers("median_income", factor=1.5, action="flag")

    # --- RAPORT I EKSPORT ---
    report = session.generate_report_text()
    print("\n" + report)

    save_csv(session.current, "output/cleaned.csv")
    save_text_report(report, "output/raport_zmian_lab6.txt")
    print("\nZapisano pliki: output/cleaned.csv oraz output/raport_zmian_lab6.txt")


if __name__ == "__main__":
    main()
