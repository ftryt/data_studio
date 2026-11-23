from pathlib import Path
import pandas as pd
from src.session import DataSession
from src.visualization import build_chart, save_figure


def main() -> None:
    input_path = Path("data/california_housing_projekt2.csv")
    if not input_path.exists():
        print(f"Place input CSV at: {input_path.resolve()}")
        return

    session = DataSession.from_csv(input_path)

    for col in ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "MedHouseVal"]:
        if col in session.current.columns:
            session.convert_to_numeric(col)

    # 1. Generate 'BEFORE' chart on raw data (showing range anomalies)
    fig_box_before = build_chart(
        session.raw,
        chart_type="box",
        x_col="HouseAge",
        title="PRZED czyszczeniem: Wiek budynku (HouseAge)",
    )
    save_figure(fig_box_before, "output/lab7_box_age_before.png")

    # 2. Run cleaning pipeline
    session.drop_duplicates()
    session.apply_range_rule("HouseAge", min_val=1, max_val=120, action="to_nan")
    for col in ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "MedHouseVal"]:
        if col in session.current.columns:
            session.clean_missing(col, strategy="median")

    # 3. Generate 'AFTER' set of all 4 required chart types (PNG + PDF/SVG)
    # Chart 1: Box plot (after cleaning)
    fig_box_after = build_chart(
        session.current,
        chart_type="box",
        x_col="HouseAge",
        title="PO czyszczeniu: Wiek budynku (HouseAge)",
    )
    save_figure(fig_box_after, "output/lab7_box_age_after.png")

    # Chart 2: Histogram
    fig_hist = build_chart(
        session.current,
        chart_type="histogram",
        x_col="MedHouseVal",
        title="Rozkład wartości domów (MedHouseVal)",
    )
    save_figure(fig_hist, "output/lab7_histogram_value.png")
    save_figure(fig_hist, "output/lab7_histogram_value.pdf")

    # Chart 3: Scatter plot with Pearson correlation
    fig_scatter = build_chart(
        session.current,
        chart_type="scatter",
        x_col="MedInc",
        y_col="MedHouseVal",
    )
    save_figure(fig_scatter, "output/lab7_scatter_income_value.png")
    save_figure(fig_scatter, "output/lab7_scatter_income_value.svg")

    # Chart 4: Bar plot (bin numeric HouseAge into categorical age groups for visualization)
    plot_df = session.current.copy()
    plot_df["AgeGroup"] = pd.cut(
        plot_df["HouseAge"],
        bins=[0, 15, 30, 45, 120],
        labels=["Nowe (1-15)", "Średnie (16-30)", "Starsze (31-45)", "Zabytkowe (46+)"],
    )
    fig_bar = build_chart(
        plot_df,
        chart_type="bar",
        x_col="AgeGroup",
        y_col="MedHouseVal",
        title="Średnia wartość domu wg grupy wiekowej budynku",
    )
    save_figure(fig_bar, "output/lab7_bar_age_group.png")

    print("Lab 7 charts generated successfully in output/ directory.")


if __name__ == "__main__":
    main()
