# scripts/plot_minimal_results.py

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main(input_csv: str):
    df = pd.read_csv(input_csv)

    # Make sure energy is positive (in case a buggy run produced negatives)
    df["energy_kwh"] = df["energy_kwh"].abs()

    # Sort so the legend/order is stable
    df = df.sort_values("energy_kwh")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Simple scatter of Energy vs Comfort
    ax.scatter(df["energy_kwh"], df["comfort_drift"], s=120)

    # Label each point with the method name
    for _, row in df.iterrows():
        ax.annotate(
            row["method"],
            (row["energy_kwh"], row["comfort_drift"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=10,
        )

    # Axis labels and title
    ax.set_xlabel("Average HVAC Energy (kWh/day)")
    ax.set_ylabel("Average Comfort Drift (°C·h/day)")
    ax.set_title("Minimal Experiment – Energy vs Comfort")

    # Nice axis limits (a bit padded around the data)
    x_min, x_max = df["energy_kwh"].min(), df["energy_kwh"].max()
    y_min, y_max = df["comfort_drift"].min(), df["comfort_drift"].max()

    padding_x = 0.2 * (x_max - x_min + 1e-6)
    padding_y = 0.2 * (y_max - y_min + 1e-6)

    ax.set_xlim(max(0, x_min - padding_x), x_max + padding_x)
    ax.set_ylim(max(0, y_min - padding_y), y_max + padding_y)

    # Show that lower-left is better
    ax.text(
        0.98,
        0.02,
        "Lower-left = better\n(less energy, less drift)",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
    )

    ax.grid(True, linestyle=":", linewidth=0.7)
    fig.tight_layout()

    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "minimal_energy_comfort.png"
    pdf_path = out_dir / "minimal_energy_comfort.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)

    print(f"Saved plots to:\n  {png_path}\n  {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV summary file (e.g., results/minimal_run/minimal_table3.csv)",
    )
    args = parser.parse_args()
    main(args.input)
