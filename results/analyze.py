from __future__ import annotations

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from uncertainty import trigger_error_correlation

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "mode"]).agg(
        n=("sample_id", "count"),
        accuracy=("correct", "mean"),
        mean_tokens=("n_tokens", "mean"),
        trigger_rate=("trigger_fired", "mean"),
    ).reset_index()
    g["accuracy"] = (g["accuracy"] * 100).round(1)
    g["trigger_rate"] = (g["trigger_rate"] * 100).round(1)
    g["mean_tokens"] = g["mean_tokens"].round(0)
    return g


def plot_efficiency_frontier(summary: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, summary["dataset"].nunique(), figsize=(6 * summary["dataset"].nunique(), 5), squeeze=False)
    for ax, (dataset_name, sub) in zip(axes[0], summary.groupby("dataset")):
        for _, row in sub.iterrows():
            ax.scatter(row["mean_tokens"], row["accuracy"], s=90)
            ax.annotate(row["mode"], (row["mean_tokens"], row["accuracy"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlabel("Mean tokens / sample")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(dataset_name)
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy vs. token cost across conditions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def calibration_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    adaptive_modes = [m for m in df["mode"].unique() if m.startswith("adaptive")]
    for (dataset_name, mode), sub in df[df["mode"].isin(adaptive_modes)].groupby(["dataset", "mode"]):
        stats = trigger_error_correlation(
            trigger_flags=sub["trigger_fired"].tolist(),
            correctness_flags=sub["correct"].tolist(),
        )
        stats.update({"dataset": dataset_name, "mode": mode})
        rows.append(stats)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    summary = summary_table(df)
    summary_path = os.path.join(args.out_dir, "summary_table.csv")
    summary.to_csv(summary_path, index=False)
    print("Summary (accuracy / tokens / trigger rate)")
    print(summary.to_string(index=False))

    plot_path = os.path.join(args.out_dir, "efficiency_frontier.png")
    plot_efficiency_frontier(summary, plot_path)
    print(f"\nSaved efficiency frontier plot -> {plot_path}")

    calib = calibration_report(df)
    calib_path = os.path.join(args.out_dir, "trigger_calibration.csv")
    calib.to_csv(calib_path, index=False)
    print("\nTrigger-error correlation (RQ4)")
    print(calib.to_string(index=False))
    print(f"\nSaved calibration table -> {calib_path}")


if __name__ == "__main__":
    main()
