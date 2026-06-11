from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def add_bar_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}" if abs(height) < 10 else f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_cohort_feature(df: pd.DataFrame, feature: str) -> None:
    sub = df[df["cohort_feature"] == feature].copy()
    sub = sub.sort_values(["recall", "n"], ascending=[True, False]).reset_index(drop=True)
    labels = sub["cohort_value"].astype(str).tolist()
    x = np.arange(len(sub))

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f"Cohort analysis - {feature}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    bars = ax.bar(x, sub["n"], color="#4C78A8")
    ax.set_title("Cantidad de datos")
    ax.set_ylabel("n")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    add_bar_labels(ax, bars)

    ax = axes[0, 1]
    bars = ax.bar(x, sub["cancellation_rate"], color="#F58518")
    ax.set_title("Cancellation rate")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    add_bar_labels(ax, bars)

    ax = axes[1, 0]
    bars = ax.bar(x, sub["recall"], color="#E45756")
    ax.set_title("Recall")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    add_bar_labels(ax, bars)

    ax = axes[1, 1]
    bottom = np.zeros(len(sub))
    colors = {"tn": "#72B7B2", "fp": "#F28E2B", "fn": "#E15759", "tp": "#B07AA1"}
    for metric in ["tn", "fp", "fn", "tp"]:
        bars = ax.bar(
            x,
            sub[metric],
            bottom=bottom,
            label=metric.upper(),
            color=colors[metric],
        )
        bottom += sub[metric].to_numpy()
    ax.set_title("Matriz de confusión")
    ax.set_ylabel("conteo")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=9, loc="upper right")

    plt.tight_layout()
    out = PLOTS_DIR / f"cohort_summary_{feature}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_comparison(df: pd.DataFrame) -> None:
    order = ["baseline", "real_data", "smote", "smote_cohort"]
    df = df.set_index("strategy").reindex([s for s in order if s in df["strategy"].tolist()]).reset_index()
    strategies = df["strategy"].tolist()
    x = np.arange(len(strategies))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Strategy comparison - CV", fontsize=16, fontweight="bold")

    metrics = [
        ("global_f1", "Global F1", "#4C78A8"),
        ("global_recall", "Global recall", "#F58518"),
        ("cohort_f1", "Cohort F1", "#E45756"),
        ("cohort_recall", "Cohort recall", "#72B7B2"),
    ]

    for ax, (metric, title, color) in zip(axes.flat, metrics):
        bars = ax.bar(x, df[metric], color=color)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=20)
        ax.grid(axis="y", alpha=0.25)
        add_bar_labels(ax, bars)

    plt.tight_layout()
    out = PLOTS_DIR / "strategy_comparison_cv.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_confusions(df: pd.DataFrame) -> None:
    order = ["baseline", "real_data", "smote", "smote_cohort"]
    df = df.set_index("strategy").reindex([s for s in order if s in df["strategy"].tolist()]).reset_index()
    strategies = df["strategy"].tolist()
    x = np.arange(len(strategies))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Strategy comparison - confusion counts", fontsize=16, fontweight="bold")

    for ax, metrics, title in [
        (axes[0], ["global_tn", "global_fp", "global_fn", "global_tp"], "Global confusion"),
        (axes[1], ["cohort_tn", "cohort_fp", "cohort_fn", "cohort_tp"], "Cohort confusion"),
    ]:
        bottom = np.zeros(len(df))
        colors = {"tn": "#72B7B2", "fp": "#F28E2B", "fn": "#E15759", "tp": "#B07AA1"}
        for metric in metrics:
            label = metric.split("_")[1].upper()
            bars = ax.bar(
                x,
                df[metric],
                bottom=bottom,
                label=label,
                color=colors[label.lower()],
            )
            bottom += df[metric].to_numpy()
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=20)
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=4, fontsize=9, loc="upper right")

    plt.tight_layout()
    out = PLOTS_DIR / "strategy_confusions.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_deltas(df: pd.DataFrame) -> None:
    base = df[df["strategy"] == "baseline"].iloc[0]
    compare = df[df["strategy"] != "baseline"].copy()
    compare = compare.sort_values("global_f1", ascending=False)

    metrics = [
        ("global_f1", "global_f1", "Global F1"),
        ("global_recall", "global_recall", "Global recall"),
        ("cohort_f1", "cohort_f1", "Cohort F1"),
        ("cohort_recall", "cohort_recall", "Cohort recall"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Delta vs baseline", fontsize=16, fontweight="bold")

    for ax, (metric, _, title) in zip(axes.flat, metrics):
        delta = compare[metric] - base[metric]
        bars = ax.bar(compare["strategy"], delta, color="#54A24B")
        ax.axhline(0, color="black", lw=1)
        ax.set_title(title)
        ax.set_ylabel("delta")
        ax.grid(axis="y", alpha=0.25)
        add_bar_labels(ax, bars)

    plt.tight_layout()
    out = PLOTS_DIR / "strategy_deltas_vs_baseline.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_focus_internal_research(df: pd.DataFrame) -> None:
    order = ["baseline", "real_data", "smote", "smote_cohort"]
    df = df.set_index("strategy").reindex([s for s in order if s in df["strategy"].tolist()]).reset_index()
    strategies = df["strategy"].tolist()
    x = np.arange(len(strategies))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        "Strategy comparison focused on internal_research",
        fontsize=16,
        fontweight="bold",
    )

    global_metrics = [
        ("global_accuracy", "Global accuracy", "#4C78A8"),
        ("global_precision", "Global precision", "#59A14F"),
        ("global_recall", "Global recall", "#F58518"),
        ("global_f1", "Global F1", "#E45756"),
    ]
    cohort_metrics = [
        ("cohort_accuracy", "Cohort accuracy", "#4C78A8"),
        ("cohort_precision", "Cohort precision", "#59A14F"),
        ("cohort_recall", "Cohort recall", "#F58518"),
        ("cohort_f1", "Cohort F1", "#E45756"),
    ]

    for col, (metric, title, color) in enumerate(global_metrics):
        ax = axes[0, col]
        bars = ax.bar(x, df[metric], color=color)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=20)
        ax.grid(axis="y", alpha=0.25)
        add_bar_labels(ax, bars)

    for col, (metric, title, color) in enumerate(cohort_metrics):
        ax = axes[1, col]
        bars = ax.bar(x, df[metric], color=color)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=20)
        ax.grid(axis="y", alpha=0.25)
        add_bar_labels(ax, bars)

    axes[0, 0].set_ylabel("global")
    axes[1, 0].set_ylabel("cohort")

    plt.tight_layout()
    out = PLOTS_DIR / "strategy_internal_research_focus.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_table_internal_research(df: pd.DataFrame) -> None:
    order = ["baseline", "real_data", "smote", "smote_cohort"]
    df = df.set_index("strategy").reindex([s for s in order if s in df["strategy"].tolist()]).reset_index()

    train = load_csv(Path("data/processed/train.csv"))
    extra = load_csv(Path("data/raw/extra_client_type_internal_research.csv"))
    smote_summary_path = Path("results/smote_cohort_summary.json")
    smote_summary = None
    if smote_summary_path.exists():
        import json

        smote_summary = json.loads(smote_summary_path.read_text(encoding="utf-8"))
    train_internal_n = int(train["client_type"].eq("internal_research").sum())
    extra_internal_n = int(extra["client_type"].eq("internal_research").sum())
    train_total_n = int(len(train))
    extra_total_n = int(len(extra))
    smote_total_n = 9620
    smote_added_n = smote_total_n - train_total_n

    data_rows = {
        "baseline": {
            "train_total_n": train_total_n,
            "internal_original_n": train_internal_n,
            "internal_added_n": 0,
            "internal_total_n": train_internal_n,
        },
        "real_data": {
            "train_total_n": train_total_n + extra_total_n,
            "internal_original_n": train_internal_n,
            "internal_added_n": extra_internal_n,
            "internal_total_n": train_internal_n + extra_internal_n,
        },
        "smote": {
            "train_total_n": smote_total_n,
            "internal_original_n": train_internal_n,
            "internal_added_n": smote_added_n,
            "internal_total_n": train_internal_n,
        },
        "smote_cohort": {
            "train_total_n": "fold-based",
            "internal_original_n": train_internal_n,
            "internal_added_n": (
                int(smote_summary["cohort_synthetic_total"])
                if smote_summary is not None
                else "fold-based"
            ),
            "internal_total_n": (
                int(train_internal_n + smote_summary["cohort_synthetic_total"])
                if smote_summary is not None
                else "fold-based"
            ),
        },
    }

    display_df = pd.DataFrame(
        {
            "strategy": df["strategy"],
            "train_total_n": [str(data_rows[s]["train_total_n"]) for s in df["strategy"]],
            "internal_original_n": [str(data_rows[s]["internal_original_n"]) for s in df["strategy"]],
            "internal_added_n": [str(data_rows[s]["internal_added_n"]) for s in df["strategy"]],
            "internal_total_n": [str(data_rows[s]["internal_total_n"]) for s in df["strategy"]],
            "global_f1": df["global_f1"].map(lambda x: f"{x:.3f}"),
            "cohort_rec": df["cohort_recall"].map(lambda x: f"{x:.3f}"),
            "cohort_f1": df["cohort_f1"].map(lambda x: f"{x:.3f}"),
        }
    )

    fig_w = 17
    fig_h = 1.2 + 0.55 * len(display_df)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(
        "internal_research - comparison table by strategy",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    table = ax.table(
        cellText=display_df.values,
        colLabels=[
            "strategy",
            "train total n",
            "internal original n",
            "internal added n",
            "internal total n",
            "global f1",
            "cohort rec",
            "cohort f1",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    header_color = "#2F4B7C"
    row_colors = ["#F3F6FA", "#FFFFFF"]

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])

    note = (
        "internal_original_n is 804 in train. real_data adds 500 rows from the extra CSV, "
        "so internal_total_n becomes 1304. smote_cohort is fold-based, so its added volume varies per fold."
    )
    plt.figtext(0.5, 0.02, note, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    out = PLOTS_DIR / "strategy_internal_research_table.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_final_test(df: pd.DataFrame, cohort_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Final model - test vs cohort", fontsize=16, fontweight="bold")

    test_row = df.iloc[0]
    coh_row = cohort_df.iloc[0]

    axes[0].bar(["accuracy", "precision", "recall", "f1"], [test_row["accuracy"], test_row["precision"], test_row["recall"], test_row["f1"]], color="#4C78A8")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Global test metrics")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(["accuracy", "precision", "recall", "f1"], [coh_row["accuracy"], coh_row["precision"], coh_row["recall"], coh_row["f1"]], color="#E45756")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("internal_research test metrics")
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = PLOTS_DIR / "final_test_overview.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_augmentation_sizes(df: pd.DataFrame) -> None:
    order = ["baseline", "real_data", "smote"]
    df = df.copy()
    df["estrategia"] = pd.Categorical(df["estrategia"], categories=order, ordered=True)
    df = df.sort_values("estrategia").reset_index(drop=True)

    display_df = pd.DataFrame(
        {
            "strategy": df["estrategia"].astype(str),
            "global_n": df["global_n"].astype(int),
            "cohort_n": df["cohort_n"].astype(int),
            "global_f1": df["global_f1"].map(lambda x: f"{x:.3f}"),
            "cohort_rec": df["cohort_recall"].map(lambda x: f"{x:.3f}"),
            "cohort_f1": df["cohort_f1"].map(lambda x: f"{x:.3f}"),
        }
    )

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.axis("off")
    ax.set_title(
        "augmentation comparison - data volume and cohort size",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )

    table = ax.table(
        cellText=display_df.values,
        colLabels=[
            "strategy",
            "global n",
            "cohort n",
            "global f1",
            "cohort rec",
            "cohort f1",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    header_color = "#2F4B7C"
    row_colors = ["#F3F6FA", "#FFFFFF"]
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])

    plt.tight_layout()
    out = PLOTS_DIR / "augmentation_sizes_table.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_internal_research_breakdown() -> None:
    train = load_csv(Path("data/processed/train.csv"))
    extra = load_csv(Path("data/raw/extra_client_type_internal_research.csv"))

    train_mask = train["client_type"].eq("internal_research")
    extra_mask = extra["client_type"].eq("internal_research")

    train_subset = train.loc[train_mask]
    extra_subset = extra.loc[extra_mask]
    combined_n = len(train_subset) + len(extra_subset)

    rows = [
        {
            "source": "train original",
            "n": len(train_subset),
            "cancelled": int(train_subset["canceled_job"].sum()),
            "not_cancelled": int((1 - train_subset["canceled_job"]).sum()),
            "cancel_rate": train_subset["canceled_job"].mean(),
        },
        {
            "source": "extra csv",
            "n": len(extra_subset),
            "cancelled": int(extra_subset["canceled_job"].sum()),
            "not_cancelled": int((1 - extra_subset["canceled_job"]).sum()),
            "cancel_rate": extra_subset["canceled_job"].mean(),
        },
        {
            "source": "combined real_data",
            "n": combined_n,
            "cancelled": int(train_subset["canceled_job"].sum() + extra_subset["canceled_job"].sum()),
            "not_cancelled": int((1 - train_subset["canceled_job"]).sum() + (1 - extra_subset["canceled_job"]).sum()),
            "cancel_rate": (train_subset["canceled_job"].sum() + extra_subset["canceled_job"].sum()) / combined_n,
        },
    ]

    display_df = pd.DataFrame(rows)
    display_df["cancel_rate"] = display_df["cancel_rate"].map(lambda x: f"{x:.3f}")

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off")
    ax.set_title(
        "internal_research - original vs added data",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )

    table = ax.table(
        cellText=display_df.values,
        colLabels=["source", "n", "cancelled", "not_cancelled", "cancel_rate"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    header_color = "#2F4B7C"
    row_colors = ["#F3F6FA", "#FFFFFF"]
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])

    note = (
        "Note: n=804 is the original internal_research cohort in train. "
        "The extra CSV adds 500 more rows, so the real_data training cohort becomes 1304."
    )
    plt.figtext(0.5, 0.02, note, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    out = PLOTS_DIR / "augmentation_internal_research_breakdown.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_train_validation_by_strategy() -> None:
    fold_metrics_path = RESULTS_DIR / "strategy_fold_metrics.csv"
    if not fold_metrics_path.exists():
        return

    df = load_csv(fold_metrics_path)
    order = ["baseline", "real_data", "smote", "smote_cohort"]
    df["strategy"] = pd.Categorical(df["strategy"], categories=order, ordered=True)
    summary = (
        df.groupby("strategy", observed=False)[
            ["train_accuracy", "train_precision", "train_recall", "train_f1", "val_accuracy", "val_precision", "val_recall", "val_f1"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "strategy",
        "train_accuracy_mean",
        "train_accuracy_std",
        "train_precision_mean",
        "train_precision_std",
        "train_recall_mean",
        "train_recall_std",
        "train_f1_mean",
        "train_f1_std",
        "val_accuracy_mean",
        "val_accuracy_std",
        "val_precision_mean",
        "val_precision_std",
        "val_recall_mean",
        "val_recall_std",
        "val_f1_mean",
        "val_f1_std",
    ]
    summary = summary.dropna(subset=["strategy"]).reset_index(drop=True)
    strategies = summary["strategy"].astype(str).tolist()
    x = np.arange(len(strategies))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("Train vs validation by strategy", fontsize=16, fontweight="bold")

    panels = [
        ("Accuracy", "train_accuracy_mean", "train_accuracy_std", "val_accuracy_mean", "val_accuracy_std", "#4C78A8", "#F58518"),
        ("Precision", "train_precision_mean", "train_precision_std", "val_precision_mean", "val_precision_std", "#59A14F", "#E45756"),
        ("Recall", "train_recall_mean", "train_recall_std", "val_recall_mean", "val_recall_std", "#1F77B4", "#FF7F0E"),
        ("F1", "train_f1_mean", "train_f1_std", "val_f1_mean", "val_f1_std", "#2CA02C", "#D62728"),
    ]

    for ax, (title, train_m, train_s, val_m, val_s, train_c, val_c) in zip(axes.flat, panels):
        train_bars = ax.bar(x - width / 2, summary[train_m], width, yerr=summary[train_s], label="train", color=train_c, alpha=0.9, capsize=4)
        val_bars = ax.bar(x + width / 2, summary[val_m], width, yerr=summary[val_s], label="validation", color=val_c, alpha=0.9, capsize=4)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=20)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        add_bar_labels(ax, train_bars)
        add_bar_labels(ax, val_bars)

    plt.tight_layout()
    out = PLOTS_DIR / "strategy_train_validation.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_best_strategy_train_val_test() -> None:
    comparison_path = RESULTS_DIR / "strategy_comparison_cv.csv"
    fold_metrics_path = RESULTS_DIR / "strategy_fold_metrics.csv"
    final_test_path = RESULTS_DIR / "final_test_metrics.csv"

    if not comparison_path.exists() or not fold_metrics_path.exists() or not final_test_path.exists():
        return

    best_strategy = load_csv(comparison_path).iloc[0]["strategy"]
    fold_df = load_csv(fold_metrics_path)
    fold_df = fold_df[fold_df["strategy"] == best_strategy].copy()
    test_df = load_csv(final_test_path)
    test_row = test_df.iloc[0]

    summary = {
        "train_accuracy": fold_df["train_accuracy"].mean(),
        "train_precision": fold_df["train_precision"].mean(),
        "train_recall": fold_df["train_recall"].mean(),
        "train_f1": fold_df["train_f1"].mean(),
        "val_accuracy": fold_df["val_accuracy"].mean(),
        "val_precision": fold_df["val_precision"].mean(),
        "val_recall": fold_df["val_recall"].mean(),
        "val_f1": fold_df["val_f1"].mean(),
        "test_accuracy": float(test_row["accuracy"]),
        "test_precision": float(test_row["precision"]),
        "test_recall": float(test_row["recall"]),
        "test_f1": float(test_row["f1"]),
    }

    metrics = ["accuracy", "precision", "recall", "f1"]
    train_vals = [summary[f"train_{m}"] for m in metrics]
    val_vals = [summary[f"val_{m}"] for m in metrics]
    test_vals = [summary[f"test_{m}"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(f"Best strategy: {best_strategy} - train vs validation vs test", fontsize=16, fontweight="bold")

    bars_train = ax.bar(x - width, train_vals, width, label="train", color="#4C78A8")
    bars_val = ax.bar(x, val_vals, width, label="validation", color="#F58518")
    bars_test = ax.bar(x + width, test_vals, width, label="test", color="#E45756")

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    add_bar_labels(ax, bars_train)
    add_bar_labels(ax, bars_val)
    add_bar_labels(ax, bars_test)

    note = (
        "Train and validation are means over CV folds for the selected strategy. "
        "Test comes from the held-out split."
    )
    plt.figtext(0.5, 0.01, note, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out = PLOTS_DIR / "best_strategy_train_validation_test.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()

    cohort_df = load_csv(RESULTS_DIR / "cohort_metrics.csv")
    strategy_df = load_csv(RESULTS_DIR / "strategy_comparison_cv.csv")
    final_test_df = load_csv(RESULTS_DIR / "final_test_metrics.csv")
    final_cohort_df = load_csv(RESULTS_DIR / "final_test_cohort_metrics.csv")
    augmentation_df = load_csv(RESULTS_DIR / "augmentation_comparison.csv")

    for feature in cohort_df["cohort_feature"].unique():
        plot_cohort_feature(cohort_df, feature)

    plot_strategy_comparison(strategy_df)
    plot_strategy_focus_internal_research(strategy_df)
    plot_strategy_table_internal_research(strategy_df)
    plot_strategy_confusions(strategy_df)
    plot_strategy_deltas(strategy_df)
    plot_final_test(final_test_df, final_cohort_df)
    plot_augmentation_sizes(augmentation_df)
    plot_internal_research_breakdown()
    plot_train_validation_by_strategy()
    plot_best_strategy_train_val_test()

    print(f"Saved plots in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
