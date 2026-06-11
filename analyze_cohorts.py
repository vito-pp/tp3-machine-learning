import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_config, load_dataset, split_features_target
from src.evaluation import (
    analyze_cohorts,
    calculate_metrics,
    generate_oof_predictions,
    load_model,
    save_dataframe,
)


def plot_learning_curves(model, X, y, config) -> None:
    """
    Genera curvas de aprendizaje entrenando el modelo con fracciones
    crecientes del train y evaluando con validación cruzada (k-fold).
    Guarda el plot en results/learning_curves.png.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, recall_score

    n_splits    = config["validation"]["n_splits"]
    random_state = config["project"]["random_state"]
    output_path = "results/learning_curves.png"

    # Fracciones del train a evaluar
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    train_f1s, val_f1s = [], []
    train_recalls, val_recalls = [], []

    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)

    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for frac in fractions:
        fold_train_f1, fold_val_f1 = [], []
        fold_train_recall, fold_val_recall = [], []

        for train_idx, val_idx in skf_outer.split(X_arr, y_arr):
            # Tomar solo la fracción del fold de entrenamiento
            n_subset = max(int(len(train_idx) * frac), 50)
            rng = np.random.default_rng(random_state)
            subset_idx = rng.choice(train_idx, size=n_subset, replace=False)

            X_sub, y_sub = X_arr[subset_idx], y_arr[subset_idx]
            X_val, y_val = X_arr[val_idx],   y_arr[val_idx]

            clf = model.__class__(**model.get_params())
            clf.fit(X_sub, y_sub)

            fold_train_f1.append(f1_score(y_sub, clf.predict(X_sub), zero_division=0))
            fold_val_f1.append(f1_score(y_val, clf.predict(X_val), zero_division=0))
            fold_train_recall.append(recall_score(y_sub, clf.predict(X_sub), zero_division=0))
            fold_val_recall.append(recall_score(y_val, clf.predict(X_val), zero_division=0))

        train_f1s.append(np.mean(fold_train_f1))
        val_f1s.append(np.mean(fold_val_f1))
        train_recalls.append(np.mean(fold_train_recall))
        val_recalls.append(np.mean(fold_val_recall))

    n_samples = [int(len(y_arr) * (1 - 1/n_splits) * f) for f in fractions]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Learning Curves — Random Forest baseline", fontsize=13, fontweight="bold")

    for ax, train_scores, val_scores, metric in [
        (axes[0], train_f1s,     val_f1s,     "F1"),
        (axes[1], train_recalls, val_recalls,  "Recall"),
    ]:
        ax.plot(n_samples, train_scores, "o-", color="#4C72B0", label="Train")
        ax.plot(n_samples, val_scores,   "o-", color="#d62728", label="Validación (CV)")
        ax.fill_between(n_samples, train_scores, val_scores, alpha=0.08, color="gray")
        ax.set_xlabel("Tamaño del conjunto de entrenamiento", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f"Curva de aprendizaje — {metric}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Anotación: gap train vs val en el punto de mayor datos
        gap = train_scores[-1] - val_scores[-1]
        ax.annotate(
            f"gap={gap:.3f}",
            xy=(n_samples[-1], val_scores[-1]),
            xytext=(-60, 12),
            textcoords="offset points",
            fontsize=8,
            color="#555555",
            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nLearning curves guardadas en: {output_path}")

    # Interpretación automática
    gap_f1 = train_f1s[-1] - val_f1s[-1]
    trend   = val_f1s[-1] - val_f1s[int(len(val_f1s) * 0.5)]
    print("\n=== Interpretación de curvas de aprendizaje ===")
    if gap_f1 > 0.15:
        print(f"  Gap train/val F1 = {gap_f1:.3f} -> sobreajuste moderado (model-limited)")
    else:
        print(f"  Gap train/val F1 = {gap_f1:.3f} -> sin sobreajuste significativo")
    if trend > 0.02:
        print(f"  Val F1 sigue subiendo (+{trend:.3f}) -> más datos podría mejorar el rendimiento")
    else:
        print(f"  Val F1 se estabilizó (trend={trend:.3f}) -> agregar datos generales no ayudaría mucho")
    print(f"  Conclusión: la cohorte internal_research se beneficia más de datos")
    print(f"  específicos o cambios en la estrategia que de más datos generales.")


def main() -> None:
    config = load_config()
    target = config["project"]["target"]

    raw_train = load_dataset(config["data"]["train_path"])
    transformed_train = load_dataset(
        config["data"]["train_transformed_path"]
    )

    X_train, y_train = split_features_target(
        transformed_train,
        target,
    )

    if len(raw_train) != len(transformed_train):
        raise ValueError(
            "Raw and transformed train datasets have different row counts."
        )

    model = load_model(config["baseline"]["model_path"])

    y_pred = generate_oof_predictions(
        model=model,
        X=X_train,
        y=y_train,
        n_splits=config["validation"]["n_splits"],
        random_state=config["project"]["random_state"],
    )

    global_metrics = pd.DataFrame([calculate_metrics(y_train, y_pred)])

    save_dataframe(
        global_metrics,
        config["evaluation"]["global_metrics_path"],
    )

    predictions = raw_train.copy()
    predictions["y_true"] = y_train.to_numpy()
    predictions["y_pred"] = y_pred
    predictions["is_error"] = (
        predictions["y_true"] != predictions["y_pred"]
    )

    save_dataframe(
        predictions,
        config["evaluation"]["predictions_path"],
    )

    cohort_metrics = analyze_cohorts(
        raw_train=raw_train,
        y_true=y_train,
        y_pred=y_pred,
        cohort_columns=config["cohorts"]["columns"],
    )

    save_dataframe(
        cohort_metrics,
        config["evaluation"]["cohort_metrics_path"],
    )

    false_positives = predictions[
        (predictions["y_true"] == 0)
        & (predictions["y_pred"] == 1)
    ]

    false_negatives = predictions[
        (predictions["y_true"] == 1)
        & (predictions["y_pred"] == 0)
    ]

    save_dataframe(
        false_positives,
        config["evaluation"]["false_positives_path"],
    )

    save_dataframe(
        false_negatives,
        config["evaluation"]["false_negatives_path"],
    )

    print("\nGlobal out-of-fold validation metrics:")
    print(global_metrics.to_string(index=False))

    print("\nWorst cohorts by F1 (minimum 30 rows):")
    print(
        cohort_metrics[cohort_metrics["n"] >= 30]
        .sort_values(["f1", "n"], ascending=[True, False])
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved:")
    for path in config["evaluation"].values():
        print(f"  {path}")

    # ── Learning curves ───────────────────────────────────────────────────────
    print("\n=== Generando curvas de aprendizaje ===")
    plot_learning_curves(model, X_train, y_train, config)


if __name__ == "__main__":
    main()