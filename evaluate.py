"""
Генератор отчёта о качестве ML-модели для дипломной работы.

Читает ml_dataset.csv и обученную модель fraud_ml_model.pkl, считает
метрики и сохраняет в папку reports/:

    - reports/roc_curve.png         — ROC-кривая с AUC
    - reports/pr_curve.png          — Precision–Recall кривая с AP
    - reports/confusion_matrix.png  — матрица ошибок (heatmap)
    - reports/feature_importance.png— важность 13 признаков
    - reports/score_distribution.png— распределение risk-score по классам
    - reports/threshold_tuning.png  — precision/recall/F1 в зависимости от порога
    - reports/metrics.md            — итоговая таблица для диплома

Все графики готовы к вставке в Word-документ диплома.

Запуск:    python3 evaluate.py
"""

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

REPORTS_DIR = Path("reports")
DATASET_PATH = "ml_dataset.csv"
MODEL_PATH = "fraud_ml_model.pkl"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def load_data_and_model():
    if not os.path.exists(DATASET_PATH):
        raise SystemExit(f"❌ Не найден {DATASET_PATH}. Запустите simulator.py.")
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"❌ Не найден {MODEL_PATH}. Запустите train_model.py.")

    df = pd.read_csv(DATASET_PATH)
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    if isinstance(saved, dict):
        model = saved["model"]
        feature_names = saved["feature_names"]
    else:
        model = saved
        feature_names = [c for c in df.columns if c != "is_fraud"]

    X = df[feature_names]
    y = df["is_fraud"]
    return df, model, feature_names, X, y


def plot_roc(y_true, y_proba, out: Path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Случайный классификатор")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая Random Forest")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return roc_auc


def plot_pr(y_true, y_proba, out: Path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR (AP = {ap:.4f})")
    baseline = y_true.mean()
    ax.axhline(baseline, color="grey", linestyle="--", lw=1,
               label=f"Baseline (доля fraud = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall кривая")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return ap


def plot_confusion(y_true, y_pred, out: Path):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Normal", "Fraud"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Фактически")
    ax.set_title("Матрица ошибок")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return cm


def plot_feature_importance(model, feature_names, out: Path):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = list(zip(feature_names, model.feature_importances_))
    importances.sort(key=lambda x: x[1])
    names, vals = zip(*importances)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(names, vals, color=sns.color_palette("viridis", len(names)))
    ax.set_xlabel("Feature Importance")
    ax.set_title("Вклад правил в решение модели")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return list(zip(names[::-1], vals[::-1]))


def plot_score_distribution(y_true, y_proba, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fraud_scores = y_proba[y_true == 1]
    normal_scores = y_proba[y_true == 0]
    bins = np.linspace(0, 1, 41)
    ax.hist(normal_scores, bins=bins, alpha=0.6, label=f"Normal (n={len(normal_scores)})", color="#2ca02c")
    ax.hist(fraud_scores, bins=bins, alpha=0.6, label=f"Fraud (n={len(fraud_scores)})", color="#d62728")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Количество транзакций (log)")
    ax.set_title("Распределение предсказанной вероятности")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_threshold_tuning(y_true, y_proba, out: Path):
    thresholds = np.linspace(0.05, 0.95, 91)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    best_idx = int(np.argmax(f1s))
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.plot(thresholds, f1s, label="F1", lw=2)
    ax.axvline(best_t, linestyle="--", color="grey",
               label=f"argmax F1 = {best_f1:.3f} @ t={best_t:.2f}")
    ax.set_xlabel("Порог классификации")
    ax.set_ylabel("Метрика")
    ax.set_title("Чувствительность модели к порогу")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return best_t, best_f1


def write_metrics_md(out: Path, info: dict):
    lines = [
        "# Метрики итоговой ML-модели",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| Размер датасета | {info['n_total']} транзакций |",
        f"| Доля fraud | {info['fraud_rate']:.2%} |",
        f"| Размер train / test | {info['n_train']} / {info['n_test']} |",
        f"| ROC-AUC | **{info['roc_auc']:.4f}** |",
        f"| Average Precision | **{info['ap']:.4f}** |",
        f"| Test F1 (fraud) | {info['test_f1']:.4f} |",
        f"| Test Precision (fraud) | {info['test_precision']:.4f} |",
        f"| Test Recall (fraud) | {info['test_recall']:.4f} |",
        f"| Best F1 @ optimal threshold | {info['best_f1']:.4f} @ t={info['best_t']:.2f} |",
        f"| Cross-validation F1 (5-fold) | {info['cv_f1_mean']:.4f} ± {info['cv_f1_std']:.4f} |",
        "",
        "## Confusion matrix (test, threshold = 0.5)",
        "",
        "| | Predicted Normal | Predicted Fraud |",
        "|---|---|---|",
        f"| Actual Normal | {info['cm'][0][0]} | {info['cm'][0][1]} |",
        f"| Actual Fraud | {info['cm'][1][0]} | {info['cm'][1][1]} |",
        "",
        "## Feature importance",
        "",
        "| Признак | Важность |",
        "|---|---|",
    ]
    for name, val in info["feature_importance"]:
        lines.append(f"| `{name}` | {val:.4f} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    df, model, feature_names, X, y = load_data_and_model()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    y_pred = (y_proba >= 0.5).astype(int)

    print("📊 Считаем метрики и строим графики...")
    roc_auc = plot_roc(y_test, y_proba, REPORTS_DIR / "roc_curve.png")
    ap = plot_pr(y_test, y_proba, REPORTS_DIR / "pr_curve.png")
    cm = plot_confusion(y_test, y_pred, REPORTS_DIR / "confusion_matrix.png")
    fi = plot_feature_importance(model, feature_names, REPORTS_DIR / "feature_importance.png")
    plot_score_distribution(y_test.values, y_proba, REPORTS_DIR / "score_distribution.png")
    best_t, best_f1 = plot_threshold_tuning(y_test, y_proba, REPORTS_DIR / "threshold_tuning.png")

    print("🔁 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                scoring="f1", n_jobs=-1)

    info = {
        "n_total": len(df),
        "fraud_rate": float(y.mean()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "roc_auc": roc_auc,
        "ap": ap,
        "test_f1": f1_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "best_t": best_t,
        "best_f1": best_f1,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "cm": cm.tolist(),
        "feature_importance": fi or [],
    }
    write_metrics_md(REPORTS_DIR / "metrics.md", info)

    print("\n" + "=" * 60)
    print("📋 ИТОГИ")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"], zero_division=0))
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"Best F1 @ t={best_t:.2f}: {best_f1:.4f}")
    print(f"CV F1 (5-fold):    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n📁 Графики сохранены в {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
