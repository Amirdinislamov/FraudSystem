import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np


def train():
    print("🔄 Загрузка датасета 'ml_dataset.csv'...")
    try:
        df = pd.read_csv("ml_dataset.csv")
    except FileNotFoundError:
        print("❌ Ошибка: Файл ml_dataset.csv не найден!")
        print("   Запустите сначала: python simulator.py")
        return

    feature_names = [col for col in df.columns if col != "is_fraud"]
    X = df[feature_names]
    y = df["is_fraud"]

    print(f"📊 Транзакций: {len(df)} | Фрод: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"📐 Признаков: {len(feature_names)}: {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🧠 Обучение Random Forest...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("📋 ОТЧЁТ О КАЧЕСТВЕ МОДЕЛИ")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"], zero_division=0))

    auc = roc_auc_score(y_test, y_proba)
    print(f"🎯 ROC-AUC Score: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📊 Матрица ошибок:")
    print(f"   True Negative  (норма верно):   {tn}")
    print(f"   False Positive (ложная тревога): {fp}")
    print(f"   False Negative (пропущен фрод):  {fn}")
    print(f"   True Positive  (фрод пойман):    {tp}")

    print("\n🏆 Важность признаков (Feature Importance):")
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"  {name:35} {imp:.3f} {bar}")

    print("\n🔁 Кросс-валидация (5-fold, метрика: F1)...")
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="f1", n_jobs=-1)
    print(f"   F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    saved = {
        "model": model,
        "feature_names": feature_names,
    }
    with open("fraud_ml_model.pkl", "wb") as f:
        pickle.dump(saved, f)

    print(f"\n💾 Модель сохранена в 'fraud_ml_model.pkl'")
    print(f"   Признаки зафиксированы: {feature_names}")


if __name__ == "__main__":
    train()