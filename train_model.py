import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

def train():
    print("🔄 Загрузка датасета 'ml_dataset.csv'...")
    try:
        df = pd.read_csv('ml_dataset.csv')
    except FileNotFoundError:
        print("❌ Ошибка: Файл ml_dataset.csv не найден!")
        return

    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    print(f"📊 Всего транзакций: {len(df)}, из них фрода: {y.sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🧠 Обучение модели Random Forest (Случайный лес)...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=7, 
        random_state=42,
        class_weight='balanced' 
    )
    model.fit(X_train, y_train)

    print("\n✅ Модель обучена! Сдаем экзамен на тестовых данных:")
    predictions = model.predict(X_test)
    
    print("-" * 50)
    print(classification_report(y_test, predictions, labels=[0, 1], target_names=['Normal', 'Fraud'], zero_division=0))
    print("-" * 50)


    print("\n🏆 Важность эвристик (Feature Importance):")
    importances = model.feature_importances_
    features = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    for name, imp in features:
        print(f"  - {name:30}: {imp:.1%}")

    with open('fraud_ml_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\n💾 Мозг успешно сохранен в файл 'fraud_ml_model.pkl'")

if __name__ == "__main__":
    train()