from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Быстро и надёжно для анализа
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Генерация имён признаков (должна точно соответствовать порядку в np.concatenate)
feat_names = (
    [f"mfcc{i+1}_mean" for i in range(13)] +
    [f"mfcc{i+1}_std" for i in range(13)] +
    ["rms", "zcr", "centroid"]
)

print(f"📋 Всего признаков: {len(feat_names)}")

# 1. Сплит (стратификация обязательна!)
X_train, X_test, y_train, y_test = train_test_split(
    X_dusha_train_features, y_dusha_train,
    test_size=0.2, random_state=42, stratify=y_dusha_train
)

# 2. Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Быстрое обучение (RandomForest для анализа важности)
print("\n🔄 Обучаем модель для анализа важности...")
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# 4. Вывод важности признаков
importance = pd.DataFrame({
    'feature': feat_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🏆 Топ-15 самых важных признаков:")
print(importance.head(15).to_string(index=False))

print(f"\n📉 Признаки с важностью < 0.01 ({(importance['importance'] < 0.01).sum()} шт.):")
print(importance[importance['importance'] < 0.01]['feature'].tolist())

# 5. Визуализация (опционально, для диплома)
plt.figure(figsize=(10, 8))
plt.barh(importance['feature'][:20][::-1], importance['importance'][:20][::-1])
plt.xlabel('Важность признака')
plt.title('Топ-20 признаков по важности (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("💾 График сохранён: feature_importance.png")