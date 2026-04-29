import pandas as pd


# проверка баланса классов
def check_class_balance_pd(labels, stage_name=""):
    df = pd.Series(labels).value_counts().sort_index()
    print(f"\n📊 Баланс классов {stage_name}:")
    print(df.to_frame(name='count').assign(percent=lambda x: x['count']/x['count'].sum()*100).round(1))
    
    if len(df) > 1:
        ratio = df.max() / df.min()
        print(f"\n   ⚠️  Разброс: {ratio:.2f}x")
        if ratio > 3:
            print("   ❗ Рекомендация: добавить class_weight='balanced' или использовать SMOTE")
    print("-" * 40)