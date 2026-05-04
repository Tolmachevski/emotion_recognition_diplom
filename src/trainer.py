# Логика обучения

import optuna
import lightgbm as lgb
import time
from sklearn.model_selection import cross_val_score
import joblib


def objective(trial):
    # Предлагаем параметры модели
    param = {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'verbosity': -1,


        # Параметры, которые подбираем
        'n_estimators': trial.suggest_int('n_estimators', 100, 600), # кол-во деревьев
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True), # Шаг коррекции ошибок
        'num_leaves': trial.suggest_int('num_leaves', 20, 150), # Кол-во листьев в деревьях
        'max_depth': trial.suggest_int('max_depth', 4, 8), # Глубина дерева
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200), # Минимум обьектов в листе
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9), # Доля признаков на итерации
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9), # Доля обьектов на итерации
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }


    # Создаем модель
    model = lgb.LGBMClassifier(**param, random_state=42, class_weight='balanced')


    # Запускаем кросс-валидацию с early-stopping, optuna сам решит, останавливать ли процесс, если score падает
    score = cross_val_score(
        model,
        X_v_tr_features_scld, 
        y_v_tr_emotions,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )
    return score.mean()