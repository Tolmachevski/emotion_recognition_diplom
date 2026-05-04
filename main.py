# Главный файл

import os # Библиотека, позволяет управлять файлами, папками, процессами и переменными окружения, обеспечивая переносимость кода на разные ОС
import numpy as np # библиотека для работы с многомерными массивами, матрицами и выполнения сложных мат вычислений
import pandas as pd # Библиотека для обработки и анализа структурированных данных (таблицы, временные ряды)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score # функция, позволяющая случайно разделить данные на обучающею и тестовую выборки, а так же сетка для подбора параметров
from sklearn.preprocessing import StandardScaler # инструмент для стандартизации признаков 
from sklearn.svm import SVC # Класс, реализующий метод опорных векторов
from sklearn.metrics import classification_report, confusion_matrix # используются для оценки качества моделей ML 
import joblib # Библиотека для эффект. работы с тяжелыми данными, парралельных вичислений и быстрого сохранения(серилизации) моделей ML 
import json # Библиотека для работы с json-файлами
import librosa # Библиотека для анализа музыки и звука, извлечения признаков, визуализации и обработки аудиоданных
import warnings
import sys
from tqdm import tqdm
import lightgbm as lgb
import time
import optuna




from portable_ffmpeg import add_to_path
add_to_path()
from augmentation_audio import augmentation_audio, load_audio_parallel
from balancer_class import check_class_balance_pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from loader import load_dusha_dataset
from extractor import extract_features_parrallel, validate_extracted_features
from utils import load_extracted_features, save_extracted_features

# Настройка PySoundFile
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')

from load_all_data import load_all_datasets # Импортируем загрузчик датасетов


RAVDESS_DIR = r'E:\diplom_ser\data\ravdess' # Датасет Ravdess
MY_DATA_DIR = r'E:\diplom_ser\data\Actor_25_me' # Мой датасет
DUSHA_DIR = r'E:\diplom_ser\data\dusha' # Датасет Души
SAVE_DIR = r'E:\diplom_ser\model' # путь сохранения модели
DUSHA_DIR_TRAINED_DATA = r'E:\diplom_ser\data\dusha\crowd\crowd_train\wavs'


# Пути для кэша
CACHE_DIR = r'E:\diplom_ser\processed'
train_cache = os.path.join(CACHE_DIR, 'dusha_crowd_train_features.npz')
test_cache = os.path.join(CACHE_DIR, 'dusha_crowd_test_features.npz')



# Начало программы
if __name__ == '__main__':
    # ============================================
    # ШАГ 1: Загрузка датасета Души
    # ============================================
    print()
    print('=' * 60)
    print('Обучение модели')
    print('=' * 60)



    print(f"\n  Загрузка датасета ДУША")

    # Загружаем тренировочные данные
    X_v_tr, y_v_tr = load_dusha_dataset(
        dusha_dir=DUSHA_DIR,
        subset='crowd_train' # Из какой папки берем данные
        # sample_size=500 # Берем только 500 файлов из датасета
    )

    # Загружаем тестовые данные
    X_v_ts, y_v_ts = load_dusha_dataset(
        dusha_dir=DUSHA_DIR,
        subset='crowd_test' # Из какой папки берем данные
        #csample_size=500 # Берем только 500 файлов из датасета
    )



    # Если не смогли загрузить тренировочную выборку
    if X_v_tr is None:
        print("ОШИБКА ЗАГРУЗКИ ДУШИ, ПРОВЕРЬ ПУТИ!")

    # Если не смогли загрузить тестовую выборку
    if X_v_ts is None:
        print("ОШИБКА ЗАГРУЗКИ ДУШИ, ПРОВЕРЬ ПУТИ!")




    # ============================================
    # ШАГ 3: Аугментация данных (пропускаем, и так большая выборка)
    # ============================================


    # ============================================
    # ШАГ 4: Извлечение признаков (пропускаем)
    # ============================================

    feat_names = ([f"mfcc{i}_m" for i in range(1,14)] + 
              [f"mfcc{i}_s" for i in range(1,14)] + 
              ["rms", "zcr", "centroid"])

    # Извлекаем признаки тренировочной выборки и сразу сохранияем их в кеш или сразу загружаем
    if os.path.exists(train_cache):
        print("🔄 Загружаем признаки (train)...")
        X_v_tr_features, y_v_tr_emotions, _, _, _ = load_extracted_features(train_cache)
    else:
        print("🔄 Извлекаем признаки (train)...")
        X_v_tr_features, y_v_tr_emotions = extract_features_parrallel(X_v_tr, y_v_tr)
        print("Извлечение признаков выполнено!")
        if not validate_extracted_features(X_v_tr_features, y_v_tr_emotions, expected_dims=29):
            raise RuntimeError("Валидация train провалена")
        save_extracted_features(X_v_tr_features, y_v_tr_emotions, feat_names, None, train_cache)


    # Извлекаем признаки тестовой выборки и сразу сохранияем их в кеш или сразу загружаем
    if os.path.exists(test_cache):
        print("🔄 Загружаем признаки (test)...")
        X_v_ts_features, y_v_ts_emotions, _, _, _ = load_extracted_features(test_cache)
    else:
        print("🔄 Извлекаем признаки (train)...")
        X_v_ts_features, y_v_ts_emotions = extract_features_parrallel(X_v_ts, y_v_ts)
        print("Извлечение признаков выполнено!")
        if not validate_extracted_features(X_v_ts_features, y_v_ts_emotions, expected_dims=29):
            raise RuntimeError("Валидация признаков провалена. Остановка обучения.")
        save_extracted_features(X_v_ts_features, y_v_ts_emotions, feat_names, None, test_cache)

    print("✅ Признаки готовы")

    

    # ============================================
    # ШАГ 5: Нормализация
    # ============================================
    print("\n ШАГ 5: Нормализация...")


    print(f"ПЕРЕД НОРМАЛИЗАЦИЕЙ ИМЕЕМ: {type(X_v_tr_features)}")
    print(f"ПЕРЕД НОРМАЛИЗАЦИЕЙ ИМЕЕМ: {type(X_v_ts)}")

    scaler = StandardScaler() # Создаем инструмент для масштабирования
    X_v_tr_features_scld = scaler.fit_transform(X_v_tr_features)
    # fit() - изучает тренировочные данные: считает среднее и std для каждого признака
    # transform() - применяет формулу к тренировочным данным
    X_v_ts_features_scld = scaler.transform(X_v_ts_features) # Применяет те же самые средние и std, но не пересчитывает заного для теста
    # fit() применяем только для ТРЕНИРОВОЧНЫХ данных, поскольку если применим - модель запомнит их, и на выходе мы получим "ложные" хорошие результаты
    # Это как при обучении ученика дать ему материалы экзамена, а потом протестировать ученика на этом же экзамене


    # Сохраняем скалер для будущего инференса
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
    print(f"💾 Скалер сохранён: {os.path.join(SAVE_DIR, 'scaler.pkl')}")

    print("Нормализация выполнена!")


    assert len(X_v_tr_features_scld) == len(y_v_tr_emotions), "Рассинхрон признаков и меток!"
    assert X_v_tr_features_scld.shape[1] == 29, f"Ожидал 29 признаков, получил {X_v_tr_features_scld.shape[1]}"
    print(f"✅ Данные готовы: {X_v_tr_features_scld.shape}, меток: {len(y_v_tr_emotions)}")


    # --- После загрузки из кэша (для train) ---
    print(f"\n🔍 DEBUG TRAIN DATA:")
    print(f"   X type: {type(X_v_tr_features)}, dtype: {getattr(X_v_tr_features, 'dtype', 'N/A')}, shape: {getattr(X_v_tr_features, 'shape', 'N/A')}")
    print(f"   y type: {type(y_v_tr_emotions)}, unique: {np.unique(y_v_tr_emotions) if hasattr(y_v_tr_emotions, '__len__') else y_v_tr_emotions}")
    print(f"   X sample [0]: {X_v_tr_features[0] if hasattr(X_v_tr_features, '__getitem__') else 'N/A'}")
    print(f"   NaN in X: {np.isnan(X_v_tr_features).any() if hasattr(X_v_tr_features, 'any') else 'N/A'}")

    
    print(" ПРОВЕРКА БАЛАНСА КЛАССОВ")
    check_class_balance_pd(y_v_tr_emotions, "(перед обучением)")

    # ============================================
    # ШАГ 6: Обучение модели
    # ============================================

    print("\n Начинаем обучать модель по методу LightGBM...")

    # Логика обучения
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
    

    print("🚀 Запуск Optuna (поиск лучших параметров)...")
    print("⏳ Это займёт время, но меньше, чем GridSearch. Следи за значениями Objective.")

    start_time = time.time()

    # Создаем исследование 
    study = optuna.create_study(direction='maximize')

    # Запускаем 50 попыток (trials)
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    elapsed = time.time() - start_time
    print(f"Оптимизация завершена за {elapsed:.1f} секунд.")
    print(f"Лучший f1-macro - {study.best_value:.4f}")
    print(f"лучшие параметры: {study.best_params}")


    # Финальное обучение модели на лучших параметрах
    best_model = lgb.LGBMClassifier(**study.best_params, class_weight='balanced', random_state=42)
    best_model.fit(X_v_tr_features_scld, y_v_tr_emotions)



    # Оценка на тестовых данных
    y_pred = best_model.predict(X_v_ts_features_scld)
    print("\n📊 Результат на тесте:")
    print(classification_report(y_v_ts_emotions, y_pred, target_names=['angry', 'happy', 'neutral', 'sad']))

    





















    # Обучение по SVM алгоритму
    """
    
    print("\n ШАГ 8: Обучение модели SVM...")
    print("\n Перед обучением модели подберем лучшие параметры")
    print("\n ШАГ 8.1: Подбор лучших парамеров для обучения модели")


    # Сетка параметров (БОЛЬШЕ вариантов для gamma!)
    param_grid = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.01],
        'max_depth': [4, 6, 8],
        'num_leaves': [31, 63, 127]
    }

    print("🔄 Подбор параметров (LightGBM)...")
    start = time.time()

    # Создаём GridSearchCV
    grid = GridSearchCV(
        lgb.LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1, force_col_wise=True),
        param_grid, cv=5, scoring='f1_macro', n_jobs=-1
    )


    print("🔄 Подбор параметров...")
    grid.fit(X_v_tr_features_scld, y_v_tr_emotions)
    print("\n📊 Оценка на тесте (с учётом дисбаланса):")
    y_pred = grid.best_estimator_.predict(X_v_ts_features_scld)
    print(classification_report(y_v_ts_emotions, y_pred, target_names=sorted(np.unique(y_v_ts_emotions))))

    
    """




    


