# Главный файл

import os # Библиотека, позволяет управлять файлами, папками, процессами и переменными окружения, обеспечивая переносимость кода на разные ОС
import numpy as np # библиотека для работы с многомерными массивами, матрицами и выполнения сложных мат вычислений
import pandas as pd # Библиотека для обработки и анализа структурированных данных (таблицы, временные ряды)
from sklearn.model_selection import train_test_split, GridSearchCV # функция, позволяющая случайно разделить данные на обучающею и тестовую выборки, а так же сетка для подбора параметров
from sklearn.preprocessing import StandardScaler # инструмент для стандартизации признаков 
from sklearn.svm import SVC # Класс, реализующий метод опорных векторов
from sklearn.metrics import classification_report, confusion_matrix # используются для оценки качества моделей ML 
import joblib # Библиотека для эффект. работы с тяжелыми данными, парралельных вичислений и быстрого сохранения(серилизации) моделей ML 
import json # Библиотека для работы с json-файлами
import librosa # Библиотека для анализа музыки и звука, извлечения признаков, визуализации и обработки аудиоданных
import warnings
import sys
from tqdm import tqdm
from portable_ffmpeg import add_to_path
add_to_path()
from augmentation_audio import augmentation_audio, load_audio_parallel
from balancer_class import check_class_balance_pd
from utils import load_extracted_features, save_extracted_features

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from loader import load_dusha_dataset
from extractor import extract_features_parrallel, validate_extracted_features

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


    """
    print(f"Тип признаков тренировочных данных - {type(X_v_tr_features)}")
    print(f"Тип эмоций тренировочных данных - {type(y_v_tr_emotions)}")

    print(f"Тип признаков тестовых данных - {type(X_v_ts_features)}")
    print(f"Тип эмоций тестовых данных - {type(y_v_ts_emotions)}")


    print("\n📊 Детальная статистика по признакам (первые 5 и последние 3):")
    feat_names = ([f"mfcc{i}_m" for i in range(1,14)] + 
                [f"mfcc{i}_s" for i in range(1,14)] + 
                ["rms", "zcr", "centroid"])

    # Проверяем только несколько ключевых столбцов
    check_indices = [0, 1, 13, 26, 27, 28]  # mfcc1_m, mfcc1_s, mfcc13_s, rms, zcr, centroid

    for idx in check_indices:
        col = X_v_tr_features[:, idx]
        print(f"{feat_names[idx]:10s} | min: {col.min():7.2f} | max: {col.max():7.2f} | mean: {col.mean():7.2f} | std: {col.std():7.2f}")

    for idx in check_indices:
        col = X_v_ts_features[:, idx]
        print(f"{feat_names[idx]:10s} | min: {col.min():7.2f} | max: {col.max():7.2f} | mean: {col.mean():7.2f} | std: {col.std():7.2f}")
    
    """
    

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




    exit()



    
    print(" ПРОВЕРКА БАЛАНСА КЛАССОВ")
    check_class_balance_pd(y_v_tr_emotions, "(перед обучением)")

    # ============================================
    # ШАГ 6: Обучение модели
    # ============================================
    print("\n ШАГ 8: Обучение модели SVM...")
    print("\n Перед обучением модели подберем лучшие параметры")
    print("\n ШАГ 8.1: Подбор лучших парамеров для обучения модели")


    # Сетка параметров (БОЛЬШЕ вариантов для gamma!)
    param_grid = {
        'C': [1, 2, 3, 4, 5, 6],
        'gamma': [0.01, 0.017, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'kernel': ['rbf']
    }

    # Создаём GridSearchCV
    grid = GridSearchCV(
        SVC(random_state=42, class_weight='balanced'),  # 1. Баланс весов
        param_grid, 
        cv=5, 
        scoring='f1_macro',  # 2. Честная метрика
        n_jobs=-1, 
        verbose=2
    )

    grid.fit(X_v_tr_features_scld, y_v_tr_emotions)
    print("\n📊 Оценка на тесте (с учётом дисбаланса):")
    y_pred = grid.best_estimator_.predict(X_v_ts_features_scld)
    print(classification_report(y_v_ts_emotions, y_pred, target_names=sorted(np.unique(y_v_ts_emotions))))


    """
    # 1. Результаты подбора
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ПОДБОРА ПАРАМЕТРОВ")
    print("=" * 60)
    print(f"✅ Лучшие параметры: {grid.best_params_}")
    print(f"✅ Лучшая точность (CV): {grid.best_score_:.2%}")

    # 2. Оценка НА ТЕСТЕ (самое важное!)
    print("\n📊 Оценка на тестовой выборке:")
    y_pred = grid.best_estimator_.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    """
    



