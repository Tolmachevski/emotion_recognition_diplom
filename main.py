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



# Начало программы
if __name__ == '__main__':
    # ============================================
    # ШАГ 1: Загрузка датасета Души
    # ============================================
    print()
    print('=' * 60)
    print('Обучение модели')
    print('=' * 60)

    #path, emotions = load_dusha()

    #print('\n ИТОГ...')
    #print(f"    Сумма аудиофайлов для обучения - {len(path)}")
    #print(f"    Сумма эмоций - {len(emotions)}")



    print(f"\n📦 ШАГ 3: Загрузка датасета ДУША")

    X_dusha_train, y_dusha_train = load_dusha_dataset(
        dusha_dir=DUSHA_DIR,
        subset='crowd_train', # Из какой папки берем данные
        sample_size=500 # Берем только 500 файлов из датасета
    )

    if X_dusha_train is None:
        print("ОШИБКА ЗАГРУЗКИ ДУШИ, ПРОВЕРЬ ПУТИ!")



    X_dusha_train_features, y_dusha_train = extract_features_parrallel(X_dusha_train, y_dusha_train)
    print("Извлечение признаков выполнено!")
    if not validate_extracted_features(X_dusha_train_features, y_dusha_train, expected_dims=29):
        raise RuntimeError("Валидация признаков провалена. Остановка обучения.")


    print(f"Тип признаков - {type(X_dusha_train_features)}")
    print(f"Тип эмоций - {type(y_dusha_train)}")


    print("\n📊 Детальная статистика по признакам (первые 5 и последние 3):")
    feat_names = ([f"mfcc{i}_m" for i in range(1,14)] + 
                [f"mfcc{i}_s" for i in range(1,14)] + 
                ["rms", "zcr", "centroid"])

    # Проверяем только несколько ключевых столбцов
    check_indices = [0, 1, 13, 26, 27, 28]  # mfcc1_m, mfcc1_s, mfcc13_s, rms, zcr, centroid

    for idx in check_indices:
        col = X_dusha_train_features[:, idx]
        print(f"{feat_names[idx]:10s} | min: {col.min():7.2f} | max: {col.max():7.2f} | mean: {col.mean():7.2f} | std: {col.std():7.2f}")





    exit()


    # ============================================
    # ШАГ 2: Разделение на тренировочную и тестовую выборки
    # ============================================
    print("\n ШАГ 2: Разделение на train/test...")

    X_train, X_test, y_train, y_test = train_test_split(
        path, # пути к аудио
        emotions, # Эмоции
        test_size=0.2, # Тестовых файлов будет 20%, а 80% в обучении
        random_state=42, # Фиксируем "Случайность при воспроизведении"
        stratify=emotions # Сохраняем пропорцию классов
    )

    print(f"    Тренировочные {len(X_train)} файлов")
    print(f"    Тестовые {len(X_test)} файлов")



    print(f"ПЕРЕД ИЗВЛЕЧЕНИЕМ ПРИЗНАКОВ ИМЕЕМ: {type(X_train)}")
    print(f"ПЕРЕД ИЗВЛЕЧЕНИЕМ ПРИЗНАКОВ ИМЕЕМ: {type(y_train)}")
    # ============================================
    # ШАГ 3: Аугментация данных (пропускаем, и так большая выборка)
    # ============================================


    # ============================================
    # ШАГ 4: Извлечение признаков
    # ============================================

    X_train_features, y_train = extract_mfcc_parralel(X_train, y_train)
    X_test_features, y_test = extract_mfcc_parralel(X_test, y_test)
    print("Извлечение признаков выполнено!")

    # ============================================
    # ШАГ 5: Нормализация
    # ============================================
    print("\n ШАГ 5: Нормализация...")


    print(f"ПЕРЕД НОРМАЛИЗАЦИЕЙ ИМЕЕМ: {type(X_train_features)}")
    print(f"ПЕРЕД НОРМАЛИЗАЦИЕЙ ИМЕЕМ: {type(X_test_features)}")

    scaler = StandardScaler() # Создаем инструмент для масштабирования
    X_train_scaled = scaler.fit_transform(X_train_features)
    # fit() - изучает тренировочные данные: считает среднее и std для каждого признака
    # transform() - применяет формулу к тренировочным данным
    X_test_scaled = scaler.transform(X_test_features) # Применяет те же самые средние и std, но не пересчитывает заного для теста
    # fit() применяем только для ТРЕНИРОВОЧНЫХ данных, поскольку если применим - модель запомнит их, и на выходе мы получим "ложные" хорошие результаты
    # Это как при обучении ученика дать ему материалы экзамена, а потом протестировать ученика на этом же экзамене


    print("Нормализация выполнена!")


    assert len(X_train_scaled) == len(y_train), "Рассинхрон признаков и меток!"
    assert X_train_scaled.shape[1] == 26, f"Ожидал 26 признаков, получил {X_train_scaled.shape[1]}"
    print(f"✅ Данные готовы: {X_train_scaled.shape}, меток: {len(y_train)}")



    print(" ПРОВЕРКА БАЛАНСА КЛАССОВ")
    check_class_balance_pd(y_train, "(перед обучением)")

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

    grid.fit(X_train_scaled, y_train)
    print("\n📊 Оценка на тесте (с учётом дисбаланса):")
    y_pred = grid.best_estimator_.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=sorted(np.unique(y_test))))



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



