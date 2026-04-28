# Это - отдельный файл для обучения модели с датасетом "Душа"

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
from tqdm import tqdm
from portable_ffmpeg import add_to_path
add_to_path()
from augmentation_audio import augmentation_audio, load_audio_parallel
from extracter_features import extract_mfcc_parralel, extract_mfcc_parralel_v05



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
    # ШАГ 1: Загрузка всех датасетов
    # ============================================
    print()
    print('=' * 60)
    print('Обучение модели')
    print('=' * 60)

    path, emotions = load_all_datasets()

    print('\n ИТОГ...')
    print(f"    Сумма аудиофайлов для обучения - {len(path)}")
    print(f"    Сумма эмоций - {len(emotions)}")



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

    X_train_features, y_test = extract_mfcc_parralel(X_train, y_train)
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
