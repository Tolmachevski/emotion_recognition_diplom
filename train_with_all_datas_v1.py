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

# Настройка PySoundFile
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

from load_all_data import load_all_datasets # Импортируем загрузчик датасетов


RAVDESS_DIR = r'E:\diplom_ser\data\ravdess' # Датасет Ravdess
MY_DATA_DIR = r'E:\diplom_ser\data\Actor_25_me' # Мой датасет
DUSHA_DIR = r'E:\diplom_ser\data\dusha' # Датасет Души
SAVE_DIR = r'E:\diplom_ser\model' # путь сохранения модели
DUSHA_DIR_TRAINED_DATA = r'E:\diplom_ser\data\dusha\crowd\crowd_train\wavs'

print()
print('=' * 60)
print('Обучение модели')
print('=' * 60)

path, emotions = load_all_datasets()

print('\n ИТОГ...')
print(f"    Сумма аудиофайлов для обучения - {len(path)}")
print(f"    Сумма эмоций - {len(emotions)}")


