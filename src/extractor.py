# Расчет признаков

import numpy as np
import librosa
import os
from tqdm import tqdm
import time

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import partial
import threading




RAVDESS_DIR = r'E:\diplom_ser\data\ravdess' # Датасет Ravdess
MY_DATA_DIR = r'E:\diplom_ser\data\Actor_25_me' # Мой датасет
DUSHA_DIR = r'E:\diplom_ser\data\dusha' # Датасет Души
SAVE_DIR = r'E:\diplom_ser\model' # путь сохранения модели
DUSHA_DIR_TRAINED_DATA = r'E:\diplom_ser\data\dusha\crowd\crowd_train\wavs'


def extract_one_features(file, target_sr=22050):
    try:
        y, sr = librosa.load(file, sr=target_sr, mono=True) # librosa.load() всегда возвращает 2 значения, поэтому...
            # y - массив аудиоданных (амплитут звука) => y = ([0.001, -0.002, 0.003, ...])
            # sr - частота дискретизации (22050 - стандарт)

        if len(y) == 0:
            print(f"    ⚠️ Пустой файл {file}")
            return None

        # 1. RMS считаем ДО нормализации, чтобы сохранить абсолютную энергию для модели интенсивности
        rms = np.mean(librosa.feature.rms(y=y)[0])

        y = librosa.util.normalize(y) # Нормализуем все аудиоданные для корректного обучения модели
        # Это важно, потому что мы не полагаемся, к примеру, на абсолютную громкость.
        # Мы убираем неинформативные различия в ГРОМКОСТИ записи, оставляя адекватные акустичесикие паттерны

        y, _ = librosa.effects.trim(y, top_db=20) # Обрезаем тишину в начале и в конце аудиофайла
        # _ это обрезанный аудиосигнал
        # 20 - это порог, ниже этого все считается тишиной


        # 2. Извлекаем mfcc
        mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=13, norm='ortho')
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)


        # 3. Извлекаем ZCR (напряжение связок \ хрипота)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])

        # 4. Spectral Centroid ("Яркость" спектра, маркер крика \ шепота)
        centroid_raw = librosa.feature.spectral_centroid(y=y, sr=target_sr)[0]
        centroid = np.mean(centroid_raw) if len(centroid_raw) > 0 else 0.0
        centroid = np.nan_to_num(centroid, nan=0.0, posinf=3000.0, neginf=0.0)


        # Итоговый вектор
        return np.concatenate([mfcc_mean, mfcc_std, [rms, zcr, centroid]])

    except Exception as e:
        return None




def extract_features_parrallel(path_list, emotions=None, max_workers=6):


    print(f"🔄 Обрабатываем {len(path_list)} файлов в {max_workers} процессах...")

    # Буфер фиксированного размера. Порядок ячеек совпадает с path_list
    results_buffer = [None] * len(path_list)
    # Маска успешных файлов. Синхронизирована с буфером
    valid_mask = [False] * len(path_list)

    start_time = time.time() # 1. Засекаем старт

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # future_to_idx связывает каждую задачу с ее исходным индексом
        future_to_idx = {executor.submit(extract_one_features, path): i for i, path in enumerate(path_list)}
        
        # Собираем результаты в правильные ячейки буфера
        for future in tqdm(as_completed(future_to_idx), total=len(path_list), desc="Извлечение признаков"):
            idx = future_to_idx[future]
            res = future.result()
            if res is not None:
                results_buffer[idx] = res
                valid_mask[idx] = True



    # Логгируем среднюю скорость
    elapsed = time.time() - start_time # 2. Считаем затраченное время
    avg_speed = len(path_list) / elapsed if elapsed > 0 else 0 # 3. Средняя скорость

    print(f"\n✅ Готово за {elapsed:.2f} сек | Средняя скорость: {avg_speed:.2f} файлов/сек")



    # Фильтрация без потери синхронности признаков и меток
    X = np.array([r for r in results_buffer if r is not None])
    
    if emotions is not None:
        y = np.array([e for e, is_valid in zip(emotions, valid_mask) if is_valid])
        return X, y
    return X




# Проверка того, что все признаки излечены корректно
def validate_extracted_features(X, y=None, expected_dims=29):
    print(f"\n🔍 ВАЛИДАЦИЯ ПРИЗНАКОВ...")
    if X is None or len(X) == 0:
        print("❌ Признаки не извлечены или пустой буфер")
        return False

    # 1. Размерность
    if X.shape[1] != expected_dims:
        print(f"❌ Ошибка размерности: ожидалось {expected_dims}, получено {X.shape[1]}")
        return False
    print(f"✅ Размерность: {X.shape[0]} файлов × {X.shape[1]} признаков")

    # 2. Корректность значений (NaN/Inf)
    nan_cnt = np.isnan(X).sum()
    inf_cnt = np.isinf(X).sum()
    if nan_cnt > 0 or inf_cnt > 0:
        print(f"⚠️ Обнаружено {nan_cnt} NaN и {inf_cnt} Inf. Заменяю на 0/границы...")
        X[:] = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    else:
        print("✅ Значения корректны (нет NaN/Inf)")

    # 3. Выборочная статистика (проверка диапазонов)
    feat_names = ([f"mfcc{i}_m" for i in range(1,14)] + 
                  [f"mfcc{i}_s" for i in range(1,14)] + 
                  ["rms", "zcr", "centroid"])
    print("\n📊 Контрольные диапазоны (min / max / mean):")
    for idx in [0, 13, 26, 27, 28]:  # mfcc1_mean, mfcc1_std, rms, zcr, centroid
        col = X[:, idx]
        print(f"  {feat_names[idx]:10s} | {col.min():7.2f} / {col.max():7.2f} / {col.mean():7.2f}")

    # 4. Синхронность меток
    if y is not None:
        if len(y) != len(X):
            print(f"❌ Рассинхрон: X={len(X)}, y={len(y)}")
            return False
        print(f"✅ Меток: {len(y)}, классов: {len(np.unique(y))}")

    return True