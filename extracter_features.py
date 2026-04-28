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


def extract_one_mfcc(file, target_sr=22050):
    try:
        y, sr = librosa.load(file, sr=target_sr, mono=True) # librosa.load() всегда возвращает 2 значения, поэтому...
            # y - массив аудиоданных (амплитут звука) => y = ([0.001, -0.002, 0.003, ...])
            # sr - частота дискретизации (22050 - стандарт)

        if len(y) == 0:
            print(f"    ⚠️ Пустой файл {file}")
            return None


        y = librosa.util.normalize(y) # Нормализуем все аудиоданные для корректного обучения модели
        # Это важно, потому что мы не полагаемся, к примеру, на абсолютную громкость.
        # Мы убираем неинформативные различия в ГРОМКОСТИ записи, оставляя адекватные акустичесикие паттерны

        y, _ = librosa.effects.trim(y, top_db=20) # Обрезаем тишину в начале и в конце аудиофайла
        # _ это обрезанный аудиосигнал
        # 20 - это порог, ниже этого все считается тишиной

        mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=13)
        

        return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    except Exception as e:
        return None




def extract_mfcc_parralel(path_list, emotions=None, max_workers=6):


    print(f"🔄 Обрабатываем {len(path_list)} файлов в {max_workers} процессах...")

    # Буфер фиксированного размера. Порядок ячеек совпадает с path_list
    results_buffer = [None] * len(path_list)
    # Маска успешных файлов. Синхронизирована с буфером
    valid_mask = [False] * len(path_list)

    start_time = time.time() # 1. Засекаем старт

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # future_to_idx связывает каждую задачу с ее исходным индексом
        future_to_idx = {executor.submit(extract_one_mfcc, path): i for i, path in enumerate(path_list)}
        
        # Собираем результаты в правильные ячейки буфера
        for future in tqdm(as_completed(future_to_idx), total=len(path_list), desc="Извлечение MFCC"):
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


    
# УСТАРЕВШЕЕ

def concatenate_all_features():
    my_features = []
    my_labels = []

    for file in tqdm(os.listdir(MY_DATA_DIR), desc="Обработка мего собственного датасета"):
        if file.endswith('.wav') or file.endswith('.m4a'): # Если файл оканчивается этими видами файла
            file_path = os.path.join(MY_DATA_DIR, file) # Создаем переменную и добавляем склееные путь к папке и имя файла
            try:
                features = extract_mfcc(file_path) # Извлекаем mfcc-признаки из файла 
                my_features.append(features) # 26 признаков добавляем в список my_features
                
                my_labels.append(file.split('_')[0]) # Обрезаем имя файла, чтобы получить эмоцию
                # Пример файла моего датасета: angry_10.m4a
            except Exception as e:
                print(f"     ❌ Проблема с файлом {file}: {e}")
    return my_features, my_labels





def extract_one_mfcc_v05(file, target_sr=22050):
    try:
        y, sr = librosa.load(file) # librosa.load() всегда возвращает 2 значения, поэтому...
            # y - массив аудиоданных (амплитут звука) => y = ([0.001, -0.002, 0.003, ...])
            # sr - частота дискретизации (22050 - стандарт)
        y = librosa.util.normalize(y) # Нормализуем все аудиоданные для корректного обучения модели
        # Это важно, потому что мы не полагаемся, к примеру, на абсолютную громкость.
        # Мы убираем неинформативные различия в ГРОМКОСТИ записи, оставляя адекватные акустичесикие паттерны

        y, _ = librosa.effects.trim(y, top_db=20) # Обрезаем тишину в начале и в конце аудиофайла
        # _ это обрезанный аудиосигнал
        # 20 - это порог, ниже этого все считается тишиной


        if len(y) == 0:
            print(f"⚠️ Пустой файл после обрезки: {file}")
            error_count += 1

        mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        return np.concatenate([mfcc_mean, mfcc_std])
    except Exception as e:
        return None




def extract_mfcc_parralel_v05(path_list, emotions=None, max_workers=6):


    print(f"🔄 Обрабатываем {len(path_list)} файлов в {max_workers} потоков...")

    all_features_mfcc = []
    valid_emotions = []


    start_time = time.time() # 1. Засекаем старт

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Запускаем обработку всех файлов
        future_to_idx = {executor.submit(extract_one_mfcc_v05, path): i 
                        for i, path in enumerate(path_list)}
        
        # Собираем результаты с прогрессом
        for future in tqdm(as_completed(future_to_idx), total=len(path_list), 
                           desc="Извлечение MFCC"):
            features = future.result()
            if features is not None:
                all_features_mfcc.append(features)
                if emotions:
                    idx = future_to_idx[future]
                    valid_emotions.append(emotions[idx])


    # Логгируем среднюю скорость
    elapsed = time.time() - start_time # 2. Считаем затраченное время
    avg_speed = len(path_list) / elapsed if elapsed > 0 else 0 # 3. Средняя скорость

    print(f"\n✅ Готово за {elapsed:.2f} сек | Средняя скорость: {avg_speed:.2f} файлов/сек")
    
    X = np.array(all_features_mfcc)
    
    if emotions:
        return X, np.array(valid_emotions)
    return X