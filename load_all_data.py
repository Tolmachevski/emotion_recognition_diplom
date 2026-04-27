# Общий файл загрузки датасетов

import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm


# Пути к файлам\папкам
RAVDESS_DIR = r'E:\diplom_ser\data\ravdess' # Датасет Ravdess
MY_DATA_DIR = r'E:\diplom_ser\data\Actor_25_me' # Мой датасет
DUSHA_DIR = r'E:\diplom_ser\data\dusha' # Датасет Души
CREMA_D_DIR = r'E:\diplom_ser\data\crema_d' # "Чистая" часть датасета крема


# Функция загрузки датасета Душа
def load_dusha_dataset(dusha_dir, subset='train', sample_size=None):
    if subset == 'train':
        tsv_path = os.path.join(dusha_dir, 'crowd', 'crowd_train', 'raw_crowd_train.tsv')
        wavs_dir = os.path.join(dusha_dir, 'crowd', 'crowd_train', 'wavs')
    else:
        tsv_path = os.path.join(dusha_dir, 'crowd', 'crowd_test', 'raw_crowd_test.tsv')
        wavs_dir = os.path.join(dusha_dir, 'crowd', 'crowd_test', 'wavs')


    # Проверка существования
    if not os.path.exists(tsv_path):
        print(f"❌ Файл не найден: {tsv_path}")
        return None, None
    
    if not os.path.exists(wavs_dir):
        print(f"❌ Папка не найдена: {wavs_dir}")
        return None, None
    
    print('\n 🔄 ЗАГРУЗКА ДАТАСЕТА DUSHA')
    # Загружаем TSV
    print(f'Загружаем данные из {subset}...')
    df = pd.read_csv(tsv_path, sep='\t')

    audio_paths = []
    emotions = []
    skipped = 0

    emotion_map = {
        'neutral': 'neutral',    # Прямое соответствие
        'positive': 'happy',     # Позитив → радость
        'sad': 'sad',            # Прямое соответствие
        'angry': 'angry',        # Прямое соответствие
    }

    
    # Оставляем только строки, где эмоция есть в нашем словаре
    df = df[df['annotator_emo'].isin(emotion_map.keys())].copy()
    
    # маппим (опасная методика!)
    # Создаём новую колонку 'emotion' с переименованными эмоциями
    df['emotion'] = df['annotator_emo'].map(emotion_map)
    
    if sample_size is not None:
        # sample() — случайная выборка из таблицы
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f'📋 Ограничиваем выборку до {sample_size} файлов')


    for idx, row in tqdm(df.iterrows(),total=len(df), desc=f"Обрабатываем {subset} из папки {DUSHA_DIR}"):
        
        # === ШАГ 10: Строим путь к файлу ===
        # Важно: используем hash_id + '.wav', а не audio_path
        # Потому что audio_path уже содержит 'wavs/', будет дублирование
        wav_filename = row['hash_id'] + '.wav'
        wav_path = os.path.join(wavs_dir, wav_filename)
        
        # === ШАГ 11: Проверяем, существует ли файл ===
        if not os.path.exists(wav_path):
            skipped += 1
            continue  # Переходим к следующей строке
        
        # === ШАГ 12: Добавляем в списки ===
        audio_paths.append(wav_path)
        emotions.append(row['emotion'])
    
    # === ШАГ 13: Выводим статистику ===
    print('✅ Датасет DUSHA загружен!')
    print(f' {subset.upper()} готово!')
    print(f'   Обработано: {len(audio_paths)} файлов')
    print(f'   Пропущено: {skipped} файлов')
    
    # === ШАГ 14: Показываем распределение эмоций ===
    if len(emotions) > 0:
        from collections import Counter
        emotion_counts = Counter(emotions)
        print(f'   Распределение эмоций:')
        for emotion, count in emotion_counts.items():
            print(f'      {emotion}: {count} ({count/len(emotions)*100:.1f}%)')
    
    # === ШАГ 15: Возвращаем результат ===
    return audio_paths, emotions



# Функция загрузки Ravdess
def load_ravdess(data_dir):

    
    # Словарь меток эмоций
    emotion_map = {
            '01': 'neutral',    # Нейтрально
            '02': 'calm',       # Спокойно
            '03': 'happy',      # Счастливо
            '04': 'sad',        # Грустно
            '05': 'angry',      # Зло
            '06': 'fearful',    # Страшно
            '07': 'disgust',    # Отвращение
            '08': 'surprised'   # Удивление
        }

    audio_paths = []
    emotions = []
    skipped = 0

    if not os.path.exists(data_dir):
            print(f"❌ Папка датасета Ravdess не найдена или нарушен путь!")
            return None, None
    
    print('\n 🔄 ЗАГРУЗКА ДАТАСЕТА RAVDESS')

    for root, dirs, files in tqdm(os.walk(data_dir)):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file) # Соединяет путь к файлу и имя файла в один
                emotion_code = file.split('-')[2] # Находим число, обозначающее эмоцию в имени файла
                emotion = emotion_map[emotion_code] # преобразуем число в слово, которое нашли в словаре emotion_map и добавлем ее в новую переменную
                audio_paths.append(file_path)
                emotions.append(emotion)
            else:
                skipped += 1


    print('✅ Датасет RAVDESS загружен!')
    print(f'   Обработано: {len(audio_paths)} файлов')
    print(f'   Пропущено: {skipped} файлов')
    
    return audio_paths, emotions
                

# Функция загрузки моего датасета
def load_my_data(my_data_dir, subset='Actor_25_me'):
    
    # Словарь меток эмоций
    emotion_map = {
            'neutral': 'neutral',    # Нейтрально
            'calm': 'calm',       # Спокойно
            'happy': 'happy',      # Счастливо
            'sad': 'sad',        # Грустно
            'angry': 'angry',      # Зло
            'fear': 'fearful',    # Страшно
            'disgust': 'disgust',    # Отвращение
            'surprised': 'surprised'   # Удивление
        }

    audio_paths = []
    emotions = []
    skipped = 0

    if not os.path.exists(my_data_dir):
            print(f"❌ Папка собственного датасета не найдена или нарушен путь!")
            return None, None
    

    print('\n 🔄 ЗАГРУЗКА СОБСТВЕННОГО ДАТАСЕТА')

    for root, dirs, files in tqdm(os.walk(my_data_dir), desc=f'Обработка {subset}'):
        for file in files:
            if file.endswith('.m4a'):
                file_path = os.path.join(root, file) # Соединяет путь к файлу и имя файла в один
                
                emotion_code = file.split('_')[0] 
                
                emotion = emotion_map[emotion_code] # Преобразуем код эмоции из имени файла в стандартное название
                audio_paths.append(file_path)
                emotions.append(emotion)
            else:
                skipped += 1

    print('✅ Собственный датасет загружен!')
    print(f' {subset.upper()} готово!')
    print(f'   Обработано: {len(audio_paths)} файлов')
    print(f'   Пропущено: {skipped} файлов')
    
    return audio_paths, emotions


# Функция загрузки датасета Crema_D
def load_crema_d(crema_dir, subset='AudioMP3'):
    

    # Словарь меток эмоций
    emotion_map = {
        'NEU': 'neutral',    # Нейтрально
        'HAP': 'happy',      # Счастливо
        'SAD': 'sad',        # Грустно
        'ANG': 'angry',      # Зло
        'FEA': 'fearful',    # Страшно
        'DIS': 'disgust',    # Отвращение
    }

    audio_paths = []
    emotions = []
    skipped = 0

    if not os.path.exists(crema_dir):
            print(f"❌ Папка датасета CREMA_D не найдена или нарушен путь!")
            return None, None
    

    print('\n 🔄 ЗАГРУЗКА ДАТАСЕТА CREMA_D')

    for root, dirs, files in tqdm(os.walk(crema_dir), desc=f'обработка {subset}'):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(root, file) # Соединяет путь к файлу и имя файла в один
                

                emotion_code = file.split('_')[2] # Вытаскиваем код эмоции из имени файла
                
                emotion = emotion_map[emotion_code] # Преобразуем код эмоции из имени файла в стандартное название
                audio_paths.append(file_path)
                emotions.append(emotion)
            else:
                skipped += 1

    if skipped > 0:
        print(" ❌ ИМЕЮТСЯ ПРОПУЩЕННЫЕ ФАЙЛЫ ДАТАСЕТА CREMA_D !!!")
        if skipped > len(audio_paths):
            print(" ⚠️  Обработка прошла только по папке AudioMP3")

    print('✅ Датасет CREMA_D загружен!')
    print(f'   {subset.upper()} готово!')
    print(f'   Обработано: {len(audio_paths)} файлов')
    print(f'   Пропущено: {skipped} файлов')
    
    return audio_paths, emotions
    










def load_all_datasets():
    audio_paths = []
    emotions = []
    
    paths, ems = load_ravdess(RAVDESS_DIR)
    audio_paths.extend(paths)
    emotions.extend(ems)

    #paths, ems = load_dusha_dataset(DUSHA_DIR)
    #audio_paths.extend(paths)
    #emotions.extend(ems)

    paths, ems = load_crema_d(CREMA_D_DIR)
    audio_paths.extend(paths)
    emotions.extend(ems)

    #paths, ems = load_my_data(MY_DATA_DIR)
    #audio_paths.extend(paths)
    #emotions.extend(ems)

    return audio_paths, emotions
