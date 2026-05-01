# Загрузка аудио

import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm


DUSHA_DIR = r'E:\diplom_ser\data\dusha' # Датасет Души



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





def load_dusha():
    audio_paths = []
    emotions = []
    
    paths, ems = load_dusha_dataset(DUSHA_DIR)
    audio_paths.extend(paths)
    emotions.extend(ems)


    return audio_paths, emotions
