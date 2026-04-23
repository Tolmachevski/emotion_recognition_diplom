# Файл загрузки души (устаревший)

import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm


# карта эмоций (у "Души" всего 4 эмоции)
EMOTION_MAP = {
    'neutral': 'neutral',
    'positive': 'happy',    # positive → happy
    'sad': 'sad',
    'angry': 'angry',
}



# Функция извлечения mfcc для Души
def extract_mfcc_features(audio_path, target_sr=22050):
    try:

        y, sr = librosa.load(audio_path, sr=target_sr)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y, top_db=20)

        mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        return np.concatenate([mfcc_mean, mfcc_std])
    except Exception as e:
        print(f'Ошибк с файлом {audio_path}: {e}')
        return None
    


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
    
    # Загружаем TSV
    print(f'Загружаем данные из {subset}...')
    df = pd.read_csv(tsv_path, sep='\t')


    # Фильтруем по эмоциям и мапим
    df = df[df['annotator_emo'].isin(EMOTION_MAP.keys())].copy()
    df['emotion'] = df['annotator_emo'].map(EMOTION_MAP)

    # Берем выборку (если нужно)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    print(f'Найдено {len(df)} файлов с подходящими эмоциями')



    # ОТЛАДКА: Покажем первые 3 пути для проверки
    print(f"\n🔍 Проверка путей:")
    for idx, row in df.head(3).iterrows():
        wav_path = os.path.join(wavs_dir, row['hash_id'] + '.wav')
        exists = os.path.exists(wav_path)
        print(f"   {row['hash_id'][:16]}... → {exists}")
    print()



    # Извлекаем признаки
    all_features = []
    all_labels = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Обработка {subset}"):
        wav_path = os.path.join(wavs_dir, row['hash_id'] + '.wav')

        if not os.path.exists(wav_path):
            skipped += 1
            continue


        features = extract_mfcc_features(wav_path)

        if features is not None:
            all_features.append(features)
            all_labels.append(row['emotion'])
        else:
            skipped += 1

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f'Обработка {subset} готова!')
    print(f"    Обработано {len(X)} файлов")
    print(f"    Прпущено {skipped} файлов")

    if len(y) > 0:
        print(f"Распределение эмоций: {dict(zip(*np.unique(y, return_counts=True)))}")
    else:
        print(" ⚠️  ВНИМАНИЕ: Ни один файл не обработан или обработан неверно!")

    return X, y