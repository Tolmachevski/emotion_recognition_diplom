# Файл аугментации данных
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import librosa
from tqdm import tqdm

# После загрузки данных мы имеем список путей к файлам и список эмоций к каждому файлу








# аугментация занимает слишком много времени для одного ядра компьютера (скрипт будет выполняться около 15 часов)
# поэтому мы ускоряем операцию в 6 раз, задействовав все 6 ядер процессора






# Загрузка одного файла
def load_one_audio(file_path):
    try:
        y, _ = librosa.load(file_path, sr=22050)
        y = librosa.util.normalize(y)
        return y
    except Exception as e:
        print(f"Ошибка с {file_path}: {e}")
        
        return None


# Загрузка парралельно всех аудиофайлов
def load_audio_parallel(file_paths, max_workers):
    print(f"Загружаем {len(file_paths)} файлов, используя {max_workers} потоков процессора...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(load_one_audio, file_paths),
            total = len(file_paths),
            desc='Загрузка аудио'
        ))

    audio_list = [r for r in results if r is not None]

    print(f"✅ Успешно загружено: {len(audio_list)} из {len(file_paths)}")
    return audio_list



# Шаг 2. Аугментируем каждое аудио (ВНИМАНИЕ! НАРУШАЕТСЯ ПОРЯДОК ВЫВОДА ДАННЫХ!)
def augmentation_audio(y, sr, n_augments=3):
    # Создаем аугментированные версии аудио, применяем только к тренировочным данным
    """
    y: аудио массив
    sr: частота дискретизации
    n_arguments: сколько версий создать

    """
    augmented = [y]

    noise = np.random.randn(len(y)) * 0.005
    augmented.append(y + noise)
    augmented.append(y * 0.8) 
    augmented.append(y * 1.2)

    shift = int(sr * 0.1) # 100 миллисекунд задержка
    augmented.append(np.roll(y, shift))

    return augmented[:n_augments + 1]