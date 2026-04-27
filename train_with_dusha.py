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




from load_dusha import load_dusha_dataset # Импортируем загрузчик Души

# Пути к файлам\папкам
RAVDESS_CSV = r'E:\diplom_ser\features.csv' # Датасет Ravdess
MY_DATA_DIR = r'E:\diplom_ser\data\Actor_25_me' # Мой датасет
DUSHA_DIR = r'E:\diplom_ser\data\dusha' # Датасет Души
SAVE_DIR = r'E:\diplom_ser\model' # путь сохранения модели
DUSHA_DIR_TRAINED_DATA = r'E:\diplom_ser\data\dusha\crowd\crowd_train\wavs'

print('Обучаем модель с датасетом ДУША')



# ============================================
# ШАГ 1: Загрузка RAVDESS
# ============================================
print("\n📦 ШАГ 1: Загрузка RAVDESS...")
df_ravdess = pd.read_csv(RAVDESS_CSV) # С помощью пандаса читаем csv-файл и помещаем в переменную
feature_cols = [f'feature_{i}' for i in range(1, 27)] # каждый столбец помечаем своим {i} с 1 по 26 включая
X_ravdess = df_ravdess[feature_cols].values # Извлекаем NumPy-массив из колонок в списке feature_cols
# X_ravdess теперь двумерный массив формы (len(строк), 26)
y_ravdess = df_ravdess['emotion'].values # Получаем одномерный массив эмоций (поскольку изначально в пандас обьекте было одно измерение)
print(f"\n      Ravdess успешно загружен: {len(X_ravdess)} файлов, 26 признаков")


# ============================================
# ШАГ 2: Загрузка моих записей (моего датасета)
# ============================================
print(f"\n📦 ШАГ 2: Загрузка моего датасета...")

# Начинаем извлекать признак mfcc для моего датасета
def extract_mfcc(audio_path):
    try:
        y, sr = librosa.load(audio_path) # librosa.load() всегда возвращает 2 значения, поэтому...
            # y - массив аудиоданных (амплитут звука) => y = ([0.001, -0.002, 0.003, ...])
            # sr - частота дискретизации (22050 - стандарт)
        y = librosa.util.normalize(y) # Нормализуем все аудиоданные для корректного обучения модели
        # Это важно, потому что мы не полагаемся, к примеру, на абсолютную громкость.
        # Мы убираем неинформативные различия в ГРОМКОСТИ записи, оставляя адекватные акустичесикие паттерны

        y, _ = librosa.effects.trim(y, top_db=20) # Обрезаем тишину в начале и в конце аудиофайла
        # _ это обрезанный аудиосигнал
        # 20 - это порог, ниже этого все считается тишиной
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050) # Проводим ресемплинг на частоту 22050
        mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=13) # Извлекаем 13 признаков из аудиосигнала
        mfcc_mean = np.mean(mfcc, axis=1) # Усредняем по времени по столбцам
        mfcc_std = np.std(mfcc, axis=1) # Считаем разброс по времени
        return np.concatenate([mfcc_mean, mfcc_std])
    except Exception as e:
        print(f"     ❌ Внутри extract_mfcc: {type(e).__name__}: {e}")
        raise
    
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
            
            

X_my = np.array(my_features)
y_my = np.array(my_labels)

print('\n Мой датасет загружен')
print(f"    ✅ Мои записи: {len(X_my)} файлов")
print(f"    ✅ Мои эмоции: {len(y_my)}")


# ============================================
# ШАГ 3: Загрузка Dusha (НАЧИНАЕМ С МАЛОГО!)
# ============================================

print(f"\n📦 ШАГ 3: Загрузка датасета ДУША")

X_dusha_train, y_dusha_train = load_dusha_dataset(
    dusha_dir=DUSHA_DIR,
    subset='crowd_train', # Из какой папки берем данные
    sample_size=500 # Берем только 500 файлов из датасета
)

if X_dusha_train is None:
    print("ОШИБКА ЗАГРУЗКИ ДУШИ, ПРОВЕРЬ ПУТИ!")




# 🔍 ПРОВЕРКА: Откуда берутся числа?
print("\n🔍 ПРОВЕРКА: Какие эмоции в каждом датасете?")
print(f"RAVDESS: {np.unique(y_ravdess)[:10]}...")  # Первые 10
print(f"Мои: {np.unique(y_my)}")
print(f"Dusha: {np.unique(y_dusha_train)}")






# ============================================
# ШАГ 4: Объединение всех данных
# ============================================

print("\n📦 ШАГ 4: Начинаем обьединять данные...")

X_all = np.vstack([X_ravdess, X_my, X_dusha_train])
y_all = np.concatenate([y_ravdess, y_my, y_dusha_train])

print(f"Всего данных: {len(X_all)} файлов. Из них...")
print(f"Ravdess - {len(X_ravdess)} файлов")
print(f"Мой - {len(X_my)} файлов")
print(f"DUSHA - {len(X_dusha_train)} файлов")


print('\n Распределение эмоций')
for emotion, count in zip(*np.unique(y_all, return_counts=True)):
    print(f"    {emotion}: {count} ({count/len(y_all)*100:.1f}%)")

    # y_all с помощью функции np.unique с параметром return_counts=True делит один массив на два кортежа:
    # ['sad', 'happy', 'angry'] - первый кортеж
    # [1, 2, 3] - сколько раз встречаются 
    # т.е имеем условно result = ([array1, array2])
    # * означает распаковку кортежа, т.е. получаем zip(array1, array2)
    # С помощью функции zip мы берем попарно каждый элемент из каждого массива и получаем:
    # пары = [('angry', 3), ('happy', 2), ('sad', 1)]
    # в emotion - эмоцию, в count - число
    # Получаем  angry: 3
    #           happy: 2
    #           sad: 1

# ============================================
# ШАГ 5: Разделение на train/test
# ============================================
print("\n ШАГ 5: Разделение на train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, # признаки 
    y_all, # Эмоции
    test_size=0.2, # Тестовых файлов будет 20%, а 80% в обучении
    random_state=42, # Фиксируем "Случайность при воспроизведении"
    stratify=y_all # Сохраняем пропорцию классов
)

print(f"Тренировочные {len(X_train)} файлов")
print(f"Тестовые {len(X_test)} файлов")



# ============================================
# ШАГ 6: Аугментация данных
# ============================================
print("ШАГ 6: Аугментация ТОЛЬКО тренировочных данных")

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

X_train_aug = augmentation_audio(X_train, sr=22050)













# ============================================
# ШАГ 7: Нормализация
# ============================================
print("\n ШАГ 7: Нормализация...")

scaler = StandartScaler() # Созаем инструмент для масштабирования
X_train_aug_scaled = scaler.fit_transform(X_train_aug)
# fit() - изучает тренировочные данные: считает среднее и std для каждого признака
# transform() - применяет формулу к тренировочным данным
X_test_scaled = scaler.transform(X_test) # Применяет те же самые средние и std, но не пересчитывает заного для теста
# fit() применяем только для ТРЕНИРОВОЧНЫХ данных, поскольку если применим - модель запомнит их, и на выходе мы получим "ложные" хорошие результаты
# Это как при обучении ученика дать ему материалы экзамена, а потом протестировать ученика на этом же экзамене


print("Нормализация выполнена")




# ============================================
# ШАГ 8: Обучение модели
# ============================================
print("\n ШАГ 8: Обучение модели SVM...")
print("\n Перед обучением модели подберем лучшие параметры")
print("\n ШАГ 8.1: Подбор лучших парамеров для обучения модели")


# Сетка параметров (БОЛЬШЕ вариантов для gamma!)
param_grid = {
    'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
    'gamma': [0.0001, 0.00015, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.009, 0.01, 0.017, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'kernel': ['rbf']
}

# Создаём GridSearchCV
grid = GridSearchCV(
    SVC(random_state=42), 
    param_grid, 
    cv=5,                    # 5-кратная кросс-валидация
    scoring='accuracy', 
    n_jobs=-1,               # Использовать все ядра CPU
    verbose=2                # Показывать прогресс
)

grid.fit(X_train_aug_scaled, y_train)

# Результаты
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ ПОДБОРА ПАРАМЕТРОВ")
print("=" * 60)
print(f"✅ Лучшие параметры: {grid.best_params_}")
print(f"✅ Лучшая точность (CV): {grid.best_score_:.2%}")


# ============================================
# ШАГ 6: Обучение модели
# ============================================
print("\n" + "=" * 60)
print("ШАГ 6: Обучение финальной модели SVM на лучших параметрах")
print("=" * 60)

model = grid.best_estimator_

model.fit(X_train_aug_scaled, y_train)

print("Модель обучена.")