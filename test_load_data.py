# Тестовый файл проверки загрузки датасетов

# test_load_data.py
from load_all_data import load_ravdess, load_dusha_dataset, load_my_data

def test_load_ravdess():
    print("🧪 Тест: load_ravdess")
    paths, emotions = load_ravdess(r'E:\diplom_ser\data\RAVDESS')
    assert len(paths) == len(emotions), "❌ Разная длина списков!"
    assert len(paths) > 0, "❌ Пустой результат!"
    print(f"✅ OK: {len(paths)} файлов\n")

def test_load_dusha():
    print("🧪 Тест: load_dusha")
    paths, emotions = load_dusha_dataset(r'E:\diplom_ser\data\dusha', sample_size=100)
    assert len(paths) == len(emotions), "❌ Разная длина списков!"
    assert len(paths) == 100, "❌ Неверное количество файлов!"
    print(f"✅ OK: {len(paths)} файлов\n")

def test_load_my_data():
    print("🧪 Тест: load_my_data")
    paths, emotions = load_my_data(r'E:\diplom_ser\data\Actor_25_me')
    assert len(paths) == len(emotions), "❌ Разная длина списков!"
    assert len(paths) > 0, "❌ Пустой результат!"
    print(f"✅ OK: {len(paths)} файлов\n")

# Запуск всех тестов
if __name__ == "__main__":
    test_load_ravdess()
    test_load_dusha()
    test_load_my_data()
    print("🎉 Все тесты пройдены!")