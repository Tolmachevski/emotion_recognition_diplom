import os
import sys


def get_resurse_path(relative_path):
    """
    Получить абсолютный путь к файлу
    Работает  в обычном режиме, и в exe-файле


    Примеры использования:
    - get_resource_path('model/emotion_model.pkl')
    - get_resource_path('templates/index.html')
    """

    if hasattr(sys, '_MEIPASS'):
        # PyInstaller распаковал все во временную папку
        # sys.MEIPASS содержит путь у этой папке
        base_path = sys._MEIPASS
    else:
        # Если запущено как обычный .py файл
        # Берем папку, где лежит этот скрипт
        base_path = os.path.dirname(os.path.abspath(__file__))


    # Соединяем базовый путь с относительным
    return os.path.join(base_path, relative_path)


def get_model_path(model_name):
    # Удобная функция для загрузки моделей
    return get_resurse_path(os.path.join('model', model_name))

def get_template_path(template_name):
    # Удобная функция для шаблонов
    return get_resurse_path(os.path.join('templates', template_name))