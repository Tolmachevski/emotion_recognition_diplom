import os
import random
import librosa
import numpy as np
import pickle
import joblib
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
from functools import wraps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['EXAMPLES_FOLDER'] = 'static/examples'
app.config['MODEL_FOLDER'] = 'model'
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'm4a'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXAMPLES_FOLDER'], exist_ok=True)

# Словарь эмоций (RAVDESS)
EMOTIONS_MAP = {
    'neutral': "нейтральный голос",
    'calm': "спокойствие",
    'happy': "счастливый голос",
    'sad': "грусть",
    'angry': "злоба",
    'fearful': "страх",
    'disgust': "отвращение",
    'surprised': "удивление",
    # Добавь другие, если есть
}

# Загрузка модели и скалера при старте приложения
model = None
scaler = None

def load_model():
    """Загрузка ML модели и скалера через joblib"""
    global model, scaler
    try:
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'emotion_model_v2.pkl')
        scaler_path = os.path.join(app.config['MODEL_FOLDER'], 'scaler_v2.pkl')
        
        # Проверяем существование файлов
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Скалер не найден: {scaler_path}")
        
        # Загрузка через joblib
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        print("✓ Модель успешно загружена")
        print(f"  - Тип модели: {type(model).__name__}")
        print(f"  - Скалер: {type(scaler).__name__}")
        
    except Exception as e:
        print(f"⚠ Ошибка загрузки модели: {e}")
        print("Используется режим заглушки")
        model = None
        scaler = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_features(audio_path):
    """
    Извлечение признаков из аудиофайла.
    ДОЛЖНО СОВПАДАТЬ с тем, что использовалось при обучении модели!
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        
        # Объединение всех признаков
        features = np.concatenate([mfccs_mean, mfccs_std])
        
        return features
    except Exception as e:
        print(f"Ошибка извлечения признаков: {e}")
        return None

def analyze_voice(audio_path):
    """Анализ аудио с использованием реальной модели"""
    global model, scaler
    
    features = extract_features(audio_path)
    
    if features is None:
        return "Ошибка извлечения признаков"
    
    if model is None or scaler is None:
        emotion_id = random.randint(1, 8)
        return f"Вы {EMOTIONS_MAP.get('neutral', 'неизвестно')} 😊"
    
    try:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        
        # Преобразуем prediction в строку и ищем в словаре
        prediction_str = str(prediction).lower().strip()
        
        # Ищем эмоцию в словаре
        if prediction_str in EMOTIONS_MAP:
            result_text = EMOTIONS_MAP[prediction_str]
        else:
            # Если не нашли — показываем что вернула модель
            return f"Неизвестно (модель вернула: {prediction})"
        
        # Добавляем уверенность
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
            return f"{result_text} (уверенность: {confidence:.2f})"
        else:
            return result_text
    
    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        return "Ошибка анализа"

def get_example_files():
    """Получение списка примеров"""
    examples = []
    if os.path.exists(app.config['EXAMPLES_FOLDER']):
        files = os.listdir(app.config['EXAMPLES_FOLDER'])
        for f in sorted(files):
            if allowed_file(f):
                examples.append(f)
    return examples

@app.route('/')
def index():
    examples = get_example_files()
    return render_template('index.html', examples=examples)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email and password:
            session['user'] = email
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Недопустимый формат файла'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Анализ с реальной моделью
    result = analyze_voice(filepath)
    
    return jsonify({'result': result, 'filename': filename})

@app.route('/use_example/<filename>')
def use_example(filename):
    """Использование примера файла"""
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['EXAMPLES_FOLDER'], filename)
    
    if os.path.exists(filepath) and allowed_file(filename):
        result = analyze_voice(filepath)
        return jsonify({'result': result, 'filename': filename})
    
    return jsonify({'error': 'Файл не найден'}), 404

@app.route('/save_to_examples', methods=['POST'])
def save_to_examples():
    """Сохранение загруженного файла в примеры"""
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Недопустимый формат файла'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['EXAMPLES_FOLDER'], filename)
    
    counter = 1
    base_name, ext = os.path.splitext(filename)
    while os.path.exists(filepath):
        filename = f"{base_name}_{counter}{ext}"
        filepath = os.path.join(app.config['EXAMPLES_FOLDER'], filename)
        counter += 1
    
    file.save(filepath)
    
    return jsonify({'success': True, 'filename': filename})

@app.route('/delete_example/<filename>', methods=['DELETE'])
def delete_example(filename):
    """Удаление файла из примеров"""
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['EXAMPLES_FOLDER'], filename)
    
    if os.path.exists(filepath) and allowed_file(filename):
        os.remove(filepath)
        return jsonify({'success': True})
    
    return jsonify({'error': 'Файл не найден'}), 404

@app.route('/get_examples')
def get_examples():
    """Получение списка примеров"""
    examples = get_example_files()
    return jsonify({'examples': examples})

# Загрузка модели при старте
load_model()

if __name__ == '__main__':
    app.run(debug=True)