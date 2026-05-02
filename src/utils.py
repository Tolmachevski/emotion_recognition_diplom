import joblib
import numpy as np
import os
from datetime import datetime

def save_extracted_features(X, y, feat_names, scaler, save_path, metadata=None):
    """Сохраняет признаки, метки, скалер и метаданные в один архив."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = {
        'X': X,
        'y': y,
        'feat_names': np.array(feat_names),
        'scaler': scaler,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat()
    }
    # np.savez_compressed сжимает данные ~3-5× без потери точности
    np.savez_compressed(save_path, **{k: v for k, v in data.items() if k != 'scaler'})
    joblib.dump(scaler, save_path.replace('.npz', '_scaler.pkl'))  # Скалер отдельно для sklearn-совместимости
    
    print(f"💾 Сохранено: {save_path} ({os.path.getsize(save_path) / 1e6:.2f} МБ)")

def load_extracted_features(load_path):
    """Загружает сохранённые признаки и скалер."""
    if not os.path.exists(load_path):
        return None
    
    # allow_pickle=True обязателен для загрузки объектов (списков, словарей)
    data = np.load(load_path, allow_pickle=True)
    scaler = joblib.load(load_path.replace('.npz', '_scaler.pkl'))
    
    # Извлечение метаданных: .item() распаковывает 0-d массив
    metadata_raw = data['metadata']
    metadata = metadata_raw.item() if hasattr(metadata_raw, 'item') else dict(metadata_raw)
    
    print(f"📦 Загружено: {data['X'].shape[0]} образцов, {data['X'].shape[1]} признаков")
    
    # feat_names тоже может прийти как 0-d array, если сохранялся как объект
    feat_names_raw = data['feat_names']
    feat_names = feat_names_raw.tolist() if hasattr(feat_names_raw, 'tolist') else list(feat_names_raw)
    
    return data['X'], data['y'], feat_names, scaler, metadata