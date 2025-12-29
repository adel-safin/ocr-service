#!/usr/bin/env python3
"""Скрипт для обучения классификатора типов документов"""
import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from utils.dataset_loader import DatasetLoader
from models.document_classifier import DocumentClassifier, DocumentClassifierTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentDataset(Dataset):
    """Датасет для обучения классификатора"""
    
    def __init__(self, training_pairs, class_to_idx, transform=None, is_training=False):
        """
        Инициализация датасета
        
        Args:
            training_pairs: Список пар (путь к изображению, тип документа)
            class_to_idx: Словарь соответствия классов индексам
            transform: Трансформации изображений
            is_training: Использовать аугментацию для обучения
        """
        self.pairs = training_pairs
        self.class_to_idx = class_to_idx
        self.is_training = is_training
        
        # Базовые трансформации
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
        
        # Аугментация для обучения
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = train_transform if is_training else (transform or base_transform)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        image_path, doc_type = self.pairs[idx]
        
        # Загрузка изображения
        image = self._load_image(image_path)
        
        # Применение трансформаций
        if self.transform:
            image = self.transform(image)
        
        # Получение индекса класса
        label = self.class_to_idx.get(doc_type, 0)
        
        return image, label
    
    def _load_image(self, image_path):
        """Загрузка изображения"""
        if image_path.endswith('.pdf'):
            from pdf2image import convert_from_path
            images = convert_from_path(image_path, dpi=200, first_page=1, last_page=1)
            return images[0] if images else Image.new('RGB', (224, 224))
        
        img = cv2.imread(image_path)
        if img is None:
            return Image.new('RGB', (224, 224))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)


def main():
    """Основная функция обучения"""
    # Параметры
    dataset_root = "../Датасет"
    batch_size = 4  # Уменьшаем для малого датасета
    epochs = 20  # Увеличиваем для лучшего обучения
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Используется устройство: {device}")
    
    # Загрузка датасета
    loader = DatasetLoader(dataset_root)
    
    # Поиск пар документов (изображение + тип)
    all_pairs = []
    doc_types = loader.get_all_document_types()
    
    logger.info(f"Найдено типов документов: {len(doc_types)}")
    
    for doc_type in doc_types:
        pairs = loader.find_document_pairs(doc_type)
        for pair in pairs:
            all_pairs.append((pair['image_path'], pair['document_type']))
        logger.info(f"  {doc_type}: {len(pairs)} пар")
    
    if not all_pairs:
        logger.error("Не найдено данных для обучения")
        return
    
    # Создание словаря классов
    unique_types = sorted(set(doc_type for _, doc_type in all_pairs))
    class_to_idx = {doc_type: idx for idx, doc_type in enumerate(unique_types)}
    idx_to_class = {idx: doc_type for doc_type, idx in class_to_idx.items()}
    
    logger.info(f"Найдено {len(unique_types)} типов документов: {unique_types}")
    logger.info(f"Всего пар для обучения: {len(all_pairs)}")
    
    # Разделение на train/val (80/20)
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    # Создание датасетов с аугментацией для обучения
    train_dataset = DocumentDataset(train_pairs, class_to_idx, is_training=True)
    val_dataset = DocumentDataset(val_pairs, class_to_idx, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Создание модели
    num_classes = len(unique_types)
    model = DocumentClassifier(num_classes=num_classes, pretrained=True)
    
    # Обучение
    trainer = DocumentClassifierTrainer(model, device=device)
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Сохранение словаря классов
    import pickle
    os.makedirs('models/document_classifier', exist_ok=True)
    with open('models/document_classifier/class_mapping.pkl', 'wb') as f:
        pickle.dump({'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class}, f)
    
    logger.info("Обучение завершено!")


if __name__ == "__main__":
    import os
    main()
