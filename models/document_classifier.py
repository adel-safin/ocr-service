"""Классификатор типов документов на основе CNN"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import pickle
import logging

logger = logging.getLogger(__name__)


class DocumentClassifier(nn.Module):
    """CNN модель для классификации типов документов"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Инициализация классификатора
        
        Args:
            num_classes: Количество классов (типов документов)
            pretrained: Использовать предобученные веса
        """
        super(DocumentClassifier, self).__init__()
        
        # Использование предобученной ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Замена последнего слоя
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        """Прямой проход"""
        return self.backbone(x)
    
    def predict(self, image_path: str, device: str = 'cpu') -> Tuple[str, float]:
        """
        Предсказание типа документа
        
        Args:
            image_path: Путь к изображению
            device: Устройство для вычислений
            
        Returns:
            Кортеж (тип документа, уверенность)
        """
        self.eval()
        
        # Загрузка и предобработка изображения
        image = self._load_image(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item()
    
    def _load_image(self, image_path: str) -> Image.Image:
        """
        Загрузка изображения
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            PIL Image
        """
        if image_path.endswith('.pdf'):
            from pdf2image import convert_from_path
            images = convert_from_path(image_path, dpi=200, first_page=1, last_page=1)
            return images[0] if images else Image.new('RGB', (224, 224))
        
        # Загрузка через OpenCV для лучшей обработки
        img = cv2.imread(image_path)
        if img is None:
            return Image.new('RGB', (224, 224))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)


class DocumentClassifierTrainer:
    """Тренер для классификатора документов"""
    
    def __init__(self, model: DocumentClassifier, device: str = 'cpu'):
        """
        Инициализация тренера
        
        Args:
            model: Модель для обучения
            device: Устройство для вычислений
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
    
    def train(self, train_loader, val_loader, epochs: int = 10):
        """
        Обучение модели
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            epochs: Количество эпох
        """
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Обучение
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Валидация
            val_acc = self._validate(val_loader)
            
            train_acc = 100 * train_correct / train_total
            avg_loss = train_loss / len(train_loader)
            
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, "
                       f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # Сохранение лучшей модели
            # Сохраняем если валидация улучшилась ИЛИ если это первая эпоха
            if val_acc > best_val_acc or (epoch == 0 and val_acc >= 0):
                best_val_acc = val_acc
                self._save_model(f"best_model_epoch_{epoch+1}.pth")
                logger.info(f"Модель сохранена (val_acc: {val_acc:.2f}%)")
            
            self.scheduler.step()
    
    def _validate(self, val_loader) -> float:
        """
        Валидация модели
        
        Args:
            val_loader: DataLoader для валидации
            
        Returns:
            Точность валидации
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total if total > 0 else 0.0
    
    def _save_model(self, filename: str):
        """Сохранение модели"""
        os.makedirs('models/document_classifier', exist_ok=True)
        torch.save(self.model.state_dict(), 
                  f"models/document_classifier/{filename}")
