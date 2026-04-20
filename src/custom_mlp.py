#!/usr/bin/env python3
"""
自定义MLP分类器，支持GELU激活函数
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def gelu(x):
    """GELU激活函数"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class GELUMLP(nn.Module):
    """支持GELU激活函数的MLP网络"""
    
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout=0.1):
        super(GELUMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CustomMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    自定义MLP分类器，支持GELU激活函数
    """
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, learning_rate='constant', learning_rate_init=0.001,
                 max_iter=50, random_state=42, batch_size=512, early_stopping=True,
                 validation_fraction=0.1, patience=3, tol=1e-4):
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = float(alpha)
        self.learning_rate = learning_rate
        self.learning_rate_init = float(learning_rate_init)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.batch_size = int(batch_size)
        self.early_stopping = early_stopping
        self.validation_fraction = float(validation_fraction)
        self.patience = int(patience)
        self.tol = float(tol)
        
        # 设置随机种子
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def fit(self, X, y):
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
        """
        X, y = check_X_y(X, y, ensure_2d=True)
        
        # 存储类别信息
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # 数据预处理
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # 检查GPU可用性
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 转换为PyTorch张量并移动到设备
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        # 创建数据集
        dataset = TensorDataset(X_tensor, y_tensor)
        # 优化数据加载器
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # 在WSL中设为0避免问题
        )
        
        # 创建模型并移动到设备
        input_size = X.shape[1]
        self.model_ = GELUMLP(
            input_size=input_size,
            hidden_sizes=self.hidden_layer_sizes,
            output_size=1,
            dropout=self.alpha
        ).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        
        # 训练模型
        self.model_.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            
            # 计算平均损失
            avg_loss = epoch_loss / batch_count
            
            # 早停检查
            if self.early_stopping and avg_loss < best_loss - self.tol:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                break
        
        return self
    
    def predict_proba(self, X):
        """
        预测概率
        
        Args:
            X: 输入特征
        
        Returns:
            预测概率
        """
        check_is_fitted(self, ['model_', 'scaler_'])
        X = check_array(X, ensure_2d=True)
        
        # 数据预处理
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 预测
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        # 返回二分类概率格式
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """
        预测类别
        
        Args:
            X: 输入特征
        
        Returns:
            预测类别
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def get_feature_importance(self):
        """
        获取特征重要性（基于第一层权重）
        
        Returns:
            特征重要性数组
        """
        check_is_fitted(self, ['model_'])
        
        # 获取第一层权重
        first_layer = self.model_.network[0]
        weights = first_layer.weight.data.cpu().numpy()
        
        # 计算每个特征的重要性（权重的L2范数）
        importance = np.linalg.norm(weights, axis=0)
        return importance 