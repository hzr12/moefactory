import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import argparse

from data_preprocess import load_train_data, encode_categorical_features, prepare_train_data, TrainDelayDataset
from models import EnsembleModel, TransformerPredictor, LSTMPredictor, Seq2SeqPredictor

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name):
    """
    训练模型
    """
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in {model_name} at epoch {epoch+1}")
                continue
                
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # 检查损失是否为NaN
                if torch.isnan(loss):
                    print(f"Warning: NaN validation loss encountered in {model_name} at epoch {epoch+1}")
                    continue
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # 检查损失是否为NaN
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"Warning: NaN loss detected in {model_name} at epoch {epoch+1}, stopping training")
            break
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./model/{model_name}_best.pth')
            print(f'Saved best {model_name} model with val loss: {best_val_loss:.4f}')
    
    return model

def train_traditional_models(X_train, y_train, X_val, y_val):
    """
    训练传统机器学习模型
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from lightgbm import LGBMRegressor
        from sklearn.metrics import mean_squared_error
        import pickle
        
        # 训练随机森林模型
        print("Training Random Forest Model...")
        rf_model = RandomForestRegressor(max_leaf_nodes=127,n_estimators=1000, random_state=3461)
        rf_model.fit(X_train, y_train)
        
        # 验证随机森林模型
        rf_pred = rf_model.predict(X_val)
        rf_mse = mean_squared_error(y_val, rf_pred)
        print(f"Random Forest Val MSE: {rf_mse:.4f}")
        
        # 保存随机森林模型
        with open('./model/random_forest_best.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        
        # 训练LightGBM模型
        print("Training LightGBM Model...")
        lgb_model = LGBMRegressor(num_leaves=127,n_estimators=1000, random_state=34646)
        lgb_model.fit(X_train, y_train)
        
        # 验证LightGBM模型
        lgb_pred = lgb_model.predict(X_val)
        lgb_mse = mean_squared_error(y_val, lgb_pred)
        print(f"LightGBM Val MSE: {lgb_mse:.4f}")
        
        # 保存LightGBM模型
        with open('./model/lightgbm_best.pkl', 'wb') as f:
            pickle.dump(lgb_model, f)
            
    except ImportError as e:
        print(f"Warning: Could not import traditional ML libraries: {e}")
        print("Skipping traditional ML model training...")

def main():
    # 创建必要的目录
    os.makedirs('./model', exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    print("Loading training data...")
    train_df = load_train_data('./datasets/train')
    print(f"Loaded {len(train_df)} training samples")
    
    # 编码分类特征
    train_df, label_encoder = encode_categorical_features(train_df)
    
    # 准备训练数据
    X, y = prepare_train_data(train_df)
    
    # 检查是否有NaN值并处理
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"NaN in X: {np.isnan(X).sum()}, NaN in y: {np.isnan(y).sum()}")
    
    # 处理NaN值
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=5364)
    
    # 训练传统机器学习模型
    train_traditional_models(X_train, y_train, X_val, y_val)
    
    # 创建数据集和数据加载器
    train_dataset = TrainDelayDataset(X_train, y_train)
    val_dataset = TrainDelayDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=34, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=34, shuffle=False)
    
    # 模型参数
    input_dim = X_train.shape[1]
    num_epochs = 3000
    learning_rate = 0.001
    
    print(f"Input dimension: {input_dim}")
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 训练集成模型
    print("Training Ensemble Model...")
    ensemble_model = EnsembleModel(input_dim=input_dim)
    ensemble_optimizer = optim.Adam(ensemble_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_model(ensemble_model, train_loader, val_loader, criterion, ensemble_optimizer, num_epochs, device, "ensemble")
    
    # 训练Transformer模型
    print("Training Transformer Model...")
    transformer_model = TransformerPredictor(input_dim=input_dim)
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_model(transformer_model, train_loader, val_loader, criterion, transformer_optimizer, num_epochs, device, "transformer")
    
    # 训练LSTM模型
    print("Training LSTM Model...")
    lstm_model = LSTMPredictor(input_size=input_dim)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_model(lstm_model, train_loader, val_loader, criterion, lstm_optimizer, num_epochs, device, "lstm")
    
    # 训练Seq2Seq模型
    print("Training Seq2Seq Model...")
    seq2seq_model = Seq2SeqPredictor(input_size=input_dim)
    seq2seq_optimizer = optim.Adam(seq2seq_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_model(seq2seq_model, train_loader, val_loader, criterion, seq2seq_optimizer, num_epochs, device, "seq2seq")
    
    # 保存标签编码器
    import pickle
    with open('./model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("Training completed!")

if __name__ == "__main__":
    main()