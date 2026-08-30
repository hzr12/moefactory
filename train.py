import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

from data_preprocess import load_train_data, encode_categorical_features, prepare_train_data, TrainDelayDataset
from models import EnsembleModel, TransformerPredictor, LSTMPredictor, Seq2SeqPredictor, TFT

"""
模型训练脚本
负责训练各种预测模型，包括深度学习模型和传统机器学习模型
"""

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_name,
                patience=50):
    """
    训练模型，增加早停机制
    
    Args:
        model (torch.nn.Module): 要训练的模型
        train_loader (torch.utils.data.DataLoader): 训练数据加载器
        val_loader (torch.utils.data.DataLoader): 验证数据加载器
        criterion (torch.nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器
        num_epochs (int): 训练轮数
        device (torch.device): 训练设备
        model_name (str): 模型名称
        patience (int): 早停耐心值
        
    Returns:
        torch.nn.Module: 训练后的模型
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    # 创建主进度条
    epoch_pbar = tqdm(range(num_epochs), desc=f'{model_name}')

    for epoch in epoch_pbar:
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_samples = 0

        # 遍历训练数据，不显示进度条
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in {model_name} at epoch {epoch + 1}")
                continue

            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)

        # 计算平均训练损失
        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_samples = 0

        # 遍历验证数据，不显示进度条
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # 检查损失是否为NaN
                if torch.isnan(loss):
                    print(f"Warning: NaN validation loss encountered in {model_name} at epoch {epoch + 1}")
                    continue
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)

        # 计算平均验证损失
        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0

        # 更新学习率
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # 检查损失是否为NaN
        if np.isnan(avg_train_loss) or np.isnan(avg_val_loss):
            print(f"Warning: NaN loss detected in {model_name} at epoch {epoch + 1}, stopping training")
            break

        # 更新epoch进度条描述，显示训练和验证损失
        epoch_pbar.set_postfix({
            'loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'./model/{model_name}_best.pth')
            epoch_pbar.write(f'Epoch {epoch + 1:04d}: saving model to ./model/{model_name}_best.pth')
        else:
            patience_counter += 1
            if patience_counter > patience:
                epoch_pbar.write(f'Early stopping at epoch {epoch + 1} for {model_name}')
                break

    return model


def train_traditional_models(X_train, y_train, X_val, y_val):
    """
    训练传统机器学习模型
    
    Args:
        X_train (numpy.ndarray): 训练特征
        y_train (numpy.ndarray): 训练目标值
        X_val (numpy.ndarray): 验证特征
        y_val (numpy.ndarray): 验证目标值
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from catboost import CatBoostRegressor
        from sklearn.metrics import mean_squared_error
        import pickle
        
        # 训练随机森林模型
        print("Training Random Forest Model...")
        rf_model = RandomForestRegressor(
            n_estimators=10000,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=3461
        )
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
        lgb_model = LGBMRegressor(
            num_leaves=127,
            learning_rate=0.01,
            n_estimators=500
        )
        lgb_model.fit(X_train, y_train)
        
        # 验证LightGBM模型
        lgb_pred = lgb_model.predict(X_val)
        lgb_mse = mean_squared_error(y_val, lgb_pred)
        print(f"LightGBM Val MSE: {lgb_mse:.4f}")
        
        # 保存LightGBM模型
        with open('./model/lightgbm_best.pkl', 'wb') as f:
            pickle.dump(lgb_model, f)
            
        # 训练XGBoost模型
        print("Training XGBoost Model...")
        xgb_model = XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            random_state=3461
        )
        xgb_model.fit(X_train, y_train)
        
        # 验证XGBoost模型
        xgb_pred = xgb_model.predict(X_val)
        xgb_mse = mean_squared_error(y_val, xgb_pred)
        print(f"XGBoost Val MSE: {xgb_mse:.4f}")
        
        # 保存XGBoost模型
        with open('./model/xgboost_best.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
            
        # 训练CatBoost模型
        print("Training CatBoost Model...")
        cat_model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            random_state=3461
        )
        cat_model.fit(X_train, y_train)
        
        # 验证CatBoost模型
        cat_pred = cat_model.predict(X_val)
        cat_mse = mean_squared_error(y_val, cat_pred)
        print(f"CatBoost Val MSE: {cat_mse:.4f}")
        
        # 保存CatBoost模型
        with open('./model/catboost_best.pkl', 'wb') as f:
            pickle.dump(cat_model, f)
            
    except ImportError as e:
        print(f"Warning: Could not import traditional ML libraries: {e}")
        print("Skipping traditional ML model training...")

def main():
    """
    主函数，负责整个训练流程
    """
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
    
    train_loader = DataLoader(train_dataset, batch_size=17, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=17, shuffle=False)
    
    # 模型参数
    input_dim = X_train.shape[1]
    num_epochs = 500  # 训练轮数
    learning_rate = 0.001
    
    print(f"Input dimension: {input_dim}")
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 训练集成模型
    print("Training Ensemble Model...")
    ensemble_model = EnsembleModel(input_dim=input_dim)
    ensemble_optimizer = optim.Adam(ensemble_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    ensemble_scheduler = optim.lr_scheduler.ReduceLROnPlateau(ensemble_optimizer, mode='min', patience=10, factor=0.9)
    train_model(ensemble_model, train_loader, val_loader, criterion, ensemble_optimizer, ensemble_scheduler, num_epochs, device, "ensemble")
    
    # 训练Transformer模型
    print("Training Transformer Model...")
    transformer_model = TransformerPredictor(input_dim=input_dim)
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    transformer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(transformer_optimizer, mode='min', patience=10, factor=0.9)
    train_model(transformer_model, train_loader, val_loader, criterion, transformer_optimizer, transformer_scheduler, num_epochs, device, "transformer")
    
    # 训练LSTM模型
    print("Training LSTM Model...")
    lstm_model = LSTMPredictor(input_size=input_dim)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(lstm_optimizer, mode='min', patience=10, factor=0.9)
    train_model(lstm_model, train_loader, val_loader, criterion, lstm_optimizer, lstm_scheduler, num_epochs, device, "lstm")
    
    # 训练Seq2Seq模型
    print("Training Seq2Seq Model...")
    seq2seq_model = Seq2SeqPredictor(input_size=input_dim)
    seq2seq_optimizer = optim.Adam(seq2seq_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    seq2seq_scheduler = optim.lr_scheduler.ReduceLROnPlateau(seq2seq_optimizer, mode='min', patience=10, factor=0.9)
    train_model(seq2seq_model, train_loader, val_loader, criterion, seq2seq_optimizer, seq2seq_scheduler, num_epochs, device, "seq2seq")
    
    # 训练TFT模型
    print("Training TFT Model...")
    tft_model = TFT(input_dim=input_dim)
    tft_optimizer = optim.Adam(tft_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    tft_scheduler = optim.lr_scheduler.ReduceLROnPlateau(tft_optimizer, mode='min', patience=10, factor=0.9)
    train_model(tft_model, train_loader, val_loader, criterion, tft_optimizer, tft_scheduler, num_epochs, device, "tft")
    
    # 保存标签编码器
    import pickle
    with open('./model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("Training completed!")

if __name__ == "__main__":
    main()