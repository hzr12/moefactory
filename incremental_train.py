import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import pickle

from data_preprocess import (load_train_data, encode_categorical_features, prepare_train_data,
                             TrainDelayDataset, split_by_date, prepare_sequence_data,
                             split_sequence_by_date, JourneySequenceDataset, MaskedMSELoss)
from models import EnsembleModel, TransformerPredictor, LSTMPredictor, Seq2SeqPredictor

#增量学习

def load_trained_model(model, model_path):
    """
    加载预训练模型
    """
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
    return model


def load_label_encoder(encoder_path):
    """
    加载标签编码器
    """
    try:
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"Successfully loaded label encoder from {encoder_path}")
        return label_encoder
    except Exception as e:
        print(f"Failed to load label encoder from {encoder_path}: {e}")
        return None


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_name,
                patience=50):
    """
    训练模型，增加早停机制
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
        for inputs, targets, lengths in train_loader:
            inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets, lengths)

            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in {model_name} at epoch {epoch + 1}")
                continue

            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_samples += int(lengths.sum())

        # 计算平均训练损失
        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_samples = 0

        # 遍历验证数据，不显示进度条
        with torch.no_grad():
            for inputs, targets, lengths in val_loader:
                inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
                outputs = model(inputs, lengths)
                loss = criterion(outputs, targets, lengths)
                # 检查损失是否为NaN
                if torch.isnan(loss):
                    print(f"Warning: NaN validation loss encountered in {model_name} at epoch {epoch + 1}")
                    continue
                val_loss += loss.item() * inputs.size(0)
                val_samples += int(lengths.sum())

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
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
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
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from catboost import CatBoostRegressor
        from sklearn.metrics import mean_squared_error
        import pickle

        # 加载已有的模型（如果存在）
        rf_model = None
        lgb_model = None
        xgb_model = None
        cat_model = None

        # 尝试加载已有的随机森林模型
        try:
            with open('./model/random_forest_best.pkl', 'rb') as f:
                rf_model = pickle.load(f)
            print("Loaded existing Random Forest model")
        except FileNotFoundError:
            print("No existing Random Forest model found, training from scratch")
            rf_model = RandomForestRegressor(
                n_estimators=1000,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=3461
            )

        # 尝试加载已有的LightGBM模型
        try:
            with open('./model/lightgbm_best.pkl', 'rb') as f:
                lgb_model = pickle.load(f)
            print("Loaded existing LightGBM model")
        except FileNotFoundError:
            print("No existing LightGBM model found, training from scratch")
            lgb_model = LGBMRegressor(
                num_leaves=127,
                n_estimators=500
            )
            
        # 尝试加载已有的XGBoost模型
        try:
            with open('./model/xgboost_best.pkl', 'rb') as f:
                xgb_model = pickle.load(f)
            print("Loaded existing XGBoost model")
        except FileNotFoundError:
            print("No existing XGBoost model found, training from scratch")
            xgb_model = XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.1,
                random_state=3461
            )
            
        # 尝试加载已有的CatBoost模型
        try:
            with open('./model/catboost_best.pkl', 'rb') as f:
                cat_model = pickle.load(f)
            print("Loaded existing CatBoost model")
        except FileNotFoundError:
            print("No existing CatBoost model found, training from scratch")
            cat_model = CatBoostRegressor(
                iterations=1000,
                depth=6,
                learning_rate=0.1,
                verbose=False,
                random_state=3461
            )

        # 继续训练随机森林模型
        print("Training Random Forest Model...")
        rf_model.fit(X_train, y_train)

        # 验证随机森林模型
        rf_pred = rf_model.predict(X_val)
        rf_mse = mean_squared_error(y_val, rf_pred)
        print(f"Random Forest Val MSE: {rf_mse:.4f}")

        # 保存随机森林模型
        with open('./model/random_forest_best.pkl', 'wb') as f:
            pickle.dump(rf_model, f)

        # 继续训练LightGBM模型
        print("Training LightGBM Model...")
        lgb_model.fit(X_train, y_train)

        # 验证LightGBM模型
        lgb_pred = lgb_model.predict(X_val)
        lgb_mse = mean_squared_error(y_val, lgb_pred)
        print(f"LightGBM Val MSE: {lgb_mse:.4f}")

        # 保存LightGBM模型
        with open('./model/lightgbm_best.pkl', 'wb') as f:
            pickle.dump(lgb_model, f)
            
        # 继续训练XGBoost模型
        print("Training XGBoost Model...")
        xgb_model.fit(X_train, y_train)

        # 验证XGBoost模型
        xgb_pred = xgb_model.predict(X_val)
        xgb_mse = mean_squared_error(y_val, xgb_pred)
        print(f"XGBoost Val MSE: {xgb_mse:.4f}")

        # 保存XGBoost模型
        with open('./model/xgboost_best.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
            
        # 继续训练CatBoost模型
        print("Training CatBoost Model...")
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
    # 创建必要的目录
    os.makedirs('./model', exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载原始标签编码器
    label_encoder = load_label_encoder('./model/label_encoder.pkl')

    # 加载增量训练数据
    print("Loading incremental training data...")
    incremental_train_df = load_train_data('./datasets/incremental/train')
    print(f"Loaded {len(incremental_train_df)} incremental training samples")

    # 如果存在已训练的标签编码器，则使用它进行编码
    if label_encoder is not None:
        try:
            incremental_train_df['车站编码'] = label_encoder.transform(incremental_train_df['车站名'])
        except ValueError as e:
            print(f"Warning: Some station names in incremental data not found in label encoder: {e}")
            # 对于未知的车站名，使用默认编码0
            incremental_train_df['车站编码'] = 0
    else:
        # 如果没有标签编码器，重新创建（这种情况应该很少发生）
        incremental_train_df, label_encoder = encode_categorical_features(incremental_train_df)
        # 保存新的标签编码器
        with open('./model/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)

    # 车次编码（新数据格式新增特征）：优先用训练时保存的车次编码器，未知车次记 0
    if '车次编码' not in incremental_train_df.columns:
        le_train = None
        train_le_path = './model/train_label_encoder.pkl'
        if os.path.exists(train_le_path):
            try:
                with open(train_le_path, 'rb') as f:
                    le_train = pickle.load(f)
            except Exception:
                le_train = None
        if le_train is not None:
            mapping = {label: idx for idx, label in enumerate(le_train.classes_)}
            incremental_train_df['车次编码'] = [mapping.get(str(v), 0)
                                              for v in incremental_train_df['车次ID'].astype(str)]
        else:
            incremental_train_df['车次编码'] = 0

    # 准备训练数据
    X, y = prepare_train_data(incremental_train_df)

    # 检查是否有NaN值并处理
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"NaN in X: {np.isnan(X).sum()}, NaN in y: {np.isnan(y).sum()}")

    # 处理NaN值
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    # 按“到达日期”分组留出验证集（与 train.py 一致，避免日期泄漏）
    X_train, X_val, y_train, y_val, val_start, val_end = split_by_date(incremental_train_df, X, y, val_ratio=0.1)
    print(f"训练集: {len(X_train)} 行 / 验证集: {len(X_val)} 行")
    print(f"验证集日期范围: {str(val_start)[:10]} ~ {str(val_end)[:10]}")

    # 训练传统机器学习模型
    train_traditional_models(X_train, y_train, X_val, y_val)

    # 创建数据集和数据加载器
    # 深度学习模型同样改用“行程序列”输入
    seq = prepare_sequence_data(incremental_train_df)
    seq_train, seq_val, seq_start, seq_end = split_sequence_by_date(seq, 0.1)
    print(f"行程序列: {seq['X'].shape[0]} 个行程，最长 {seq['X'].shape[1]} 站")

    train_loader = DataLoader(JourneySequenceDataset(seq_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(JourneySequenceDataset(seq_val), batch_size=32, shuffle=False)

    # 模型参数
    input_dim = seq['X'].shape[2]
    num_epochs = 200  # 减少训练轮数以适应增量学习
    learning_rate = 0.0001  # 使用较小的学习率进行微调

    print(f"Input dimension: {input_dim}")

    # 定义损失函数（只在真实站点上计算，忽略 padding）
    criterion = MaskedMSELoss()

    # 训练集成模型
    print("Incremental Training Ensemble Model...")
    ensemble_model = EnsembleModel(input_dim=input_dim)
    # 加载已有的模型权重
    ensemble_model = load_trained_model(ensemble_model, './model/ensemble_best.pth')
    ensemble_optimizer = optim.Adam(ensemble_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    ensemble_scheduler = optim.lr_scheduler.ReduceLROnPlateau(ensemble_optimizer, mode='min', patience=10, factor=0.9, min_lr=1e-6)
    train_model(ensemble_model, train_loader, val_loader, criterion, ensemble_optimizer, ensemble_scheduler, num_epochs,
                device, "ensemble")

    # 训练Transformer模型
    print("Incremental Training Transformer Model...")
    transformer_model = TransformerPredictor(input_dim=input_dim)
    # 加载已有的模型权重
    transformer_model = load_trained_model(transformer_model, './model/transformer_best.pth')
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    transformer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(transformer_optimizer, mode='min', patience=10, factor=0.9, min_lr=1e-6)
    train_model(transformer_model, train_loader, val_loader, criterion, transformer_optimizer, transformer_scheduler,
                num_epochs, device, "transformer")

    # 训练LSTM模型
    print("Incremental Training LSTM Model...")
    lstm_model = LSTMPredictor(input_size=input_dim)
    # 加载已有的模型权重
    lstm_model = load_trained_model(lstm_model, './model/lstm_best.pth')
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(lstm_optimizer, mode='min', patience=10, factor=0.9, min_lr=1e-6)
    train_model(lstm_model, train_loader, val_loader, criterion, lstm_optimizer, lstm_scheduler, num_epochs, device,
                "lstm")

    # 训练Seq2Seq模型
    print("Incremental Training Seq2Seq Model...")
    seq2seq_model = Seq2SeqPredictor(input_size=input_dim)
    # 加载已有的模型权重
    seq2seq_model = load_trained_model(seq2seq_model, './model/seq2seq_best.pth')
    seq2seq_optimizer = optim.Adam(seq2seq_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    seq2seq_scheduler = optim.lr_scheduler.ReduceLROnPlateau(seq2seq_optimizer, mode='min', patience=10, factor=0.9, min_lr=1e-6)
    train_model(seq2seq_model, train_loader, val_loader, criterion, seq2seq_optimizer, seq2seq_scheduler, num_epochs,
                device, "seq2seq")

    print("Incremental training completed!")


if __name__ == "__main__":
    main()