import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

from data_preprocess import (load_train_data, encode_categorical_features, prepare_train_data,
                             TrainDelayDataset, split_by_date, prepare_sequence_data,
                             split_sequence_by_date, JourneySequenceDataset, MaskedMSELoss,
                             build_gnn_graph, save_graph_hist_state)
from models import (EnsembleModel, TransformerPredictor, LSTMPredictor, Seq2SeqPredictor, TFT,
                    StationGNN, TCNLite, set_gnn_graph)

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


def _load_best_params():
    """读取 tune_models.py 搜索得到的最优参数（若存在），否则返回空字典。"""
    import json
    p = './model/best_params.json'
    if os.path.exists(p):
        try:
            with open(p, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: 无法读取最优参数 {p}: {e}")
    return {}


def train_traditional_models(X_train, y_train, X_val, y_val):
    """
    训练传统机器学习模型（Random Forest, LightGBM, XGBoost, CatBoost）。
    若 model/best_params.json 存在，则用其中的超参数覆盖默认配置（由 tune_models.py 产出）。
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from catboost import CatBoostRegressor
        from sklearn.metrics import mean_squared_error
        import pickle

        best = _load_best_params()

        # 默认配置（被调参证实合理的起点）；best_params.json 中的值会覆盖它们
        rf_defaults = dict(n_estimators=1000, max_depth=10, min_samples_split=5,
                           min_samples_leaf=2, n_jobs=-1, random_state=3461)
        lgb_defaults = dict(num_leaves=63, learning_rate=0.01, n_estimators=500, verbose=-1, random_state=3461)
        # XGBoost 必须用温和的学习率 + 子采样/列采样 + L2 正则，否则在 4000+ 样本上严重过拟合
        xgb_defaults = dict(n_estimators=800, max_depth=6, learning_rate=0.02, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=5, min_child_weight=5,
                            tree_method='hist', random_state=3461)
        cat_defaults = dict(iterations=1000, depth=6, learning_rate=0.1, verbose=False, random_state=3461)

        rf_params = {**rf_defaults, **best.get('random_forest', {})}
        lgb_params = {**lgb_defaults, **best.get('lightgbm', {})}
        xgb_params = {**xgb_defaults, **best.get('xgboost', {})}
        cat_params = {**cat_defaults, **best.get('catboost', {})}
        if best:
            print(f"已载入 tune_models.py 的最优参数，覆盖模型: {list(best.keys())}")

        # 训练随机森林模型
        print("Training Random Forest Model...")
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)
        rf_mse = mean_squared_error(y_val, rf_model.predict(X_val))
        print(f"Random Forest Val MSE: {rf_mse:.4f}")
        with open('./model/random_forest_best.pkl', 'wb') as f:
            pickle.dump(rf_model, f)

        # 训练LightGBM模型
        print("Training LightGBM Model...")
        lgb_model = LGBMRegressor(**lgb_params)
        lgb_model.fit(X_train, y_train)
        lgb_mse = mean_squared_error(y_val, lgb_model.predict(X_val))
        print(f"LightGBM Val MSE: {lgb_mse:.4f}")
        with open('./model/lightgbm_best.pkl', 'wb') as f:
            pickle.dump(lgb_model, f)

        # 训练XGBoost模型
        print("Training XGBoost Model...")
        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train)
        xgb_mse = mean_squared_error(y_val, xgb_model.predict(X_val))
        print(f"XGBoost Val MSE: {xgb_mse:.4f}")
        with open('./model/xgboost_best.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)

        # 训练CatBoost模型
        print("Training CatBoost Model...")
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_train, y_train)
        cat_mse = mean_squared_error(y_val, cat_model.predict(X_val))
        print(f"CatBoost Val MSE: {cat_mse:.4f}")
        with open('./model/catboost_best.pkl', 'wb') as f:
            pickle.dump(cat_model, f)

    except ImportError as e:
        print(f"Warning: Could not import traditional ML libraries: {e}")
        print("Skipping traditional ML model training...")

def main():
    """
    主函数，负责整个训练流程
    """
    parser = argparse.ArgumentParser(description='训练延误预测模型')
    parser.add_argument('--skip-ml', action='store_true',
                        help='跳过 4 个传统机器学习模型(RF/LGB/XGB/CatBoost)')
    parser.add_argument('--epochs', type=int, default=500, help='DL 训练轮数')
    args = parser.parse_args()

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

    # 按“到达日期”分组留出验证集：避免同一天的行分到两侧，导致按天常量的天气特征变成日期指纹
    X_train, X_val, y_train, y_val, val_start, val_end = split_by_date(train_df, X, y, val_ratio=0.1)
    print(f"训练集: {len(X_train)} 行 / 验证集: {len(X_val)} 行")
    print(f"验证集日期范围: {str(val_start)[:10]} ~ {str(val_end)[:10]}")

    # 训练传统机器学习模型（--skip-ml 可跳过）
    if not args.skip_ml:
        train_traditional_models(X_train, y_train, X_val, y_val)
    else:
        print("跳过传统机器学习模型 (--skip-ml)")

    # 创建数据集和数据加载器
    # 深度学习模型改用“行程序列”输入：一趟车沿途各站构成一个时间步
    seq = prepare_sequence_data(train_df)
    seq_train, seq_val, seq_start, seq_end = split_sequence_by_date(seq, 0.1)
    print(f"行程序列: {seq['X'].shape[0]} 个行程，最长 {seq['X'].shape[1]} 站")
    print(f"序列验证集日期范围: {str(seq_start)[:10]} ~ {str(seq_end)[:10]}")

    train_loader = DataLoader(JourneySequenceDataset(seq_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(JourneySequenceDataset(seq_val), batch_size=32, shuffle=False)

    # 模型参数
    input_dim = seq['X'].shape[2]
    num_epochs = args.epochs
    learning_rate = 0.001

    print(f"Input dimension: {input_dim}")

    # 构建站点图并注入 StationGNN（真实铁路网邻接，边权=距离反比）
    stations = list(label_encoder.classes_)
    edge_index, edge_weight, num_nodes = build_gnn_graph(stations)
    set_gnn_graph(edge_index, edge_weight, num_nodes)
    torch.save({'edge_index': edge_index, 'edge_weight': edge_weight,
                'num_nodes': num_nodes}, './model/station_graph_gnn.pkl')

    # 定义损失函数（只在真实站点上计算，忽略 padding）
    criterion = MaskedMSELoss()

    # 训练集成模型
    print("Training Ensemble Model...")
    ensemble_model = EnsembleModel(input_dim=input_dim)
    ensemble_optimizer = optim.Adam(ensemble_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    ensemble_scheduler = optim.lr_scheduler.ReduceLROnPlateau(ensemble_optimizer, mode='min', patience=10, factor=0.9, min_lr=1e-5)
    train_model(ensemble_model, train_loader, val_loader, criterion, ensemble_optimizer, ensemble_scheduler, num_epochs, device, "ensemble")

    # 训练Transformer模型
    print("Training Transformer Model...")
    transformer_model = TransformerPredictor(input_dim=input_dim)
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    transformer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(transformer_optimizer, mode='min', patience=10, factor=0.9, min_lr=1e-5)
    train_model(transformer_model, train_loader, val_loader, criterion, transformer_optimizer, transformer_scheduler, num_epochs, device, "transformer")

    # 训练LSTM模型
    print("Training LSTM Model...")
    lstm_model = LSTMPredictor(input_size=input_dim)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(lstm_optimizer, mode='min', patience=10, factor=0.9, min_lr=1e-5)
    train_model(lstm_model, train_loader, val_loader, criterion, lstm_optimizer, lstm_scheduler, num_epochs, device, "lstm")

    # 训练Seq2Seq模型
    print("Training Seq2Seq Model...")
    seq2seq_model = Seq2SeqPredictor(input_size=input_dim)
    seq2seq_optimizer = optim.Adam(seq2seq_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    seq2seq_scheduler = optim.lr_scheduler.ReduceLROnPlateau(seq2seq_optimizer, mode='min', patience=10, factor=0.9, min_lr=1e-5)
    train_model(seq2seq_model, train_loader, val_loader, criterion, seq2seq_optimizer, seq2seq_scheduler, num_epochs, device, "seq2seq")

    # 训练TFT模型
    print("Training TFT Model...")
    tft_model = TFT(input_dim=input_dim)
    tft_optimizer = optim.Adam(tft_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    tft_scheduler = optim.lr_scheduler.ReduceLROnPlateau(tft_optimizer, mode='min', patience=10, factor=0.9, min_lr=1e-5)
    train_model(tft_model, train_loader, val_loader, criterion, tft_optimizer, tft_scheduler, num_epochs, device, "tft")

    # 训练StationGNN（图神经网络：延误沿铁路网传播）
    print("Training StationGNN Model...")
    stgnn_model = StationGNN(input_dim=input_dim, num_nodes=num_nodes)
    stgnn_optimizer = optim.Adam(stgnn_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    stgnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(stgnn_optimizer, mode='min', patience=10, factor=0.9, min_lr=1e-5)
    train_model(stgnn_model, train_loader, val_loader, criterion, stgnn_optimizer, stgnn_scheduler, num_epochs, device, "stgnn")

    # 训练TCN-lite（轻量因果卷积，提供集成多样性）
    print("Training TCN-lite Model...")
    tcn_model = TCNLite(input_dim=input_dim)
    tcn_optimizer = optim.Adam(tcn_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    tcn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(tcn_optimizer, mode='min', patience=10, factor=0.9, min_lr=1e-5)
    train_model(tcn_model, train_loader, val_loader, criterion, tcn_optimizer, tcn_scheduler, num_epochs, device, "tcnlite")

    # 保存标签编码器（车站 + 车次）
    import pickle
    with open('./model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    encoders = train_df.attrs.get('label_encoders') or {}
    if encoders.get('train') is not None:
        with open('./model/train_label_encoder.pkl', 'wb') as f:
            pickle.dump(encoders['train'], f)

    # 保存站点历史延误状态（供预测路径的图历史特征查表）
    save_graph_hist_state(train_df)

    print("Training completed!")

if __name__ == "__main__":
    main()