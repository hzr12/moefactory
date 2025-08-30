import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
import catboost as cat
from torch.utils.data import DataLoader, TensorDataset, Dataset
# 直接复制数据处理部分
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

#多模融合

class TrainDelayDataset(Dataset):
    def __init__(self, data, targets=None):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return torch.FloatTensor(self.data[idx]), torch.FloatTensor([self.targets[idx]])
        else:
            return torch.FloatTensor(self.data[idx])

def load_train_data(train_dir):
    """
    加载训练数据
    """
    all_data = []
    
    for filename in os.listdir(train_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(train_dir, filename)
            df = pd.read_csv(file_path)
            all_data.append(df)
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def load_test_data(test_file):
    """
    加载测试数据
    """
    df = pd.read_csv(test_file)
    return df

def extract_time_features(df, is_train=True):
    """
    提取时间特征
    """
    df = df.copy()
    
    # 处理出发时间特征
    if '出发时间' in df.columns and not df['出发时间'].isna().all():
        # 分离小时和分钟
        departure_time = df['出发时间'].str.split(':', expand=True)
        # 处理NaN值，用0填充
        df['出发小时'] = pd.to_numeric(departure_time[0], errors='coerce').fillna(0).astype(int)
        df['出发分钟'] = pd.to_numeric(departure_time[1], errors='coerce').fillna(0).astype(int)
    else:
        df['出发小时'] = 0
        df['出发分钟'] = 0
    
    # 处理日期特征
    if '出发日期' in df.columns and not df['出发日期'].isna().all():
        departure_date = pd.to_datetime(df['出发日期'], errors='coerce').fillna(pd.Timestamp.now())
        df['出发月份'] = departure_date.dt.month
        df['出发日'] = departure_date.dt.day
        df['出发星期'] = departure_date.dt.dayofweek
    else:
        df['出发月份'] = 1
        df['出发日'] = 1
        df['出发星期'] = 0

    # 车站与时间段的交互
    df['车站_小时交互'] = df['车站编码'] * df['出发小时']
    df['车站_星期交互'] = df['车站编码'] * df['出发星期']

    # 时间特征交互
    df['小时_星期交互'] = df['出发小时'] * df['出发星期']

    # 处理到达时间特征
    if '到达时间' in df.columns and not df['到达时间'].isna().all():
        arrival_time = df['到达时间'].str.split(':', expand=True)
        df['到达小时'] = pd.to_numeric(arrival_time[0], errors='coerce').fillna(0).astype(int)
        df['到达分钟'] = pd.to_numeric(arrival_time[1], errors='coerce').fillna(0).astype(int)
        df['到达时间_小时'] = pd.to_datetime(df['到达时间'], format='%H:%M', errors='coerce').dt.hour.fillna(0)
        df['到达时间_分钟'] = pd.to_datetime(df['到达时间'], format='%H:%M', errors='coerce').dt.minute.fillna(0)
        df['到达时间_小时_分钟'] = df['到达时间_小时'] * df['到达时间_分钟']
    else:
        df['到达小时'] = 0

    return df

def encode_categorical_features(train_df, test_df=None):
    """
    编码分类特征
    """
    # 处理车站名
    le_station = LabelEncoder()
    
    if test_df is not None:
        # 合并训练和测试数据进行编码
        all_stations = pd.concat([train_df['车站名'], test_df['车站名']], ignore_index=True)
        le_station.fit(all_stations)
        
        # 对训练数据编码
        try:
            train_df['车站编码'] = le_station.transform(train_df['车站名'])
        except ValueError:
            # 处理训练数据中可能存在的未知标签
            train_df['车站编码'] = 0
        
        # 对测试数据编码
        try:
            test_df['车站编码'] = le_station.transform(test_df['车站名'])
        except ValueError:
            # 处理测试数据中可能存在的未知标签
            test_df['车站编码'] = 0
            
        return train_df, test_df, le_station
    else:
        # 仅处理训练数据
        le_station.fit(train_df['车站名'])
        try:
            train_df['车站编码'] = le_station.transform(train_df['车站名'])
        except ValueError:
            train_df['车站编码'] = 0
        return train_df, le_station

def prepare_train_data(train_df):
    """
    准备训练数据
    """
    # 提取时间特征
    train_df = extract_time_features(train_df, is_train=True)
    # 选择特征列
    feature_columns = ['出发小时', '出发分钟', '出发月份', '出发日', '出发星期', '车站编码', '车站_小时交互', '车站_星期交互', '小时_星期交互', '到达小时', '到达分钟', '到达时间_小时', '到达时间_分钟', '到达时间_小时_分钟']
    X = train_df[feature_columns].values
    y = train_df['延误分钟'].fillna(0).values  # 处理目标值中的NaN
    
    return X, y

def prepare_test_data(test_df):
    """
    准备测试数据
    """
    # 提取时间特征
    test_df = extract_time_features(test_df, is_train=False)
    
    # 选择特征列
    feature_columns = ['出发小时', '出发分钟', '出发月份', '出发日', '出发星期', '车站编码', '车站_小时交互', '车站_星期交互', '小时_星期交互', '到达小时', '到达分钟', '到达时间_小时', '到达时间_分钟', '到达时间_小时_分钟']
    X = test_df[feature_columns].values
    
    return X


# 集成模型定义
class IntegratedEnsembleModel(nn.Module):
    """
    集成所有7个模型的集成模型：
    3个深度学习模型：Transformer、LSTM、Seq2Seq
    4个传统机器学习模型：随机森林、LightGBM、XGBoost、CatBoost
    """
    def __init__(self, input_dim=14):
        super(IntegratedEnsembleModel, self).__init__()
        self.n_features = input_dim
        
        # 深度学习模型
        self.transformer = TransformerPredictor(input_dim=input_dim)
        self.lstm = LSTMPredictor(input_size=input_dim)
        self.seq2seq = Seq2SeqPredictor(input_size=input_dim)
        
        # 传统机器学习模型（在训练时初始化）
        self.traditional_models = {}
        
        # 学习权重层 - 使用softmax确保权重和为1
        self.weights = nn.Parameter(torch.ones(7))  # 7个模型的权重
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        # 检查输入维度
        if x.dim() == 2:
            # 如果是2D张量 (batch_size, features)，扩展为3D (batch_size, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, but got {x.dim()}D")
        
        # 深度学习模型预测
        transformer_pred = self.transformer(x).squeeze()
        lstm_pred = self.lstm(x).squeeze()
        seq2seq_pred = self.seq2seq(x).squeeze()
        
        # 如果是单个样本，调整形状
        if transformer_pred.dim() == 0:
            transformer_pred = transformer_pred.unsqueeze(0)
            lstm_pred = lstm_pred.unsqueeze(0)
            seq2seq_pred = seq2seq_pred.unsqueeze(0)
        
        # 获取权重
        weights = self.softmax(self.weights)
        
        # 返回加权预测结果（深度学习模型部分）
        dl_prediction = weights[0] * transformer_pred + weights[1] * lstm_pred + weights[2] * seq2seq_pred
        
        return dl_prediction
    
    def load_traditional_models(self, model_dir='./model'):
        """
        加载传统机器学习模型
        """
        try:
            # 加载随机森林模型
            rf_path = os.path.join(model_dir, 'random_forest_best.pkl')
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.traditional_models['random_forest'] = pickle.load(f)
                    
            # 加载LightGBM模型
            lgb_path = os.path.join(model_dir, 'lightgbm_best.pkl')
            if os.path.exists(lgb_path):
                with open(lgb_path, 'rb') as f:
                    self.traditional_models['lightgbm'] = pickle.load(f)
                    
            # 加载XGBoost模型
            xgb_path = os.path.join(model_dir, 'xgboost_best.pkl')
            if os.path.exists(xgb_path):
                with open(xgb_path, 'rb') as f:
                    self.traditional_models['xgboost'] = pickle.load(f)
                    
            # 加载CatBoost模型
            cat_path = os.path.join(model_dir, 'catboost_best.pkl')
            if os.path.exists(cat_path):
                with open(cat_path, 'rb') as f:
                    self.traditional_models['catboost'] = pickle.load(f)
                    
        except Exception as e:
            print(f"Error loading traditional models: {e}")
    
    def predict_with_all_models(self, X_dl, X_traditional):
        """
        使用所有模型进行预测
        X_dl: 用于深度学习模型的输入 (3D张量)
        X_traditional: 用于传统机器学习模型的输入 (2D数组)
        """
        device = next(self.transformer.parameters()).device
        
        # 深度学习模型预测
        self.eval()
        with torch.no_grad():
            if X_dl.dim() == 2:
                X_dl = X_dl.unsqueeze(1)
            
            transformer_pred = self.transformer(X_dl).squeeze()
            lstm_pred = self.lstm(X_dl).squeeze()
            seq2seq_pred = self.seq2seq(X_dl).squeeze()
            
            # 转换为numpy数组
            transformer_pred = transformer_pred.cpu().numpy()
            lstm_pred = lstm_pred.cpu().numpy()
            seq2seq_pred = seq2seq_pred.cpu().numpy()
            
            # 确保是一维数组
            if transformer_pred.ndim == 0:
                transformer_pred = np.expand_dims(transformer_pred, axis=0)
                lstm_pred = np.expand_dims(lstm_pred, axis=0)
                seq2seq_pred = np.expand_dims(seq2seq_pred, axis=0)
        
        # 传统机器学习模型预测
        traditional_predictions = {}
        for model_name, model in self.traditional_models.items():
            try:
                pred = model.predict(X_traditional)
                traditional_predictions[model_name] = pred
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                traditional_predictions[model_name] = np.zeros(X_traditional.shape[0])
        
        # 获取权重
        weights = self.softmax(self.weights).detach().cpu().numpy()
        
        # 组合所有预测结果
        all_predictions = [transformer_pred, lstm_pred, seq2seq_pred] + list(traditional_predictions.values())
        
        # 加权平均
        final_predictions = np.zeros_like(transformer_pred)
        for i, pred in enumerate(all_predictions):
            final_predictions += weights[i] * pred.reshape(-1)
        
        return final_predictions, all_predictions, weights


# 从models.py复制模型定义
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=6, d_model=32, nhead=8, num_layers=5, dim_feedforward=128, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.n_features = input_dim  # 记录特征数量
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, src):
        # 检查输入维度
        if src.dim() == 2:
            # 如果是2D张量 (batch_size, features)，扩展为3D (batch_size, 1, features)
            src = src.unsqueeze(1)
        elif src.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, but got {src.dim()}D")
        
        # 输入投影
        src = self.input_projection(src)
        
        # 添加位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码
        output = self.transformer_encoder(src)
        
        # 全局平均池化
        output = output.mean(dim=1)
        
        # 输出层
        output = self.output_layer(output)
        return output

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=5, dropout=0.1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_features = input_size  # 记录特征数量
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # 检查输入维度
        if x.dim() == 2:
            # 如果是2D张量 (batch_size, features)，扩展为3D (batch_size, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, but got {x.dim()}D")
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        output = lstm_out[:, -1, :]
        
        # 应用dropout和全连接层
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

class Seq2SeqPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.2):
        super(Seq2SeqPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_features = input_size  # 记录特征数量
        
        # 编码器
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # 检查输入维度
        if x.dim() == 2:
            # 如果是2D张量 (batch_size, features)，扩展为3D (batch_size, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, but got {x.dim()}D")
        
        # 编码器
        _, (hidden, cell) = self.encoder(x)
        
        # 解码器
        decoder_output, _ = self.decoder(x, (hidden, cell))
        
        # 取最后一个时间步的输出
        output = decoder_output[:, -1, :]
        
        # 输出层
        output = self.output_layer(output)
        
        return output

import math

def train_integrated_ensemble(train_loader, val_loader, X_train_traditional, X_val_traditional, y_train_traditional, y_val_traditional, device, model_save_path='./model/integrated_ensemble_best.pth'):
    """
    训练集成模型
    """
    # 获取输入维度
    for batch in train_loader:
        input_dim = batch[0].shape[-1]
        break
    
    # 创建模型
    model = IntegratedEnsembleModel(input_dim=input_dim)
    
    # 加载预训练的基础模型
    try:
        model.transformer.load_state_dict(torch.load('./model/transformer_best.pth', map_location=device))
        model.lstm.load_state_dict(torch.load('./model/lstm_best.pth', map_location=device))
        model.seq2seq.load_state_dict(torch.load('./model/seq2seq_best.pth', map_location=device))
        print("Pre-trained deep learning models loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load pre-trained deep learning models: {e}")
    
    # 冻结深度学习模型参数
    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.lstm.parameters():
        param.requires_grad = False
    for param in model.seq2seq.parameters():
        param.requires_grad = False
    
    # 加载传统机器学习模型
    model.load_traditional_models()
    
    model.to(device)
    
    # 定义优化器（只优化权重参数）
    optimizer = optim.Adam([model.weights], lr=0.01)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    num_epochs = 200
    
    print("Starting integrated ensemble weight training...")
    print(f"Initial weights: {model.softmax(model.weights).detach().cpu().numpy()}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
        
        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze())
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
        
        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0
        scheduler.step(avg_val_loss)
        
        # 打印进度
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            weights = model.softmax(model.weights).detach().cpu().numpy()
            print(f"Epoch {epoch+1:03d}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"  Weights: T={weights[0]:.3f}, L={weights[1]:.3f}, S={weights[2]:.3f}")
            # 显示传统机器学习模型的权重
            if len(weights) >= 7:
                print(f"  Traditional Weights: RF={weights[3]:.3f}, LGB={weights[4]:.3f}, XGB={weights[5]:.3f}, CAT={weights[6]:.3f}")
        
        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    final_weights = model.softmax(model.weights).detach().cpu().numpy()
    print(f"Final weights: T={final_weights[0]:.3f}, L={final_weights[1]:.3f}, S={final_weights[2]:.3f}")
    if len(final_weights) >= 7:
        print(f"Traditional Weights: RF={final_weights[3]:.3f}, LGB={final_weights[4]:.3f}, XGB={final_weights[5]:.3f}, CAT={final_weights[6]:.3f}")
    
    return model


def main():
    """
    主函数 - 训练集成模型
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载训练数据
    print("Loading training data...")
    train_df = load_train_data('./datasets/train')
    print(f"Loaded {len(train_df)} training samples")
    
    # 编码分类特征
    train_df, label_encoder = encode_categorical_features(train_df)
    
    # 准备训练数据
    X, y = prepare_train_data(train_df)
    
    # 处理NaN值
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0).astype(np.float32)
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=5364)
    
    # 创建数据集和数据加载器
    train_dataset = TrainDelayDataset(X_train, y_train)
    val_dataset = TrainDelayDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=17, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=17, shuffle=False)
    
    # 训练集成模型
    ensemble_model = train_integrated_ensemble(
        train_loader, val_loader, 
        X_train, X_val, 
        y_train, y_val, 
        device
    )
    
    print("Integrated ensemble training completed!")


if __name__ == "__main__":
    main()