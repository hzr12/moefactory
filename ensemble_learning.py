import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from new_data_preprocess import load_train_data, encode_categorical_features, prepare_train_data, TrainDelayDataset, load_test_data, prepare_test_data
from models import TransformerPredictor, LSTMPredictor, Seq2SeqPredictor


class WeightedEnsembleModel(nn.Module):
    """
    加权集成模型，通过学习权重来融合多个模型的预测结果
    """
    def __init__(self, input_dim=14):
        super(WeightedEnsembleModel, self).__init__()
        self.n_features = input_dim
        
        # 基础模型
        self.transformer = TransformerPredictor(input_dim=input_dim)
        self.lstm = LSTMPredictor(input_size=input_dim)
        self.seq2seq = Seq2SeqPredictor(input_size=input_dim)
        
        # 传统机器学习模型占位符
        self.traditional_models = {}
        
        # 学习权重层 - 使用softmax确保权重和为1
        self.weights = nn.Parameter(torch.ones(7))  # 7个模型 (4个深度学习+3个传统机器学习)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        # 检查输入维度
        if x.dim() == 2:
            # 如果是2D张量 (batch_size, features)，扩展为3D (batch_size, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, but got {x.dim()}D")
        
        # 各个深度学习模型的预测
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
        
        # 返回加权预测结果
        return weights[0] * transformer_pred + weights[1] * lstm_pred + weights[2] * seq2seq_pred
    
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
    
    def predict_with_traditional_models(self, X):
        """
        使用传统机器学习模型进行预测
        """
        predictions = {}
        
        for model_name, model in self.traditional_models.items():
            try:
                pred = model.predict(X)
                predictions[model_name] = pred
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                predictions[model_name] = np.zeros(X.shape[0])
                
        return predictions


def train_weighted_ensemble(train_loader, val_loader, device, model_save_path='./model/weighted_ensemble_best.pth'):
    """
    训练加权集成模型
    """
    # 获取输入维度
    for batch in train_loader:
        input_dim = batch[0].shape[-1]
        break
    
    # 创建模型
    model = WeightedEnsembleModel(input_dim=input_dim)
    
    # 加载预训练的基础模型
    try:
        model.transformer.load_state_dict(torch.load('./model/transformer_best.pth', map_location=device))
        model.lstm.load_state_dict(torch.load('./model/lstm_best.pth', map_location=device))
        model.seq2seq.load_state_dict(torch.load('./model/seq2seq_best.pth', map_location=device))
        print("Pre-trained models loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load pre-trained models: {e}")
    
    # 冻结基础模型参数
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
    
    print("Starting ensemble weight training...")
    print(f"Initial weights: {model.softmax(model.weights).detach().cpu().numpy()}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for inputs, targets in train_loader:
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
    
    return model


def predict_with_ensemble(model, test_loader, X_test_traditional, device):
    """
    使用集成模型进行预测
    """
    # 深度学习模型预测
    model.eval()
    dl_predictions = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            dl_predictions.extend(outputs.cpu().numpy())
    
    dl_predictions = np.array(dl_predictions)
    
    # 传统机器学习模型预测
    traditional_predictions = model.predict_with_traditional_models(X_test_traditional)
    
    # 获取权重
    weights = model.softmax(model.weights).detach().cpu().numpy()
    
    # 组合所有预测结果
    all_predictions = [dl_predictions] + list(traditional_predictions.values())
    
    # 加权平均
    final_predictions = np.zeros_like(dl_predictions)
    for i, pred in enumerate(all_predictions):
        if i < 3:  # 深度学习模型
            final_predictions += weights[i] * pred.reshape(-1, 1)
        else:  # 传统机器学习模型 (这里简化处理，实际应该有各自的权重)
            final_predictions += weights[i] * pred.reshape(-1, 1)
    
    return final_predictions.flatten(), traditional_predictions, dl_predictions


def main():
    """
    主函数 - 训练和测试集成模型
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
    
    # 训练加权集成模型
    ensemble_model = train_weighted_ensemble(train_loader, val_loader, device)
    
    print("Ensemble training completed!")


if __name__ == "__main__":
    main()