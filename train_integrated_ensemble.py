import torch
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data_preprocess import load_train_data, encode_categorical_features, prepare_train_data
from integrated_ensemble import IntegratedEnsembleModel

def train_integrated_ensemble():
    """
    训练集成模型并计算动态权重
    """
    print("Training Integrated Ensemble Model...")
    
    # 创建必要的目录
    os.makedirs('./model', exist_ok=True)
    
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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5364)
    
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    
    # 创建集成模型
    input_dim = X_train.shape[1]
    ensemble_model = IntegratedEnsembleModel(input_dim=input_dim)
    
    # 加载所有预训练模型
    print("Loading pre-trained models...")
    ensemble_model.load_models('./model')
    
    # 计算动态权重
    weights = ensemble_model.calculate_weights(X_val, y_val)
    
    # 保存模型信息（包括权重）
    ensemble_model.save_model_info('./model/integrated_ensemble_info.pkl')
    
    # 保存标签编码器
    with open('./model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # 在验证集上评估集成模型
    print("Evaluating integrated ensemble model on validation set...")
    val_predictions = ensemble_model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    print(f"Integrated Ensemble Model Validation MSE: {val_mse:.4f}")
    
    return ensemble_model, weights

def main():
    """
    主函数
    """
    model, weights = train_integrated_ensemble()
    print("Integrated ensemble model training completed!")
    print(f"Model weights: {weights}")

if __name__ == "__main__":
    main()