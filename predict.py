import torch
import pandas as pd
import numpy as np
import os
import pickle
import argparse

from data_preprocess import load_test_data, encode_categorical_features, prepare_test_data, load_train_data
from models import EnsembleModel, TransformerPredictor, LSTMPredictor, Seq2SeqPredictor

def load_trained_model(model, model_path, device):
    """
    加载训练好的模型
    """
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None
    return model

def load_traditional_model(model_path):
    """
    加载传统机器学习模型
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Successfully loaded traditional model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading traditional model from {model_path}: {e}")
        return None

def predict_with_model(model, test_loader, device):
    """
    使用深度学习模型进行预测
    """
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)  # 从tuple中提取tensor
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
    return np.array(predictions).flatten()

def predict_with_traditional_model(model, X_test):
    """
    使用传统机器学习模型进行预测
    """
    try:
        predictions = model.predict(X_test)
        return predictions
    except Exception as e:
        print(f"Error predicting with traditional model: {e}")
        return np.zeros(X_test.shape[0])

def save_predictions(test_df, predictions, model_name):
    """
    保存单个模型的预测结果
    """
    # 四舍五入为整数
    model_predictions = np.round(predictions).astype(int)
    
    # 创建结果DataFrame
    result_df = test_df[['车次ID', '车站名', '出发日期', '出发时间']].copy()
    result_df['预测延误分钟'] = model_predictions
    
    # 保存预测结果
    filename = f'./predictions/prediction_results_{model_name}.csv'
    result_df.to_csv(filename, index=False)
    print(f"{model_name} predictions saved to {filename}")

def main():
    # 创建必要的目录
    os.makedirs('./predictions', exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载测试数据
    print("Loading test data...")
    test_file = './20250826_G339.csv'
    test_df = load_test_data(test_file)
    print(f"Loaded {len(test_df)} test samples")
    
    # 加载训练数据以获取标签编码器
    train_df = load_train_data('./datasets/train')
    
    # 编码分类特征
    train_df, test_df, label_encoder = encode_categorical_features(train_df, test_df)
    
    # 准备测试数据
    X_test = prepare_test_data(test_df)
    
    # 处理NaN值
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"NaN in test data: {np.isnan(X_test).sum()}")
    
    # 创建测试数据加载器
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 输入维度
    input_dim = X_test.shape[1]
    
    # 加载模型并进行预测
    predictions = {}
    
    # 随机森林模型预测
    print("Predicting with Random Forest Model...")
    rf_model_path = './model/random_forest_best.pkl'
    if os.path.exists(rf_model_path):
        rf_model = load_traditional_model(rf_model_path)
        if rf_model is not None:
            rf_preds = predict_with_traditional_model(rf_model, X_test)
            predictions['random_forest'] = rf_preds
            save_predictions(test_df, rf_preds, 'random_forest')
    else:
        print(f"Random Forest model not found at {rf_model_path}")
    
    # LightGBM模型预测
    print("Predicting with LightGBM Model...")
    lgb_model_path = './model/lightgbm_best.pkl'
    if os.path.exists(lgb_model_path):
        lgb_model = load_traditional_model(lgb_model_path)
        if lgb_model is not None:
            lgb_preds = predict_with_traditional_model(lgb_model, X_test)
            predictions['lightgbm'] = lgb_preds
            save_predictions(test_df, lgb_preds, 'lightgbm')
    else:
        print(f"LightGBM model not found at {lgb_model_path}")
    
    # 集成模型预测
    print("Predicting with Ensemble Model...")
    ensemble_model = EnsembleModel(input_dim=input_dim)
    ensemble_model_path = './model/ensemble_best.pth'
    if os.path.exists(ensemble_model_path):
        ensemble_model = load_trained_model(ensemble_model, ensemble_model_path, device)
        if ensemble_model is not None:
            ensemble_preds = predict_with_model(ensemble_model, test_loader, device)
            predictions['ensemble'] = ensemble_preds
            save_predictions(test_df, ensemble_preds, 'ensemble')
    else:
        print(f"Ensemble model not found at {ensemble_model_path}")
    
    # Transformer模型预测
    print("Predicting with Transformer Model...")
    transformer_model = TransformerPredictor(input_dim=input_dim)
    transformer_model_path = './model/transformer_best.pth'
    if os.path.exists(transformer_model_path):
        transformer_model = load_trained_model(transformer_model, transformer_model_path, device)
        if transformer_model is not None:
            transformer_preds = predict_with_model(transformer_model, test_loader, device)
            predictions['transformer'] = transformer_preds
            save_predictions(test_df, transformer_preds, 'transformer')
    else:
        print(f"Transformer model not found at {transformer_model_path}")
    
    # LSTM模型预测
    print("Predicting with LSTM Model...")
    lstm_model = LSTMPredictor(input_size=input_dim)
    lstm_model_path = './model/lstm_best.pth'
    if os.path.exists(lstm_model_path):
        lstm_model = load_trained_model(lstm_model, lstm_model_path, device)
        if lstm_model is not None:
            lstm_preds = predict_with_model(lstm_model, test_loader, device)
            predictions['lstm'] = lstm_preds
            save_predictions(test_df, lstm_preds, 'lstm')
    else:
        print(f"LSTM model not found at {lstm_model_path}")
    
    # Seq2Seq模型预测
    print("Predicting with Seq2Seq Model...")
    seq2seq_model = Seq2SeqPredictor(input_size=input_dim)
    seq2seq_model_path = './model/seq2seq_best.pth'
    if os.path.exists(seq2seq_model_path):
        seq2seq_model = load_trained_model(seq2seq_model, seq2seq_model_path, device)
        if seq2seq_model is not None:
            seq2seq_preds = predict_with_model(seq2seq_model, test_loader, device)
            predictions['seq2seq'] = seq2seq_preds
            save_predictions(test_df, seq2seq_preds, 'seq2seq')
    else:
        print(f"Seq2Seq model not found at {seq2seq_model_path}")
    
    # 移除了计算平均预测结果的部分，改为分别保存每个模型的预测结果
    if predictions:
        print("All model predictions have been saved separately.")
    else:
        print("No predictions were made due to model loading errors")

if __name__ == "__main__":
    main()