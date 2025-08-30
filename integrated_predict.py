import torch
import pandas as pd
import numpy as np
import os
import pickle
import argparse
import glob

from data_preprocess import load_test_data, encode_categorical_features, prepare_test_data, load_train_data
from integrated_ensemble_learning import IntegratedEnsembleModel

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

def predict_with_integrated_model(model, X_test_dl, X_test_traditional, device):
    """
    使用集成模型进行预测
    """
    # 转换为PyTorch张量
    X_test_tensor = torch.FloatTensor(X_test_dl).to(device)
    
    # 使用集成模型进行预测
    predictions, all_predictions, weights = model.predict_with_all_models(X_test_tensor, X_test_traditional)
    
    print(f"Model weights: {weights}")
    
    return predictions

def save_predictions(test_df, predictions, model_name, test_file_name):
    """
    保存预测结果
    """
    # 四舍五入为整数
    model_predictions = np.round(predictions).astype(int)
    
    # 创建结果DataFrame
    result_df = test_df[['车次ID', '车站名', '出发日期', '出发时间']].copy()
    result_df['预测延误分钟'] = model_predictions
    
    # 保存预测结果
    os.makedirs('./predictions', exist_ok=True)
    filename = f'./predictions/prediction_results_{model_name}_{test_file_name}.csv'
    result_df.to_csv(filename, index=False)
    print(f"{model_name} predictions for {test_file_name} saved to {filename}")

def process_single_file(test_file, train_df, device):
    """
    处理单个测试文件
    """
    print(f"Loading test data from {test_file}...")
    test_df = load_test_data(test_file)
    print(f"Loaded {len(test_df)} test samples")
    
    # 获取测试文件名（不含路径和扩展名）
    test_file_name = os.path.splitext(os.path.basename(test_file))[0]
    
    # 编码分类特征
    train_df_encoded, test_df_encoded, label_encoder = encode_categorical_features(train_df.copy(), test_df.copy())
    
    # 准备测试数据
    X_test = prepare_test_data(test_df_encoded)
    
    # 处理NaN值
    X_test = np.nan_to_num(X_test, nan=0.0).astype(np.float32)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"NaN in test data: {np.isnan(X_test).sum()}")
    
    # 输入维度
    input_dim = X_test.shape[1]
    
    # 加载集成模型并进行预测
    print("Predicting with Integrated Ensemble Model...")
    integrated_model = IntegratedEnsembleModel(input_dim=input_dim)
    integrated_model_path = './model/integrated_ensemble_best.pth'
    
    if os.path.exists(integrated_model_path):
        integrated_model = load_trained_model(integrated_model, integrated_model_path, device)
        if integrated_model is not None:
            # 加载传统机器学习模型
            integrated_model.load_traditional_models()
            
            # 进行预测
            integrated_preds = predict_with_integrated_model(integrated_model, X_test, X_test, device)
            save_predictions(test_df, integrated_preds, 'integrated_ensemble', test_file_name)
            return integrated_preds
        else:
            print("Failed to load integrated ensemble model")
    else:
        print(f"Integrated ensemble model not found at {integrated_model_path}")
    
    return None

def main():
    # 创建必要的目录
    os.makedirs('./predictions', exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 查找所有测试文件
    test_files = glob.glob('./datasets/test/*.csv')
    if not test_files:
        print("No test files found in ./datasets/test/")
        return
    
    print(f"Found {len(test_files)} test files")
    
    # 加载训练数据以获取标签编码器
    train_df = load_train_data('./datasets/train')
    
    # 处理每个测试文件
    all_predictions = {}
    for test_file in test_files:
        test_file_name = os.path.splitext(os.path.basename(test_file))[0]
        print(f"\nProcessing test file: {test_file_name}")
        predictions = process_single_file(test_file, train_df, device)
        all_predictions[test_file_name] = predictions
    
    if all_predictions:
        print("\nAll test files have been processed and predictions saved.")
    else:
        print("\nNo predictions were made due to model loading errors")

if __name__ == "__main__":
    main()