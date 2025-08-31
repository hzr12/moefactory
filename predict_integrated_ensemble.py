import torch
import pandas as pd
import numpy as np
import os
import pickle
import glob
from data_preprocess import load_test_data, encode_categorical_features, prepare_test_data, load_train_data
from integrated_ensemble import IntegratedEnsembleModel

def load_model_info(model_info_path):
    """
    加载模型信息
    """
    try:
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
        print(f"Successfully loaded model info from {model_info_path}")
        return model_info
    except Exception as e:
        print(f"Error loading model info from {model_info_path}: {e}")
        return None

def save_predictions(test_df, predictions, test_file_name):
    """
    保存集成模型的预测结果
    """
    # 四舍五入为整数
    model_predictions = np.round(predictions).astype(int)
    
    # 创建结果DataFrame
    result_df = test_df[['车次ID', '车站名', '出发日期', '出发时间']].copy()
    result_df['预测延误分钟'] = model_predictions
    
    # 保存预测结果
    filename = f'./predictions/prediction_results_integrated_ensemble_{test_file_name}.csv'
    result_df.to_csv(filename, index=False)
    print(f"Integrated ensemble predictions for {test_file_name} saved to {filename}")

def process_single_file(test_file, train_df, model_info=None):
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
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"NaN in test data: {np.isnan(X_test).sum()}")
    
    # 输入维度
    input_dim = X_test.shape[1]
    
    # 创建集成模型
    ensemble_model = IntegratedEnsembleModel(input_dim=input_dim)
    
    # 加载所有模型
    ensemble_model.load_models('./model')
    
    # 如果有模型信息（包含权重），则加载
    if model_info is not None:
        ensemble_model.weights = model_info.get('weights', ensemble_model.weights)
        print(f"Loaded dynamic weights: {ensemble_model.weights}")
    else:
        # 尝试从文件加载模型信息
        model_info_path = './model/integrated_ensemble_info.pkl'
        if os.path.exists(model_info_path):
            model_info = load_model_info(model_info_path)
            if model_info is not None:
                ensemble_model.weights = model_info.get('weights', ensemble_model.weights)
                print(f"Loaded dynamic weights from file: {ensemble_model.weights}")
    
    # 进行预测
    print("Predicting with Integrated Ensemble Model...")
    predictions = ensemble_model.predict(X_test)
    
    # 保存预测结果
    save_predictions(test_df, predictions, test_file_name)
    
    return predictions

def main():
    """
    主函数
    """
    # 创建必要的目录
    os.makedirs('./predictions', exist_ok=True)
    
    # 查找所有测试文件
    test_files = glob.glob('./datasets/test/*.csv')
    if not test_files:
        print("No test files found in ./datasets/test/")
        return
    
    print(f"Found {len(test_files)} test files")
    
    # 加载训练数据以获取标签编码器
    train_df = load_train_data('./datasets/train')
    
    # 尝试加载模型信息（包含权重）
    model_info = None
    model_info_path = './model/integrated_ensemble_info.pkl'
    if os.path.exists(model_info_path):
        model_info = load_model_info(model_info_path)
    
    # 处理每个测试文件
    for test_file in test_files:
        test_file_name = os.path.splitext(os.path.basename(test_file))[0]
        print(f"\nProcessing test file: {test_file_name}")
        predictions = process_single_file(test_file, train_df, model_info)
    
    print("\nAll test files have been processed and predictions saved.")

if __name__ == "__main__":
    main()