import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error
from data_preprocess import TrainDelayDataset
from torch.utils.data import DataLoader
from models import TransformerPredictor, LSTMPredictor, Seq2SeqPredictor, TFT

"""
集成模型模块
负责集成多种模型的预测结果，使用动态权重分配策略
"""

class IntegratedEnsembleModel:
    """
    集成所有模型的综合集成模型类
    包括传统机器学习模型和深度学习模型，使用动态权重分配策略
    """
    
    def __init__(self, input_dim=6):
        """
        初始化集成模型
        
        Args:
            input_dim (int): 输入特征维度
        """
        # 深度学习模型
        self.transformer_model = TransformerPredictor(input_dim=input_dim)
        self.lstm_model = LSTMPredictor(input_size=input_dim)
        self.seq2seq_model = Seq2SeqPredictor(input_size=input_dim)
        self.tft_model = TFT(input_dim=input_dim)
        self.dl_ensemble_model = None  # 深度学习集成模型
        
        # 传统机器学习模型
        self.rf_model = None  # 随机森林模型
        self.lgb_model = None  # LightGBM模型
        self.xgb_model = None  # XGBoost模型
        self.cat_model = None  # CatBoost模型
        
        # 模型权重
        self.weights = {
            'transformer': 1.0,
            'lstm': 1.0,
            'seq2seq': 1.0,
            'tft': 1.0,
            'rf': 1.0,
            'lgb': 1.0,
            'xgb': 1.0,
            'cat': 1.0
        }
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_models(self, model_dir='./model'):
        """
        加载所有训练好的模型
        
        Args:
            model_dir (str): 模型存储目录
        """
        # 加载深度学习模型
        try:
            self.transformer_model.load_state_dict(
                torch.load(f'{model_dir}/transformer_best.pth', map_location=self.device)
            )
            self.transformer_model.to(self.device)
            self.transformer_model.eval()
            print("Successfully loaded Transformer model")
        except Exception as e:
            print(f"Failed to load Transformer model: {e}")
            
        try:
            self.lstm_model.load_state_dict(
                torch.load(f'{model_dir}/lstm_best.pth', map_location=self.device)
            )
            self.lstm_model.to(self.device)
            self.lstm_model.eval()
            print("Successfully loaded LSTM model")
        except Exception as e:
            print(f"Failed to load LSTM model: {e}")
            
        try:
            self.seq2seq_model.load_state_dict(
                torch.load(f'{model_dir}/seq2seq_best.pth', map_location=self.device)
            )
            self.seq2seq_model.to(self.device)
            self.seq2seq_model.eval()
            print("Successfully loaded Seq2Seq model")
        except Exception as e:
            print(f"Failed to load Seq2Seq model: {e}")
            
        try:
            self.tft_model.load_state_dict(
                torch.load(f'{model_dir}/tft_best.pth', map_location=self.device)
            )
            self.tft_model.to(self.device)
            self.tft_model.eval()
            print("Successfully loaded TFT model")
        except Exception as e:
            print(f"Failed to load TFT model: {e}")
            
        # 加载传统机器学习模型
        try:
            with open(f'{model_dir}/random_forest_best.pkl', 'rb') as f:
                self.rf_model = pickle.load(f)
            print("Successfully loaded Random Forest model")
        except Exception as e:
            print(f"Failed to load Random Forest model: {e}")
            
        try:
            with open(f'{model_dir}/lightgbm_best.pkl', 'rb') as f:
                self.lgb_model = pickle.load(f)
            print("Successfully loaded LightGBM model")
        except Exception as e:
            print(f"Failed to load LightGBM model: {e}")
            
        try:
            with open(f'{model_dir}/xgboost_best.pkl', 'rb') as f:
                self.xgb_model = pickle.load(f)
            print("Successfully loaded XGBoost model")
        except Exception as e:
            print(f"Failed to load XGBoost model: {e}")
            
        try:
            with open(f'{model_dir}/catboost_best.pkl', 'rb') as f:
                self.cat_model = pickle.load(f)
            print("Successfully loaded CatBoost model")
        except Exception as e:
            print(f"Failed to load CatBoost model: {e}")
    
    def predict_dl_models(self, X):
        """
        使用深度学习模型进行预测
        
        Args:
            X (numpy.ndarray): 输入特征
            
        Returns:
            dict: 各深度学习模型的预测结果
        """
        # 创建数据加载器
        dataset = TrainDelayDataset(X, np.zeros(len(X)))  # 只需要输入，不需要标签
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 存储预测结果
        transformer_preds = []
        lstm_preds = []
        seq2seq_preds = []
        tft_preds = []
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                
                # Transformer预测
                if hasattr(self, 'transformer_model') and self.transformer_model is not None:
                    try:
                        pred = self.transformer_model(inputs)
                        transformer_preds.extend(pred.cpu().numpy().flatten())
                    except Exception as e:
                        print(f"Error in Transformer prediction: {e}")
                        transformer_preds.extend([0] * inputs.size(0))
                
                # LSTM预测
                if hasattr(self, 'lstm_model') and self.lstm_model is not None:
                    try:
                        pred = self.lstm_model(inputs)
                        lstm_preds.extend(pred.cpu().numpy().flatten())
                    except Exception as e:
                        print(f"Error in LSTM prediction: {e}")
                        lstm_preds.extend([0] * inputs.size(0))
                
                # Seq2Seq预测
                if hasattr(self, 'seq2seq_model') and self.seq2seq_model is not None:
                    try:
                        pred = self.seq2seq_model(inputs)
                        seq2seq_preds.extend(pred.cpu().numpy().flatten())
                    except Exception as e:
                        print(f"Error in Seq2Seq prediction: {e}")
                        seq2seq_preds.extend([0] * inputs.size(0))
                
                # TFT预测
                if hasattr(self, 'tft_model') and self.tft_model is not None:
                    try:
                        pred = self.tft_model(inputs)
                        tft_preds.extend(pred.cpu().numpy().flatten())
                    except Exception as e:
                        print(f"Error in TFT prediction: {e}")
                        tft_preds.extend([0] * inputs.size(0))
        
        return {
            'transformer': np.array(transformer_preds),
            'lstm': np.array(lstm_preds),
            'seq2seq': np.array(seq2seq_preds),
            'tft': np.array(tft_preds)
        }
    
    def predict_ml_models(self, X):
        """
        使用传统机器学习模型进行预测
        
        Args:
            X (numpy.ndarray): 输入特征
            
        Returns:
            dict: 各传统机器学习模型的预测结果
        """
        predictions = {}
        
        # 随机森林预测
        if self.rf_model is not None:
            try:
                predictions['rf'] = self.rf_model.predict(X)
            except Exception as e:
                print(f"Error in Random Forest prediction: {e}")
                predictions['rf'] = np.zeros(len(X))
        
        # LightGBM预测
        if self.lgb_model is not None:
            try:
                predictions['lgb'] = self.lgb_model.predict(X)
            except Exception as e:
                print(f"Error in LightGBM prediction: {e}")
                predictions['lgb'] = np.zeros(len(X))
        
        # XGBoost预测
        if self.xgb_model is not None:
            try:
                predictions['xgb'] = self.xgb_model.predict(X)
            except Exception as e:
                print(f"Error in XGBoost prediction: {e}")
                predictions['xgb'] = np.zeros(len(X))
        
        # CatBoost预测
        if self.cat_model is not None:
            try:
                predictions['cat'] = self.cat_model.predict(X)
            except Exception as e:
                print(f"Error in CatBoost prediction: {e}")
                predictions['cat'] = np.zeros(len(X))
        
        return predictions
    
    def calculate_weights(self, X_val, y_val):
        """
        根据验证集表现动态计算模型权重
        
        Args:
            X_val (numpy.ndarray): 验证集特征
            y_val (numpy.ndarray): 验证集目标值
            
        Returns:
            dict: 各模型的权重
        """
        print("Calculating dynamic weights based on validation performance...")
        
        # 获取深度学习模型预测
        dl_predictions = self.predict_dl_models(X_val)
        
        # 获取传统机器学习模型预测
        ml_predictions = self.predict_ml_models(X_val)
        
        # 合并所有预测
        all_predictions = {**dl_predictions, **ml_predictions}
        
        # 计算每个模型的MSE（越小越好）
        model_mse = {}
        for model_name, preds in all_predictions.items():
            if len(preds) == len(y_val):
                mse = mean_squared_error(y_val, preds)
                model_mse[model_name] = mse
                print(f"{model_name} MSE: {mse:.4f}")
            else:
                print(f"Skipping {model_name} due to prediction length mismatch")
        
        # 计算权重（MSE的倒数作为基础权重）
        weights = {}
        total_weight = 0
        
        for model_name, mse in model_mse.items():
            # 使用MSE的倒数作为基础权重，加一个小常数防止除零
            weight = 1.0 / (mse + 1e-8)
            weights[model_name] = weight
            total_weight += weight
        
        # 归一化权重，使总和为1
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] = weights[model_name] / total_weight
        
        self.weights = weights
        print(f"Calculated weights: {self.weights}")
        return self.weights
    
    def predict(self, X):
        """
        使用集成模型进行预测
        
        Args:
            X (numpy.ndarray): 输入特征
            
        Returns:
            numpy.ndarray: 集成预测结果
        """
        # 获取深度学习模型预测
        dl_predictions = self.predict_dl_models(X)
        
        # 获取传统机器学习模型预测
        ml_predictions = self.predict_ml_models(X)
        
        # 合并所有预测
        all_predictions = {**dl_predictions, **ml_predictions}
        
        # 使用权重进行加权平均
        weighted_predictions = np.zeros(len(X))
        total_weight = 0
        
        for model_name, preds in all_predictions.items():
            if model_name in self.weights and len(preds) == len(X):
                weight = self.weights[model_name]
                weighted_predictions += weight * preds
                total_weight += weight
                print(f"Applied {model_name} with weight {weight:.4f}")
            else:
                print(f"Skipping {model_name} in final prediction")
        
        # 归一化（如果总权重不为1）
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            weighted_predictions = weighted_predictions / total_weight
            print(f"Normalized predictions by total weight: {total_weight:.4f}")
        
        return weighted_predictions
    
    def save_model_info(self, filepath):
        """
        保存模型信息（权重等）
        
        Args:
            filepath (str): 保存文件路径
        """
        model_info = {
            'weights': self.weights
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_info, f)
            print(f"Model info saved to {filepath}")
        except Exception as e:
            print(f"Failed to save model info: {e}")
    
    def load_model_info(self, filepath):
        """
        加载模型信息（权重等）
        
        Args:
            filepath (str): 加载文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            with open(filepath, 'rb') as f:
                model_info = pickle.load(f)
            
            self.weights = model_info.get('weights', self.weights)
            print(f"Model info loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Failed to load model info: {e}")
            return False
