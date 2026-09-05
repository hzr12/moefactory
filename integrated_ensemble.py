import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error
from data_preprocess import TrainDelayDataset, flatten_sequence_predictions, sequence_row_order
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
    
    def predict_dl_models(self, X_seq, lengths, row_pos, n_rows):
        """
        使用深度学习模型在“行程序列”上预测，并还原成按原始行排列的结果

        Args:
            X_seq: (n_journeys, max_len, n_features) 序列特征
            lengths: 每个行程的真实站点数
            row_pos: 每个时间步对应的原始行位置
            n_rows: 原始数据行数

        Returns:
            dict: 各深度学习模型的预测结果（按原始行顺序）
        """
        inputs = torch.FloatTensor(X_seq).to(self.device)
        lens = torch.LongTensor(lengths).to(self.device)
        predictions = {}

        candidates = [('transformer', self.transformer_model), ('lstm', self.lstm_model),
                      ('seq2seq', self.seq2seq_model), ('tft', self.tft_model)]
        with torch.no_grad():
            for name, model in candidates:
                if model is None:
                    continue
                try:
                    pred = model(inputs, lens).cpu().numpy()
                    # 摊平后按行号取回与 X_flat 对应的子集（行序一致）
                    pred_by_row = flatten_sequence_predictions(pred, {'row_pos': row_pos})
                    predictions[name] = pred_by_row[sequence_row_order({'row_pos': row_pos})]
                except Exception as e:
                    print(f"Error in {name} prediction: {e}")
                    predictions[name] = np.zeros(n_rows)

        return predictions

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
    
    def calculate_weights(self, X_val_flat, y_val, X_val_seq=None, lengths=None, row_pos=None):
        """
        根据验证集表现动态计算模型权重
        """
        print("Calculating dynamic weights based on validation performance...")

        # 获取深度学习模型预测（行程序列）
        dl_predictions = {}
        if X_val_seq is not None and row_pos is not None:
            dl_predictions = self.predict_dl_models(X_val_seq, lengths, row_pos, len(X_val_flat))

        # 获取传统机器学习模型预测
        ml_predictions = self.predict_ml_models(X_val_flat)

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

    def predict(self, X_flat, X_seq=None, lengths=None, row_pos=None):
        """
        使用集成模型进行预测

        Args:
            X_flat: 扁平特征，供传统机器学习模型使用
            X_seq / lengths / row_pos: 行程序列，供深度学习模型使用（可缺省）

        Returns:
            numpy.ndarray: 集成预测结果
        """
        n_rows = len(X_flat)

        # 获取深度学习模型预测
        dl_predictions = {}
        if X_seq is not None and row_pos is not None:
            dl_predictions = self.predict_dl_models(X_seq, lengths, row_pos, n_rows)

        # 获取传统机器学习模型预测
        ml_predictions = self.predict_ml_models(X_flat)

        # 合并所有预测
        all_predictions = {**dl_predictions, **ml_predictions}

        # 使用权重进行加权平均
        weighted_predictions = np.zeros(n_rows)
        total_weight = 0

        for model_name, preds in all_predictions.items():
            if model_name in self.weights and len(preds) == n_rows:
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
