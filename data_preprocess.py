import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

"""
数据预处理模块
负责数据加载、特征提取和预处理
"""

class TrainDelayDataset(Dataset):
    """
    列车延误数据集类
    继承自PyTorch的Dataset类，用于加载和处理训练数据
    """
    def __init__(self, data, targets=None):
        """
        初始化数据集
        
        Args:
            data (numpy.ndarray): 输入特征数据
            targets (numpy.ndarray, optional): 目标值数据，默认为None
        """
        self.data = data
        self.targets = targets
    
    def __len__(self):
        """
        获取数据集长度
        
        Returns:
            int: 数据集样本数量
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple or torch.Tensor: 如果有目标值，返回(特征, 目标值)元组；否则返回特征张量
        """
        if self.targets is not None:
            return torch.FloatTensor(self.data[idx]), torch.FloatTensor([self.targets[idx]])
        else:
            return torch.FloatTensor(self.data[idx])

def load_train_data(train_dir):
    """
    加载训练数据
    
    Args:
        train_dir (str): 训练数据目录路径
        
    Returns:
        pandas.DataFrame: 合并后的训练数据
    """
    all_data = []
    
    # 遍历目录中的所有CSV文件
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
    
    Args:
        test_file (str): 测试数据文件路径
        
    Returns:
        pandas.DataFrame: 测试数据
    """
    df = pd.read_csv(test_file)
    return df

def extract_time_features(df, is_train=True):
    """
    提取时间特征
    
    Args:
        df (pandas.DataFrame): 输入数据
        is_train (bool): 是否为训练数据
        
    Returns:
        pandas.DataFrame: 提取特征后的数据
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

    # 车站与时间段的交互特征
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
    
    Args:
        train_df (pandas.DataFrame): 训练数据
        test_df (pandas.DataFrame, optional): 测试数据，默认为None
        
    Returns:
        tuple: 如果提供了测试数据，返回(train_df, test_df, le_station)；否则返回(train_df, le_station)
    """
    # 处理车站名
    le_station = LabelEncoder()
    
    if test_df is not None:
        # 合并训练和测试数据进行编码，确保所有可能的类别都被覆盖
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
    
    Args:
        train_df (pandas.DataFrame): 原始训练数据
        
    Returns:
        tuple: (特征矩阵X, 目标值y)
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
    
    Args:
        test_df (pandas.DataFrame): 原始测试数据
        
    Returns:
        numpy.ndarray: 特征矩阵X
    """
    # 提取时间特征
    test_df = extract_time_features(test_df, is_train=False)
    
    # 选择特征列
    feature_columns = ['出发小时', '出发分钟', '出发月份', '出发日', '出发星期', '车站编码', '车站_小时交互', '车站_星期交互', '小时_星期交互', '到达小时', '到达分钟', '到达时间_小时', '到达时间_分钟', '到达时间_小时_分钟']
    X = test_df[feature_columns].values
    
    return X
