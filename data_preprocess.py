import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

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
    feature_columns = ['出发小时', '出发分钟', '出发月份', '出发日', '出发星期', '车站编码']
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
    feature_columns = ['出发小时', '出发分钟', '出发月份', '出发日', '出发星期', '车站编码']
    X = test_df[feature_columns].values
    
    return X