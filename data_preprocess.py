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

# 数值天气特征（20 列新数据格式）。“天气情况”为文本冗余列、“距离”在 G339 历史数据中缺失，均不作为特征。
WEATHER_FEATURES = ['当日最高温', '当日最低温', '当日降水量', '当日降雨量', '当日降雪量',
                    '降水小时数', '最大风速', '最大阵风', '平均云量', '平均相对湿度', '天气代码']

# 基础时间/车站特征
BASE_FEATURES = ['出发小时', '出发分钟', '出发月份', '出发日', '出发星期', '车站编码',
                 '车站_小时交互', '车站_星期交互', '小时_星期交互', '到达小时', '到达分钟',
                 '到达时间_小时', '到达时间_分钟', '到达时间_小时_分钟']


# 行程特征（按“同一车次 + 同一到达日期”的行程分组、沿停站顺序计算）
JOURNEY_FEATURES = ['站序', '行程站点数', '站序占比', '是否始发站', '是否终到站',
                    '前一站延误分钟', '累计延误分钟', '停站时长', '站间运行时间']


def _feature_columns(df):
    """最终特征列 = 基础特征 + 车次编码 + 行程特征 + 天气特征。"""
    return list(BASE_FEATURES) + ['车次编码'] + list(JOURNEY_FEATURES) + list(WEATHER_FEATURES)


def _safe_transform(le, values):
    """逐值转换，未知标签记为 0（与原整列 try/except 行为一致但更精细）。"""
    mapping = {label: idx for idx, label in enumerate(le.classes_)}
    return [mapping.get(v, 0) for v in values]


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
    编码分类特征（车站 + 车次）

    Args:
        train_df (pandas.DataFrame): 训练数据
        test_df (pandas.DataFrame, optional): 测试数据，默认为None

    Returns:
        tuple: 如果提供了测试数据，返回(train_df, test_df, le_station)；否则返回(train_df, le_station)
    """
    le_station = LabelEncoder()
    le_train = LabelEncoder()

    train_df['车次ID'] = train_df['车次ID'].astype(str)

    if test_df is not None:
        test_df['车次ID'] = test_df['车次ID'].astype(str)
        # 合并训练和测试数据进行编码，确保所有可能的类别都被覆盖
        le_station.fit(pd.concat([train_df['车站名'], test_df['车站名']], ignore_index=True))
        le_train.fit(pd.concat([train_df['车次ID'], test_df['车次ID']], ignore_index=True))

        train_df['车站编码'] = _safe_transform(le_station, train_df['车站名'])
        test_df['车站编码'] = _safe_transform(le_station, test_df['车站名'])
        train_df['车次编码'] = _safe_transform(le_train, train_df['车次ID'])
        test_df['车次编码'] = _safe_transform(le_train, test_df['车次ID'])

        # 编码器挂在 attrs 上，训练脚本可直接取用保存（保持原有返回签名不变）
        train_df.attrs['label_encoders'] = {'station': le_station, 'train': le_train}
        return train_df, test_df, le_station
    else:
        # 仅处理训练数据
        le_station.fit(train_df['车站名'])
        le_train.fit(train_df['车次ID'])
        train_df['车站编码'] = _safe_transform(le_station, train_df['车站名'])
        train_df['车次编码'] = _safe_transform(le_train, train_df['车次ID'])
        train_df.attrs['label_encoders'] = {'station': le_station, 'train': le_train}
        return train_df, le_station

def _time_to_minutes(series):
    """把 'HH:MM' 形式的时间列转成当日分钟数（无法解析返回 NaN）。"""
    parts = series.astype(str).str.strip().str.split(':', expand=True)
    hours = pd.to_numeric(parts[0], errors='coerce')
    minutes = pd.to_numeric(parts[1], errors='coerce')
    return hours * 60 + minutes


def add_journey_features(df):
    """按行程（车次 + 到达日期）构造沿停站顺序的特征。

    每个 CSV 是一趟车按停站顺序排列的行，因此行号即站序。新增：
      站序 / 行程站点数 / 站序占比 / 是否始发站 / 是否终到站
      前一站延误分钟、累计延误分钟、停站时长、站间运行时间

    注意：前一站延误与累计延误取自同一行程内其它站的“延误分钟”，仅在“已知前序站实际延误”
    的场景可用（实时滚动预测、事后分析）。若要做纯发车前预测，应剔除这两列。
    """
    df = df.copy()
    day = pd.to_datetime(df['到达日期'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('unknown')
    df['_journey_id'] = df['车次ID'].astype(str) + '_' + day

    df['站序'] = df.groupby('_journey_id', sort=False).cumcount() + 1
    df['行程站点数'] = df.groupby('_journey_id', sort=False)['站序'].transform('max')
    df['是否始发站'] = (df['站序'] == 1).astype(int)
    df['是否终到站'] = (df['站序'] == df['行程站点数']).astype(int)
    df['站序占比'] = df['站序'] / df['行程站点数'].replace(0, np.nan)

    delay = pd.to_numeric(df['延误分钟'], errors='coerce').fillna(0.0)
    df['前一站延误分钟'] = delay.groupby(df['_journey_id'], sort=False).shift(1).fillna(0.0)
    # 必须减掉当前站自身，否则等于把标签直接作为特征（目标泄漏）
    df['累计延误分钟'] = delay.groupby(df['_journey_id'], sort=False).cumsum() - delay

    arrive = _time_to_minutes(df['到达时间'])
    depart = _time_to_minutes(df['出发时间'])
    dwell = depart - arrive
    df['停站时长'] = dwell.where(dwell >= 0, dwell + 1440).fillna(0.0)  # 跨零点修正

    prev_depart = depart.groupby(df['_journey_id'], sort=False).shift(1)
    run = arrive - prev_depart
    df['站间运行时间'] = run.where(run >= 0, run + 1440).fillna(0.0)
    return df


def _build_feature_frame(df):
    """统一的特征构造流程：行程特征 -> 时间特征 -> 数值化。"""
    df = add_journey_features(df)
    df = extract_time_features(df)
    df = _coerce_weather_numeric(df)
    for c in JOURNEY_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _coerce_weather_numeric(df):
    """把天气特征统一为数值（空串/缺失→NaN，由调用方 nan_to_num 处理）；旧数据缺列时补 NaN。"""
    for c in WEATHER_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    if '车次编码' not in df.columns:
        df['车次编码'] = 0
    return df


def split_by_date(df, X, y, val_ratio=0.1, date_col='到达日期'):
    """按日期分组留出验证集：取日期排序后最后 val_ratio 比例的日期作为验证集。

    同一天的所有行要么全在训练、要么全在验证。随机行划分会把同一天的行分到两侧，
    使“按天常量”的天气特征沦为一个日期指纹，评估结果虚高，也不符合“预测一个全新日期”的真实场景。
    """
    dates = pd.to_datetime(df[date_col], errors='coerce')
    unique_dates = pd.DatetimeIndex(sorted(dates.dropna().unique()))
    n_val_dates = max(1, int(round(len(unique_dates) * val_ratio)))
    val_dates = unique_dates[-n_val_dates:]
    is_val = dates.isin(val_dates).values

    return X[~is_val], X[is_val], y[~is_val], y[is_val], val_dates[0], val_dates[-1]


def prepare_train_data(train_df):
    """
    准备训练数据
    
    Args:
        train_df (pandas.DataFrame): 原始训练数据
        
    Returns:
        tuple: (特征矩阵X, 目标值y)
    """
    # 构造全部特征（行程特征 + 时间特征 + 天气数值化）
    train_df = _build_feature_frame(train_df)
    # 选择特征列：基础特征 + 车次编码 + 行程特征 + 天气特征
    X = train_df[_feature_columns(train_df)].values.astype(float)
    y = pd.to_numeric(train_df['延误分钟'], errors='coerce').fillna(0).values  # 处理目标值中的NaN
    
    return X, y

def prepare_test_data(test_df):
    """
    准备测试数据
    
    Args:
        test_df (pandas.DataFrame): 原始测试数据
        
    Returns:
        numpy.ndarray: 特征矩阵X
    """
    # 构造全部特征（行程特征 + 时间特征 + 天气数值化）
    test_df = _build_feature_frame(test_df)

    # 选择特征列：基础特征 + 车次编码 + 行程特征 + 天气特征
    X = test_df[_feature_columns(test_df)].values.astype(float)
    
    return X


def prepare_sequence_data(df, max_len=None):
    """把按行排列的停站记录重组成“行程序列”张量，供 LSTM/Transformer/TFT 使用。

    一趟车（车次ID + 到达日期）的沿途各站构成一个序列，站序即时间步，
    这样循环/注意力机制才真正作用于时序结构（此前是 seq_len=1，等于空转）。

    返回 dict:
      X: (n_journeys, max_len, n_features) float32
      y: (n_journeys, max_len) float32
      lengths: (n_journeys,) 每个行程的真实站点数
      row_pos: (n_journeys, max_len) 每个时间步对应的原始行位置（padding 位为 -1）
      dates: (n_journeys,) 每个行程的日期，便于按日期分组划分
    """
    fdf = _build_feature_frame(df).reset_index(drop=True)
    cols = _feature_columns(fdf)
    values = np.nan_to_num(fdf[cols].values.astype(float)).astype(np.float32)
    targets = pd.to_numeric(fdf['延误分钟'], errors='coerce').fillna(0).values.astype(np.float32)

    groups = [np.asarray(idx, dtype=int)
              for _, idx in fdf.groupby('_journey_id', sort=False).groups.items()]
    lengths = np.array([len(g) for g in groups], dtype=np.int64)
    max_steps = int(lengths.max()) if max_len is None else int(max_len)
    n, d = len(groups), values.shape[1]

    X = np.zeros((n, max_steps, d), dtype=np.float32)
    y = np.zeros((n, max_steps), dtype=np.float32)
    row_pos = np.full((n, max_steps), -1, dtype=np.int64)
    dates = []
    for i, g in enumerate(groups):
        k = min(len(g), max_steps)
        X[i, :k] = values[g[:k]]
        y[i, :k] = targets[g[:k]]
        row_pos[i, :k] = g[:k]
        dates.append(fdf['到达日期'].iloc[g[0]])

    return {'X': X, 'y': y, 'lengths': lengths, 'row_pos': row_pos,
            'dates': pd.to_datetime(pd.Series(dates), errors='coerce').values}


def split_sequence_by_date(seq, val_ratio=0.1):
    """按日期分组切分序列数据：取日期排序后最后 val_ratio 比例的日期作为验证集。"""
    dates = pd.to_datetime(pd.Series(seq['dates']), errors='coerce')
    unique_dates = pd.DatetimeIndex(dates.dropna().unique()).sort_values()
    n_val_dates = max(1, int(round(len(unique_dates) * val_ratio)))
    val_dates = unique_dates[-n_val_dates:]
    is_val = dates.isin(val_dates).values

    def _take(mask):
        return {'X': seq['X'][mask], 'y': seq['y'][mask], 'lengths': seq['lengths'][mask],
                'row_pos': seq['row_pos'][mask], 'dates': seq['dates'][mask]}

    return _take(~is_val), _take(is_val), val_dates[0], val_dates[-1]


def flatten_sequence_predictions(preds, seq):
    """把 (n_journeys, max_len) 的序列预测还原成按原始行顺序排列的一维数组。"""
    preds = np.asarray(preds).reshape(-1)
    row_pos = np.asarray(seq['row_pos']).reshape(-1)
    mask = row_pos >= 0
    out = np.zeros(int(mask.sum()), dtype=float)
    out[row_pos[mask]] = preds[mask]
    return out
