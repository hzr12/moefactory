"""
模型超参数随机搜索（不依赖 Optuna，仅用 sklearn + numpy）。

对 4 个传统机器学习模型（RandomForest / LightGBM / XGBoost / CatBoost）做
按“到达日期”分组的留出验证，随机搜索超参数，找到验证集 MSE 最小的一组配置，
并把最优参数写入 model/best_params.json；加 --save 还会用最优配置重训并保存模型。

用法示例：
    python tune_models.py                  # 每个模型搜索 20 组（默认）
    python tune_models.py --trials 40      # 每个模型搜索 40 组
    python tune_models.py --models xgboost lightgbm --trials 30
    python tune_models.py --save           # 搜索完用最优配置重训并保存 .pkl
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from data_preprocess import (
    load_train_data, encode_categorical_features, prepare_train_data, split_by_date,
)

SEED = 3461

# 参数搜索空间：list=离散候选；tuple(lo,hi)=均匀；tuple(lo,hi,'logint'/'logfloat')=对数均匀
SPACES = {
    'random_forest': {
        'n_estimators': (100, 1000, 'logint'),
        'max_depth': (4, 18, 'int'),
        'min_samples_leaf': (1, 4, 'int'),
        'min_samples_split': (2, 10, 'int'),
        'max_features': (0.4, 1.0, 'float'),
    },
    'lightgbm': {
        'num_leaves': (15, 127, 'logint'),
        'learning_rate': (0.005, 0.1, 'logfloat'),
        'n_estimators': (200, 1000, 'logint'),
        'subsample': (0.6, 1.0, 'float'),
        'colsample_bytree': (0.6, 1.0, 'float'),
        'min_child_samples': (5, 40, 'int'),
        'reg_lambda': (0.0, 10.0, 'float'),
    },
    'xgboost': {
        'n_estimators': (200, 1000, 'logint'),
        'max_depth': (3, 10, 'int'),
        'learning_rate': (0.005, 0.1, 'logfloat'),
        'subsample': (0.6, 1.0, 'float'),
        'colsample_bytree': (0.6, 1.0, 'float'),
        'reg_lambda': (0.0, 10.0, 'float'),
        'min_child_weight': (1, 10, 'int'),
        # 固定项：直方图树更快、且不给默认过拟合配置
        'tree_method': 'hist',
    },
    'catboost': {
        'iterations': (200, 1000, 'logint'),
        'depth': (4, 10, 'int'),
        'learning_rate': (0.005, 0.1, 'logfloat'),
        'l2_leaf_reg': (0.0, 10.0, 'float'),
        'subsample': (0.6, 1.0, 'float'),
        'colsample_bylevel': (0.6, 1.0, 'float'),
        'random_strength': (0.0, 10.0, 'float'),
        'verbose': False,
    },
}

CONSTRUCTORS = {
    'random_forest': RandomForestRegressor,
    'lightgbm': LGBMRegressor,
    'xgboost': XGBRegressor,
    'catboost': CatBoostRegressor,
}

FIXED = {
    'random_forest': {'n_jobs': -1, 'random_state': SEED},
    'lightgbm': {'verbose': -1, 'random_state': SEED},
    'xgboost': {'random_state': SEED},
    'catboost': {'random_state': SEED},
}

SAVE_NAME = {
    'random_forest': 'random_forest_best.pkl',
    'lightgbm': 'lightgbm_best.pkl',
    'xgboost': 'xgboost_best.pkl',
    'catboost': 'catboost_best.pkl',
}


def sample_params(space, rng):
    """从搜索空间里采一组参数（去掉固定项由 FIXED 统一补）。"""
    out = {}
    for key, spec in space.items():
        if key in FIXED.get(_model_of(space), {}):
            continue
        if isinstance(spec, list):
            out[key] = rng.choice(spec)
        elif isinstance(spec, tuple) and len(spec) == 3 and spec[2] == 'logint':
            lo, hi, _ = spec
            out[key] = int(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
        elif isinstance(spec, tuple) and len(spec) == 3 and spec[2] == 'logfloat':
            lo, hi, _ = spec
            out[key] = float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
        elif isinstance(spec, tuple) and len(spec) == 3 and spec[2] == 'int':
            lo, hi, _ = spec
            out[key] = rng.randint(lo, hi + 1)
        elif isinstance(spec, tuple) and len(spec) == 3 and spec[2] == 'float':
            lo, hi, _ = spec
            out[key] = rng.uniform(lo, hi)
        elif isinstance(spec, tuple) and len(spec) == 2:
            lo, hi = spec
            out[key] = rng.uniform(lo, hi) if isinstance(lo, float) else rng.randint(lo, hi + 1)
    return out


def _model_of(space):
    for name, sp in SPACES.items():
        if sp is space:
            return name
    return ''


def build_model(name, params):
    return CONSTRUCTORS[name](**{**FIXED.get(name, {}), **params})


def main():
    parser = argparse.ArgumentParser(description='ML 模型超参数随机搜索')
    parser.add_argument('--trials', type=int, default=20, help='每个模型搜索的组数')
    parser.add_argument('--models', nargs='+', default=list(SPACES.keys()),
                        choices=list(SPACES.keys()), help='只搜索指定模型')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='按日期留出的验证集比例')
    parser.add_argument('--save', action='store_true', help='用最优配置重训并保存模型')
    args = parser.parse_args()

    os.makedirs('./model', exist_ok=True)
    rng = random.Random(SEED)

    print('Loading training data...')
    df = load_train_data('./datasets/train')
    df, _ = encode_categorical_features(df)
    X, y = prepare_train_data(df)
    X, y = np.nan_to_num(X), np.nan_to_num(y)
    Xtr, Xva, ytr, yva, s0, s1 = split_by_date(df, X, y, val_ratio=args.val_ratio)
    baseline = mean_squared_error(yva, np.zeros_like(yva))
    print(f'训练 {len(Xtr)} / 验证 {len(Xva)} 行, 特征 {X.shape[1]} 维')
    print(f'基线(全预测0) MSE: {baseline:.4f}  验证日期 {str(s0)[:10]}~{str(s1)[:10]}\n')

    best = {}
    for name in args.models:
        print(f'=== 搜索 {name} ({args.trials} 组) ===')
        best_mse, best_params = float('inf'), None
        for t in range(args.trials):
            params = sample_params(SPACES[name], rng)
            try:
                m = build_model(name, params)
                t0 = time.time()
                m.fit(Xtr, ytr)
                mse = mean_squared_error(yva, m.predict(Xva))
            except Exception as e:
                print(f'   trial {t + 1} 失败: {e}')
                continue
            if mse < best_mse:
                best_mse, best_params = mse, params
            print(f'   trial {t + 1:2d}  MSE {mse:7.4f}  {round(time.time() - t0, 1)}s  {params}')
        best[name] = {'mse': best_mse, 'params': best_params}
        print(f'   >>> {name} 最优 MSE {best_mse:.4f}: {best_params}\n')

    # 写入 model/best_params.json（只覆盖本次搜索到的模型）
    path = './model/best_params.json'
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = {}
    existing.update({k: v['params'] for k, v in best.items()})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f'最优参数已写入 {path}')

    if args.save:
        print('\n用最优配置重训并保存模型...')
        for name in args.models:
            m = build_model(name, best[name]['params'])
            m.fit(Xtr, ytr)
            with open(os.path.join('./model', SAVE_NAME[name]), 'wb') as f:
                import pickle
                pickle.dump(m, f)
            print(f'   saved {SAVE_NAME[name]} (MSE {best[name]["mse"]:.4f})')


if __name__ == '__main__':
    main()
