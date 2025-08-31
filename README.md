# 列车延误预测系统

基于PyTorch和多种机器学习算法的列车延误预测系统

## 项目概述

本项目通过分析列车运行历史数据，使用深度学习和传统机器学习模型预测列车延误情况。系统支持多种模型训练和预测，并提供模型集成方案以提高预测准确性。

## 功能特性

- **多模型支持**：支持多种深度学习和传统机器学习模型
- **数据预处理**：完整的数据预处理和特征工程流程
- **模型训练**：支持多种模型的训练和验证
- **预测功能**：基于训练模型对测试数据进行预测
- **模型集成**：提供模型集成方案，提高预测准确性
- **结果输出**：将预测结果保存为CSV文件

## 技术架构

### 深度学习模型
- Transformer模型
- LSTM模型
- Seq2Seq模型
- 集成模型（综合以上三种模型）

### 传统机器学习模型
- 随机森林 (Random Forest)
- LightGBM
- XGBoost
- CatBoost

## 安装依赖

```bash
pip install -r requirements.txt
```

项目依赖：
- torch>=1.9.0
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- lightgbm>=3.3.0
- xgboost>=1.5.0
- catboost>=1.0.0
- tqdm>=4.62.0

## 项目结构

```
.
├── data_preprocess.py             # 数据预处理模块
├── models.py                      # 深度学习模型定义
├── train.py                       # 模型训练主程序
├── predict.py                     # 模型预测主程序
├── integrated_ensemble.py          # 集成模型
├── train_integrated_ensemble.py   # 集成模型训练
├── predict_integrated_ensemble.py # 集成模型预测
├── incremental_train.py           # 增量训练
├── datasets/                      # 数据集目录
│   ├── train/                     # 训练数据
│   └── test/                      # 测试数据
├── model/                         # 模型保存目录
└── predictions/                   # 预测结果保存目录
```

## 数据格式

### 训练数据格式
训练数据应包含以下列：
- 车次ID：列车唯一标识符
- 车站名：车站名称
- 出发日期：列车出发日期
- 出发时间：列车出发时间 (HH:MM格式)
- 到达时间：列车到达时间 (HH:MM格式)
- 延误分钟：实际延误分钟数（目标变量）

### 测试数据格式
测试数据应包含以下列：
- 车次ID：列车唯一标识符
- 车站名：车站名称
- 出发日期：列车出发日期
- 出发时间：列车出发时间 (HH:MM格式)
- 到达时间：列车到达时间 (HH:MM格式)

## 使用方法

### 1. 数据准备

将训练数据放在 [datasets/train/](file:///D:/pytorch/%E8%BD%A6%E7%AE%B1/datasets/train) 目录下，测试数据放在 [datasets/test/](file:///D:/pytorch/%E8%BD%A6%E7%AE%B1/datasets/test) 目录下。数据文件应为CSV格式。

### 2. 模型训练

```bash
python train.py
```

该命令将使用训练数据训练所有模型，并将训练好的模型保存在 [model/](file:///D:/pytorch/%E8%BD%A6%E7%AE%B1/model) 目录下。

### 3. 模型预测

```bash
python predict.py
```

该命令将使用训练好的模型对测试数据进行预测，并将预测结果保存在 [predictions/](file:///D:/pytorch/%E8%BD%A6%E7%AE%B1/predictions) 目录下。

### 4. 集成模型训练与预测

```bash
# 训练集成模型
python train_integrated_ensemble.py

# 使用集成模型进行预测
python predict_integrated_ensemble.py
```

### 5. 增量训练

```bash
python incremental_train.py
```

增量训练允许在已有模型基础上使用新数据继续训练。

## 特征工程

系统实现了丰富的特征工程，包括：

- 时间特征提取（小时、分钟、星期等）
- 车站编码
- 时间与车站的交叉特征
- 周期性时间编码
- 历史统计特征

## 模型集成策略

项目提供两种模型集成策略：

1. 简单平均：对所有模型预测结果进行平均
2. 加权集成：根据模型在验证集上的表现动态分配权重

集成模型会自动计算各子模型的权重，性能越好的模型获得越高的权重。

## 训练优化

- 早停机制防止过拟合
- 学习率调度器自动调整学习率
- 梯度裁剪防止梯度爆炸
- 模型权重衰减正则化

## 预测结果

预测结果将保存为CSV文件，包含以下字段：
- 车次ID
- 车站名
- 出发日期
- 出发时间
- 预测延误分钟（整数）

每个模型都会生成独立的预测结果文件，文件命名格式为：
- 单模型预测：`prediction_results_{model_name}_{test_file_name}.csv`
- 平均集成预测：`prediction_results_average_{test_file_name}.csv`
- 动态权重集成预测：`prediction_results_integrated_ensemble_{test_file_name}.csv`

## 性能监控

训练过程中使用进度条显示训练进度，并实时显示训练和验证损失。模型会在验证损失改善时自动保存最佳权重。

## 许可证

本项目仅供学习和研究使用。