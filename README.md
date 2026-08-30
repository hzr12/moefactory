# 🚄 火车延误预测系统

混合集成学习系统，结合 **6个深度学习模型 + 4个机器学习模型**，用于高速铁路到站延误分钟数预测。所有模型预测 **延误分钟数**（正数=晚点，负数=早点，零=准点）。预测值**不裁剪**。

---

## 1. 快速开始

```bash
pip install -r requirements.txt

# 完整训练流程 (预计1-2h)
python train.py

# 增量训练 (30-60min)
python incremental_train.py

# 训练集成器权重 (5-10min)
python train_integrated_ensemble.py

# 预测（推荐）
python predict_integrated_ensemble.py
```

---

## 2. 架构总览

```
┌──────────────────────────────────────┐
│         config.py (配置)              │  ← 所有超参数统一管理
├──────────────────────────────────────┤
│       data_preprocess.py             │  ← 特征工程、Dataset
├──────────────────────────────────────┤
│  ┌────────────────────────────────┐  │
│  │   StationEmbeddingModelV2     │  │  ← 独立预训练Embedding
│  │   (8D车站 + 4D列车类型 + 4D时段)│  │     供ML模型使用
│  └────────────┬───────────────────┘  │
│               ↓                      │
│  ┌────────────────────────────────┐  │
│  │  ML模型: RF/LGB/XGB/CAT       │  │  ← 4个传统模型
│  │  输入: 17D时间 + 12D嵌入     │  │     (共29D)
│  └────────────────────────────────┘  │
│                                       │
│  ┌────────────────────────────────┐  │
│  │  6个DL模型:                   │  │  ← 端到端学习
│  │  Transformer/LSTM/Seq2Seq/    │  │     内部4D+2D+2D嵌入
│  │  TCN/Neural ODE/StationGNN    │  │
│  └────────────┬───────────────────┘  │
│               ↓                      │
│  ┌────────────────────────────────┐  │
│  │  IntegratedEnsembleModel      │  │  ← Bagging权重 + Stacking
│  │  (10个模型, 加权/Stacking集成) │  │
│  └────────────────────────────────┘  │
├──────────────────────────────────────┤
│    predict_integrated_ensemble.py    │  ← 逐站顺序预测
│    + LagFeatureStrategy              │     带误差传播缓解
└──────────────────────────────────────┘
```

---

## 3. 特征工程

### 3.1 原始数据格式

训练数据CSV列：

| 列名 | 类型 | 示例 | 说明 |
|------|------|------|------|
| `车次ID` | str | G339 | 列车车次编号 |
| `车站名` | str | 北京西 | 站点名称 |
| `到达日期` | str | 2025-08-22 | 到达日期 |
| `到达时间` | str | 12:26 | 到达时间 |
| `出发日期` | str | 2025-08-22 | 出发日期 |
| `出发时间` | str | 12:26 | 出发时间 |
| `延误分钟` | float | 0 | 目标值：延误分钟（正=晚点，负=早点） |

测试数据**完全相同，但无`延误分钟`列**。

### 3.2 完整特征规格

所有特征列在 `data_preprocess.py` 中定义为常量：

#### 14维 Sin/Cos 时间编码 (`SIN_COS_FEATURE_COLUMNS`)

| 特征 | 公式 | 周期 |
|------|------|------|
| `出发小时_sin` | sin(2π × hour/24) | 24h |
| `出发小时_cos` | cos(2π × hour/24) | 24h |
| `出发分钟_sin` | sin(2π × minute/60) | 60min |
| `出发分钟_cos` | cos(2π × minute/60) | 60min |
| `出发月份_sin` | sin(2π × month/12) | 12mo |
| `出发月份_cos` | cos(2π × month/12) | 12mo |
| `出发日_sin` | sin(2π × day/31) | 31d |
| `出发日_cos` | cos(2π × day/31) | 31d |
| `出发星期_sin` | sin(2π × weekday/7) | 7d |
| `出发星期_cos` | cos(2π × weekday/7) | 7d |
| `到达小时_sin` | 同出发小时 | 24h |
| `到达小时_cos` | 同出发小时_cos | 24h |
| `到达分钟_sin` | 同出发分钟 | 60min |
| `到达分钟_cos` | 同出发分钟_cos | 60min |

#### 2维二值特征 (`BINARY_FEATURE_COLUMNS`)

| 特征 | 值 | 逻辑 |
|------|-----|------|
| `是否周末` | 0/1 | 周六/周日=1 |
| `是否节假日` | 0/1 | 法定节假日+周末=1 |

法定节假日：元旦(1,1)、劳动节(5,1)-(5,3)、国庆节(10,1)-(10,3)。

#### 1维滞后特征 (`LAG_FEATURE_COLUMNS`)

| 特征 | 说明 |
|------|------|
| `上一站延误` | 按车次分组取上一站延误值（shift(1)），NaN填充为0 |

#### 列车类型编码（整数索引 → DL端到端Embedding）

| 编码 | 前缀 | 类型 |
|------|------|------|
| 0 | G | 高铁 |
| 1 | D | 动车 |
| 2 | C | 城际 |
| 3 | K | 快速 |
| 4 | T | 特快 |
| 5 | Z | 直达 |
| 6 | Other | 其他 |

#### 发车时段编码（整数索引 → DL端到端Embedding）

| 编码 | 时段 | 小时范围 |
|------|------|----------|
| 0 | 凌晨 | 0:00-5:59 |
| 1 | 早高峰 | 6:00-8:59 |
| 2 | 上午 | 9:00-11:59 |
| 3 | 午间 | 12:00-13:59 |
| 4 | 下午 | 14:00-16:59 |
| 5 | 晚高峰 | 17:00-19:59 |
| 6 | 夜间 | 20:00-23:59 |

### 3.3 DL模型输入（17维时间 + 3个整数索引）

`ALL_TIME_FEATURE_COLUMNS` = `SIN_COS_FEATURE_COLUMNS` (14D) + `STD_FEATURE_COLUMNS` (3D: `是否周末`, `是否节假日`, `上一站延误`)

总原始输入：**17维时间特征**(float) + **3个整数索引**(车站编码, 列车类型编码, 发车时段编码)

模型内部拼接：`[time_features(17), station_embedding(4), tt_embedding(2), dw_embedding(2)]` = **最终25维输入**。

### 3.4 ML模型输入（29维）

ML模型使用：`[time_features(17), station_embedding(8), station_bias(1), station_variability(1), tt_embedding(4), dw_embedding(4)]` = **29维**（全部来自独立预训练的StationEmbeddingModelV2）。

### 3.5 标准化策略

- **Sin/Cos特征**：值域固定在[-1,1]，**无需标准化**。
- **二值特征 + 滞后特征**：`StandardScaler`（仅在训练集上拟合，变换训练集和验证集）。
- **目标值y**：`StandardScaler`（预测后反标准化恢复原始值）。

### 3.6 车站编码

```python
le_station = LabelEncoder()
le_station.fit(train_df['车站名'])
# 测试集中未知车站 → 编码为 -1
```

---

## 4. 数据加载与Dataset

### `load_train_data(train_dir)`

读取`train_dir`下所有CSV文件，合并为单个DataFrame。跳过空文件。

### `load_test_data(test_file)`

读取单个测试CSV文件。

### `TrainDelayDataset` (PyTorch `Dataset`)

```python
class TrainDelayDataset(Dataset):
    def __init__(self, time_features, station_indices, targets=None, train_type_idx=None, dep_window_idx=None)

    def __getitem__(self, idx):
        # 训练模式: (time_feat, station_idx, targets, tt_idx, dw_idx) — 5元组
        # 预测模式 (targets=None): (time_feat, station_idx, tt_idx, dw_idx) — 4元组
```

### `prepare_train_data(train_df)`

返回：`(X_time, station_indices, y, scaler, num_stations, X_ml, scaler_y, train_type_idx, dep_window_idx)`

- 通过`add_lag_features()`添加滞后特征：`df.groupby('车次ID')['延误分钟'].shift(1).fillna(0)`
- 提取时间特征
- 对非sin/cos特征和目标y做标准化
- `X_ml = X_time.copy()`（占位，后续由`build_ml_features_from_embedding`填充）

### `prepare_test_data(test_df, scaler, num_stations, lag_values)`

返回：`(X_time, station_indices, X_ml, train_type_idx, dep_window_idx)`

- `lag_values`参数填充`上一站延误`列（None时为全0）
- 非sin/cos特征用训练集scaler标准化

### `split_by_train_id(df, test_size, random_state)`

**按车次ID**分层划分，而非随机切分——确保同一车次的站点不会泄漏到另一集合中。

### `build_ml_features_from_embedding(X_time, station_indices, embedding_weights, ...)`

拼接构建29维ML特征：`[X_time, station_emb(8D), bias(1D), variability(1D), tt_emb(4D), dw_emb(4D)]`。

---

## 5. 配置管理 (`config.py`)

```python
@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    path: PathConfig
    feature: FeatureConfig
```

### `ModelConfig` — 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `time_dim` | 14 | 时间特征维度（运行时动态更新为17） |
| `num_stations` | 18 | 车站数量（运行时动态更新） |
| `emb_dim` | 4 | DL模型车站嵌入维度 |
| `tt_emb_dim` | 2 | DL模型列车类型嵌入维度（7类→2维） |
| `dw_emb_dim` | 2 | DL模型发车时段嵌入维度（7时段→2维） |
| `station_emb_dim` | 8 | 独立车站嵌入维度（供ML模型用） |
| `station_emb_hidden` | 64 | 独立嵌入模型隐藏层大小 |
| `station_emb_tt_dim` | 4 | 独立列车类型嵌入维度（供ML模型用） |
| `station_emb_dw_dim` | 4 | 独立发车时段嵌入维度（供ML模型用） |
| `transformer_d_model` | 64 | Transformer隐藏层维度 |
| `transformer_nhead` | 4 | 注意力头数 |
| `transformer_num_layers` | 2 | 编码器层数 |
| `transformer_dim_feedforward` | 256 | FFN隐藏层维度 |
| `transformer_dropout` | 0.15 | Dropout |
| `lstm_hidden_size` | 64 | LSTM隐藏层维度 |
| `lstm_num_layers` | 2 | LSTM层数 |
| `lstm_dropout` | 0.15 | Dropout |
| `seq2seq_hidden_size` | 64 | Seq2Seq隐藏层维度 |
| `seq2seq_num_layers` | 2 | LSTM层数 |
| `seq2seq_dropout` | 0.25 | Dropout |
| `tcn_num_channels` | [32,64,128] | 每层通道数 |
| `tcn_kernel_size` | 3 | 卷积核大小 |
| `tcn_dropout` | 0.2 | Dropout |
| `ode_hidden_dim` | 64 | ODE隐藏层维度 |
| `ode_dropout` | 0.2 | Dropout |
| `ode_num_steps` | 10 | Euler求解步数 |
| `ode_dt` | 0.1 | 时间步长 |
| `gnn_hidden_dim` | 64 | GNN隐藏层维度 |
| `gnn_num_heads` | 4 | 注意力头数 |
| `gnn_dropout` | 0.2 | Dropout |
| `transformer_lr` | 0.0005 | 各模型独立学习率 |
| `lstm_lr` | 0.001 | 各模型独立学习率 |
| `seq2seq_lr` | 0.001 | 各模型独立学习率 |
| `tcn_lr` | 0.001 | 各模型独立学习率 |
| `ode_lr` | 0.0003 | ODE用小lr防止发散 |
| `gnn_lr` | 0.0008 | 各模型独立学习率 |

### `TrainingConfig` — 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_epochs` | 200 | 最大训练轮数 |
| `batch_size` | 128 | 批次大小 |
| `learning_rate` | 0.001 | 基础学习率 |
| `weight_decay` | 0.01 | AdamW权重衰减 |
| `max_lr` | 0.005 | OneCycleLR峰值 |
| `patience` | 30 | 早停耐心值 |
| `huber_delta` | 5.0 | Huber损失delta |
| `grad_max_norm` | 1.0 | 梯度裁剪阈值 |
| `use_amp` | True | 混合精度训练 |
| `warmup_epochs` | 5 | 预热轮数 |
| `use_cosine_annealing` | True | 余弦退火调度 |
| `incremental_lr` | 0.0001 | 增量训练LR（正常1/10） |
| `incremental_epochs` | 10 | 增量训练轮数 |
| `incremental_patience` | 3 | 增量训练早停 |
| `ensemble_n_bootstrap` | 10 | Bagging采样次数 |

### `PathConfig`

所有模型/编码器/标准化文件在`./model/`。预测输出在`./predictions/`。训练数据在`./datasets/train/`。

### `FeatureConfig`

定义`sin_cos_features`、`binary_features`、`lag_features`列表。`time_features`属性拼接三者（共17维）。

---

## 6. 各模型完整层结构

### 6.1 `DynamicEmbedding` (models.py:18)

统一处理已知和未知车站索引。

```python
class DynamicEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=-1):
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.unknown_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.01)

    def forward(self, indices):
        # indices: (batch,) → 已知 → self.embedding(known_indices)
        # indices: (batch,) → 未知 (<0或≥num_embeddings) → self.unknown_embedding

    def expand_embeddings(self, new_num_embeddings):
        # 创建新Embedding，复制旧权重，新车站随机初始化
```

### 6.2 `TransformerPredictor` (models.py:93)

```
输入:  time_features(batch,17), station_indices(batch,1), tt_idx(batch,1), dw_idx(batch,1)
        │
        ├── station_embedding(DynamicEmbedding, 18→4D) → (batch,4)
        ├── train_type_embedding(Embedding, 7→2D) → (batch,2)
        ├── dep_window_embedding(Embedding, 7→2D) → (batch,2)
        │
        └── concat([time(17), station(4), tt(2), dw(2)]) → (batch,25)
            │
            ├── unsqueeze(1) → (batch,1,25)
            ├── input_projection(Linear 25→64) → (batch,1,64)
            ├── PositionalEncoding(dropout=0.15, max_len=5000)
            │   └── pe = sin/cos(position/10000^(2i/d_model))
            ├── TransformerEncoder × 2 (batch_first=True)
            │   └── 每层: MultiheadAttention(64, 4头) → FFN(64→256→64)
            ├── mean(dim=1) → (batch,64)
            │
            └── output_layer: Linear(64→32) → ReLU → Dropout(0.15) → Linear(32→1)
                → (batch,1)
```

### 6.3 `LSTMPredictor` (models.py:234)

```
输入:  time_features(batch,17), station_indices(batch,1), tt_idx(batch,1), dw_idx(batch,1)
        │
        ├── station_embedding(DynamicEmbedding, 18→4D) → (batch,4)
        ├── tt_embedding(Embedding, 7→2D) → (batch,2)
        ├── dw_embedding(Embedding, 7→2D) → (batch,2)
        │
        ├── concat([time(17), station(4), tt(2), dw(2)]) → (batch,25)
        ├── unsqueeze(1) → (batch,1,25)
        │
        ├── LSTM(25→64, num_layers=2, batch_first=True, dropout=0.15)
        │   └── output: (batch,1,64), (h_n, c_n): 各(2,batch,64)
        ├── output[:, -1, :] → (batch,64)
        ├── Dropout(0.15)
        │
        └── fc(Linear 64→1) → (batch,1)
```

### 6.4 `Seq2SeqPredictor` (models.py:315)

```
输入:  time_features(batch,17), station_indices(batch,1), tt_idx(batch,1), dw_idx(batch,1)
        │
        ├── station_embedding(DynamicEmbedding, 18→4D) → (batch,4)
        ├── tt_embedding(Embedding, 7→2D) → (batch,2)
        ├── dw_embedding(Embedding, 7→2D) → (batch,2)
        │
        ├── concat([time(17), station(4), tt(2), dw(2)]) → (batch,25)
        ├── unsqueeze(1) → (batch,1,25)
        │
        ├── Encoder LSTM(25→64, num_layers=2, batch_first=True, dropout=0.25)
        │   └── encoder_output: (batch,1,64)
        ├── MultiheadAttention(64, 4头, batch_first=True)
        │   └── attn_output = Attention(encoder_output, encoder_output, encoder_output)
        ├── LayerNorm(64) — 残差: encoder_output + attn_output
        ├── output[:, -1, :] → (batch,64)
        │
        └── output_layer: Linear(64→32) → ReLU → Dropout(0.25) → Linear(32→1)
            → (batch,1)
```

### 6.5 `TCNPredictor` (models.py:896)

```
输入:  time_features(batch,17), station_indices(batch,1), tt_idx(batch,1), dw_idx(batch,1)
        │
        ├── station_embedding(DynamicEmbedding, 18→4D) → (batch,4)
        ├── tt_embedding(Embedding, 7→2D) → (batch,2)
        ├── dw_embedding(Embedding, 7→2D) → (batch,2)
        │
        ├── concat([time(17), station(4), tt(2), dw(2)]) → (batch,25)
        ├── unsqueeze(-1) → (batch, 25, 1)
        │
        ├── TCN块 (3层):
        │   第1层: Conv1d(25→32, k=3, dilation=1, padding=2) + weight_norm
        │           → ReLU → Dropout(0.2) → BatchNorm1d(32)
        │   第2层: Conv1d(32→64, k=3, dilation=2, padding=4) + weight_norm
        │           → ReLU → Dropout(0.2) → BatchNorm1d(64)
        │   第3层: Conv1d(64→128, k=3, dilation=4, padding=8) + weight_norm
        │           → ReLU → Dropout(0.2) → BatchNorm1d(128)
        │   → (batch, 128, seq_len')
        │
        ├── x[:, :, -1:] → (batch, 128, 1)  # 因果：只取最后时间步
        ├── squeeze(-1) → (batch, 128)
        │
        └── output_layer: Linear(128→32) → ReLU → Dropout(0.2) → Linear(32→1)
            → (batch,1)
```

### 6.6 `NeuralODEPredictor` (models.py:1023)

连续时间动力学模型。核心思想：`d(延误)/dt = f(延误, 车站, 特征) - 衰减 × 状态`。

```
输入:  time_features(batch,17), station_indices(batch,1), tt_idx(batch,1), dw_idx(batch,1)
        │
        ├── station_embedding(DynamicEmbedding, 18→4D) → (batch,4)
        ├── tt_embedding(Embedding, 7→2D) → (batch,2)
        ├── dw_embedding(Embedding, 7→2D) → (batch,2)
        │
        ├── concat([time(17), station(4), tt(2), dw(2)]) → (batch,25)
        │
        ├── initial_encoder: Linear(25→64) → LayerNorm(64) → ReLU → Dropout(0.2)
        │   → state: (batch,64)
        │
        ├── ODE求解器 (Euler法, num_steps=10, dt=0.1):
        │   for step in range(10):
        │       derivative = ode_func_net(state)
        │       │   └── Linear(64→64) → Tanh → Dropout(0.2) → Linear(64→64) → Tanh
        │       decay_rate = decay_rate_net(state)
        │       │   └── Linear(64→32) → ReLU → Linear(32→1) → Sigmoid → (batch,1)
        │       adjusted = derivative - decay_rate * state
        │       state = state + dt * adjusted  → (batch,64)
        │
        └── output_layer: Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→1)
            → (batch,1)
```

### 6.7 `StationGNN` (models.py:1692)

图神经网络，建模站点间延误传播关系。

```
输入:  time_features(batch,17), station_indices(batch,1), tt_idx(batch,1), dw_idx(batch,1)
        │
        ├── station_embedding(DynamicEmbedding, 18→4D) → 所有车站embedding: (num_stations,4)
        ├── tt_embedding(Embedding, 7→2D) → (batch,2)
        ├── dw_embedding(Embedding, 7→2D) → (batch,2)
        │
        ├── concat([time(17), tt(2), dw(2)]) → (batch,19)
        ├── time_encoder: Linear(19→64) → LayerNorm(64) → ReLU → Dropout(0.2)
        │   → time_repr: (batch,64)
        │
        ├── all_station_emb: Embedding(18→4) → emb_proj(Linear 4→64) → (num_stations,64)
        │
        ├── 图卷积 (2层):
        │   选项A (GAT): SimpleGATConv(64→64, heads=4)
        │       └── Q/K/V投影 → einsum注意力 → softmax → 加权求和 → out_proj → LayerNorm
        │   选项B (SGC): SimpleSGConv(64→64, hops=2)
        │       └── D^{-1/2}(I+A)D^{-1/2}归一化 → K跳聚合 → Linear → LayerNorm
        │   
        ├── ReLU
        ├── conv2: (同上)
        │
        ├── 从graph_out索引batch站点的图表示: (batch,64)
        ├── concat([time_repr, graph_repr]) → (batch,128)
        ├── fuse_proj(Linear 128→64) → ReLU
        │
        └── output_layer: Linear(64→64) → ReLU → Dropout(0.2) → Linear(64→1)
            → (batch,1)
```

#### `SimpleGATConv` (models.py:1546) — GAT模式

```
Q = W_q(x) → reshape (N, heads, head_dim)
K = W_k(x) → reshape (N, heads, head_dim)
V = W_v(x) → reshape (N, heads, head_dim)
attn = einsum('ihd,jhd→hij', Q, K) / sqrt(head_dim)
attn = softmax(mask(attn, adj_matrix))  # 仅关注邻居节点
out = einsum('hij,hjd→ihd', attn, V) → reshape → out_proj → LayerNorm(residual)
```

#### `SimpleSGConv` (models.py:1622) — SGC模式

```
adj_norm = D^{-1/2}(I+A)D^{-1/2}
h = x
for _ in range(num_hops): h = adj_norm @ h
out = Linear(h) → Dropout → LayerNorm
```

### 6.8 `StationEmbeddingModelV2` (models.py:544)

独立车站嵌入模型，支持多任务学习：

- **主任务**：延误预测（与DL模型类似，但使用8维车站嵌入 + 4维tt + 4维dw）
- **图正则化**：邻接车站在嵌入空间中更近
- **辅助任务**：车站邻接关系预测（给定两个embedding，预测是否邻接）
- **过渡损失**：`T(emb_up, emb_down) × delay_up ≈ delay_down - delay_up`

```
station_emb(DynamicEmbedding, 18→8D)
train_type_embedding(Embedding, 7→4D)
dep_window_embedding(Embedding, 7→4D)
station_bias(DynamicEmbedding, 18→1D)
station_variability(DynamicEmbedding, 18→1D)  # softplus确保正值

主任务MLP: [time(17) + station_emb(8) + tt(4) + dw(4)] → Linear(33→64) → LayerNorm → ReLU
         → Dropout → Linear(64→32) → LayerNorm → ReLU → Dropout → Linear(32→1)

辅任务头: [emb_i || emb_j] → Linear(16→32) → ReLU → Linear(32→1) → Sigmoid

损失 = HuberLoss(delay_pred, y)
     + 0.1 × graph_reg_loss(embeddings, adjacency_matrix)
     + 0.1 × adj_pred_loss
     + 0.1 × transition_loss
```

### 6.9 `StationGraph` (models.py:1186)

定义站点拓扑结构，带权邻接矩阵：

- **主线边**（权重=1.0）：京广高铁沿线顺序站点
- **支线边**（权重=0.8）：如北京西↔天津西、石家庄↔济南西
- **换乘枢纽边**（权重=0.5）：跨线路连接

默认18个站点，训练时通过`from_train_data()`动态构建。

### 6.10 `GraphRegularizationLoss` (models.py:469)

```python
loss = mean(||emb_i - emb_j||²)  (邻接车站)
     + 0.1 × mean(ReLU(margin - ||emb_i - emb_j||))  (非邻接车站)
```

### 6.11 ML模型（通过scikit-learn/LightGBM/XGBoost/CatBoost训练）

| 模型 | 关键参数 |
|------|---------|
| `RandomForestRegressor` | n_estimators=1000, max_depth=15, min_samples_split=5, min_samples_leaf=2 |
| `LGBMRegressor` | num_leaves=127, learning_rate=0.01, n_estimators=500 |
| `XGBRegressor` | n_estimators=1000, max_depth=6, learning_rate=0.1 |
| `CatBoostRegressor` | iterations=1000, depth=6, learning_rate=0.1 |

---

## 7. 训练流程 (`train.py`)

### 步骤详细说明：

1. **加载数据**：`load_train_data()` → 合并所有CSV文件
2. **编码车站**：`LabelEncoder` → `station_to_code` 映射
3. **按车次ID切分**：80/20 训练/验证集划分
4. **准备特征**：`prepare_train_data()` → 特征工程 + 标准化
5. **构建站点图**：`StationGraph.from_train_data()` 从实际站名构建
6. **提取连续站点对**：`extract_consecutive_pairs()` 用于过渡损失
7. **训练StationEmbeddingModelV2**：
   - 优化器：AdamW (lr=0.001, weight_decay=0.01)
   - 调度器：OneCycleLR (max_lr=0.005)
   - 损失：多任务（主延误 + 图正则化 + 邻接预测 + 过渡损失）
   - 主损失：HuberLoss(delta=5.0)
   - 梯度裁剪：max_norm=1.0
   - 早停：patience=30
8. **提取并保存Embedding**：station(8D), tt(4D), dw(4D), bias(1D), variability(1D)
9. **构建ML特征**：`build_ml_features_from_embedding()` → 29维特征
10. **训练ML模型**：RF、LightGBM、XGBoost、CatBoost 在29维特征上
11. **训练6个DL模型**：每个模型调用`train_model()`：
    - 优化器：AdamW（各模型独立学习率）
    - 调度器：OneCycleLR (max_lr=0.005)
    - 损失：HuberLoss(delta=5.0)
    - 混合精度（AMP，如果可用）
    - 梯度裁剪：max_norm=1.0
    - 早停：patience=30
12. **保存配置**：`model_config.pkl` 包含所有架构参数

### `train_model()` 函数详情：

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, model_name, patience=50, use_onecycle=False, config=None, use_amp=None):
    # 训练循环（支持AMP: autocast + GradScaler）
    # 验证循环
    # 按val_loss保存最佳模型为 ./model/{model_name}_best.pth
    # 早停
```

### `train_emb_v2_model()` 函数详情：

```python
def train_emb_v2_model(model, train_loader, val_loader, optimizer, scheduler,
                       num_epochs, device, model_name, adjacency_matrix=None,
                       consecutive_pairs=None, transition_weight=0.1, patience=50, use_onecycle=False):
    # 使用model.compute_loss()而非外部criterion
    # 支持图正则化、邻接预测、过渡损失
```

---

## 8. 集成模型 (`integrated_ensemble.py`)

### `IntegratedEnsembleModel`

协调所有10个模型的类：

```python
class IntegratedEnsembleModel:
    def __init__(self, time_dim, num_stations, emb_dim, scaler_y=None, config=None):
        # 创建所有6个DL模型（Transformer, LSTM, Seq2Seq, TCN, ODE, GNN）
        # 参数来自config字典（回退到default_config）
        # ML模型通过pickle单独加载
```

### 两层集成架构

#### 第0层：Bagging动态权重

```python
def calculate_weights(self, time_features, station_indices, y_val, X_ml=None,
                      n_bootstrap=10, train_type_idx=None, dep_window_idx=None):
    # 每次Bootstrap迭代：
    #   1. 有放回采样验证集
    #   2. 获取所有10个模型的预测
    #   3. 计算每个模型的MSE
    #   4. weight_i = 1 / (MSE_i + 1e-8)
    #   5. 归一化权重使之和为1
    # 对所有Bootstrap迭代的权重取平均
```

#### 第1层：Stacking元学习器

```python
def train_stacking_meta_learner(self, ..., meta_learner_type='ridge'):
    # 第0层：10个基模型 → 10维预测向量
    # 第1层：元学习器训练 (10维预测 → 真实y)
    #
    # ridge: sklearn.linear_model.RidgeCV(alphas=[0.01,0.05,0.1,0.5,1.0,5.0,10.0,50.0,100.0])
    # lightgbm: LGBMRegressor(n_estimators=200, max_depth=3, lr=0.05, num_leaves=7, ...)
```

#### 预测

```python
def predict(self, time_features, station_indices, X_ml=None, use_stacking=True, ...):
    # 1. 如果有Stacking + 元学习器：获取10个模型预测 → meta_learner.predict()
    # 2. 否则：使用Bagging权重做加权平均
    # 3. 用scaler_y反标准化预测结果
```

### Stacking 预测架构

```
测试样本
    │
    ├──→ Transformer → ┐
    ├──→ LSTM → ───────┤
    ├──→ Seq2Seq → ────┤
    ├──→ TCN → ────────┤──→ [10维预测向量] → 元学习器 → 最终预测
    ├──→ Neural ODE → ─┤                     (Ridge/LightGBM)
    ├──→ StationGNN → ─┤
    ├──→ RF → ─────────┤
    ├──→ LightGBM → ───┤
    ├──→ XGBoost → ────┤
    └──→ CatBoost → ───┘
```

---

## 9. 误差修正预测 (`predict_integrated_ensemble.py`)

### 逐站预测流程

```
对于每个车次ID:
    1. 重置 LagFeatureStrategy
    2. 按出发时间排序站点
    3. 对于每个站点:
        a. prev_delay = lag_strategy.process(上一站预测值, 位置)
        b. prepare_test_data(该行, lag_values=[prev_delay])
        c. pred = ensemble_model.predict(X_time, station_indices, ...)
        d. 保存pred
        e. lag_strategy.process_lag_feature(pred, 位置) → 用于下一站的prev_delay
```

### `LagFeatureStrategy`

```python
class LagFeatureStrategy:
    def __init__(self, decay_factor=0.85, window_size=3):
        self.recent_delays = deque(maxlen=window_size)

    def process_lag_feature(self, predicted_delay, position):
        # 1. 添加到历史记录
        # 2. 移动平均: mean(recent_delays)
        # 3. 衰减: adjusted = smoothed × (decay_factor ** (position - 1))
```

### `ErrorPropagationAnalyzer`

记录每个站点位置的预测误差，生成包含平均误差、标准差、误差增长率的报告。

---

## 10. 增量训练 (`incremental/`)

### `StationExpander`

检测增量数据中的新车站，更新`LabelEncoder`，扩展`DynamicEmbedding`层。

### `EmbeddingTrainer`

加载预训练`StationEmbeddingModelV2`，扩展embedding以支持新车站，增量训练（lr=0.0001, epochs=10, patience=3），提取并保存更新后的embedding。

### `DLModelTrainer`

增量训练DL模型（Transformer/LSTM/Seq2Seq）：
- AdamW (lr=0.0001, weight_decay=0.01)
- OneCycleLR (max_lr=0.0005)
- 梯度裁剪 (max_norm=1.0)
- 早停 (patience=3)
- 最多10轮

### `MLModelTrainer`

在新数据上重新训练RF/LightGBM/XGBoost/CatBoost（完整重训练，非增量）。

### `VersionManager`

将版本信息保存到`logs/incremental/incremental_version.json`，版本号递增（v1, v2, ...）。

---

## 11. 优化工具 (`optimizations.py`)

### `PreLNTransformerEncoderLayer`

Pre-LayerNorm Transformer编码器：`x → LN → Attention → Add → LN → FFN → Add`。比Post-LN更稳定。

### `AttentionLSTM`

在LSTM输出上添加注意力：`attn_weights = softmax(Linear(lstm_output))`, `context = sum(lstm_output × attn_weights)`。

### `ResidualGNNLayer`

GCN层：`output = LN(Linear(AX) + X)` 带残差连接。

### `WarmupCosineScheduler`

学习率调度：线性预热 → 余弦退火衰减。

### `create_stacking_ensemble()`

创建`StackingEnsemble`类：基模型预测 → Ridge元学习器。

---

## 12. 完整文件参考

### 核心源文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `config.py` | 268 | `Config`, `ModelConfig`, `TrainingConfig`, `PathConfig`, `FeatureConfig` 配置类 |
| `models.py` | 1854 | 所有6个DL模型 + StationEmbeddingModelV1/V2 + StationGraph + GNN层 |
| `data_preprocess.py` | 711 | 数据加载、特征工程、Dataset、编码、标准化 |
| `integrated_ensemble.py` | 721 | `IntegratedEnsembleModel` 含Bagging权重 + Stacking |
| `optimizations.py` | 213 | Pre-LN Transformer, AttentionLSTM, ResidualGNN, WarmupCosine, StackingEnsemble |
| `train.py` | 874 | 完整训练流程（Embedding → ML → DL） |
| `train_integrated_ensemble.py` | 258 | 集成器权重训练（Bagging + Stacking） |
| `predict.py` | 306 | 单模型预测脚本 |
| `predict_integrated_ensemble.py` | 400 | 集成模型预测（含误差修正） |
| `incremental_train.py` | 329 | 增量训练主脚本 |

### 增量模块 (`incremental/`)

| 文件 | 行数 | 说明 |
|------|------|------|
| `__init__.py` | 19 | 模块导出 |
| `config.py` | 85 | `IncrementalConfig` 配置类 |
| `station_expander.py` | 175 | `StationExpander` — 新车站检测 + embedding扩展 |
| `embedding_trainer.py` | 264 | `EmbeddingTrainer` — 增量embedding训练 |
| `model_trainer.py` | 224 | `DLModelTrainer` + `MLModelTrainer` |
| `version_manager.py` | 128 | `VersionManager` — 版本号管理 |

### 保存的模型文件 (`model/`)

| 文件 | 类型 | 说明 |
|------|------|------|
| `transformer_best.pth` | PyTorch | Transformer权重 |
| `lstm_best.pth` | PyTorch | LSTM权重 |
| `seq2seq_best.pth` | PyTorch | Seq2Seq权重 |
| `tcn_best.pth` | PyTorch | TCN权重 |
| `ode_best.pth` | PyTorch | Neural ODE权重 |
| `gnn_best.pth` | PyTorch | StationGNN权重 |
| `station_emb_best.pth` | PyTorch | StationEmbeddingModelV2权重 |
| `random_forest_best.pkl` | Pickle | 随机森林 |
| `lightgbm_best.pkl` | Pickle | LightGBM |
| `xgboost_best.pkl` | Pickle | XGBoost |
| `catboost_best.pkl` | Pickle | CatBoost |
| `stacking_meta_learner.pkl` | Pickle | Stacking元学习器 + model_order |
| `label_encoder.pkl` | Pickle | 车站LabelEncoder |
| `scaler.pkl` | Pickle | 特征StandardScaler |
| `scaler_y.pkl` | Pickle | 目标StandardScaler |
| `station_embedding.pkl` | Pickle | 车站嵌入 (num_stations × 8) |
| `tt_embedding.pkl` | Pickle | 列车类型嵌入 (7 × 4) |
| `dw_embedding.pkl` | Pickle | 发车时段嵌入 (7 × 4) |
| `station_bias.pkl` | Pickle | 车站偏差 (num_stations × 1) |
| `station_variability.pkl` | Pickle | 车站变异性 (num_stations × 1) |
| `station_graph.pkl` | Pickle | StationGraph 含邻接矩阵 |
| `model_config.pkl` | Pickle | 架构参数字典 |
| `integrated_ensemble_info.pkl` | Pickle | Bagging权重 |

---

## 13. 依赖 (`requirements.txt`)

```
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
catboost>=1.0.0
tqdm>=4.62.0
```

---

## 14. 关键设计决策与易错提示

1. **预测值不裁剪**：正数=晚点，负数=早点，零=准点。
2. **滞后特征零填充**：首站无上一站 → 滞后=0。
3. **未知车站编码**：测试集中训练集没有的车站 → 编码=-1 → DynamicEmbedding映射到可学习的未知向量。
4. **所有DL模型统一forward签名**：`(time_features, station_indices, train_type_idx, dep_window_idx)` → 返回 `(batch, 1)`。
5. **TrainDelayDataset返回5元组**：训练时 `(time_feat, station_idx, targets, tt_idx, dw_idx)`。预测时 targets=None → 返回4元组。
6. **StationGraph全局单例**：`models.station_graph` 全局实例，train.py中从数据动态构建并保存。
7. **model_config.pkl可能缺少架构参数**：旧版只保存了 time_dim/num_stations/emb_dim。`IntegratedEnsembleModel` 会回退到 `default_config`。
8. **逐站预测时滞后特征**：将上一站预测值作为下一站的滞后特征 → 误差累积。通过 `LagFeatureStrategy`（衰减 + 移动平均）缓解。
9. **标准化策略**：非sin/cos特征（二值+滞后）用 `StandardScaler`。Sin/Cos特征已在[-1,1]。
10. **按车次ID划分验证集**：确保同一车次的所有站点不会跨训练集/验证集泄漏。

---

## 15. 复现指南

从零完整复现本系统：

1. 创建 `datasets/train/`，放入CSV文件（每日期一个文件，列名：车次ID, 车站名, 到达日期, 到达时间, 出发日期, 出发时间, 延误分钟）
2. 创建 `datasets/test/`，放入测试CSV文件（列名同上，不含延误分钟）
3. 运行 `python train.py`（训练 Embedding → ML → DL 模型）
4. 运行 `python train_integrated_ensemble.py`（训练集成权重 + Stacking）
5. 运行 `python predict_integrated_ensemble.py`（生成预测）

所有超参数在 `config.py` 中。所有模型架构在 `models.py` 中。所有数据处理在 `data_preprocess.py` 中。集成逻辑在 `integrated_ensemble.py` 中。