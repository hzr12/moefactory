import torch
import torch.nn as nn
import math

"""
模型定义文件，包含多种预测模型实现
"""

class TransformerPredictor(nn.Module):
    """
    Transformer预测模型
    基于Transformer编码器的时间序列预测模型
    """
    def __init__(self, input_dim=6, d_model=32, nhead=8, num_layers=5, dim_feedforward=128, dropout=0.1):
        """
        初始化Transformer预测模型
        
        Args:
            input_dim (int): 输入特征维度
            d_model (int): 模型隐藏层维度
            nhead (int): 多头注意力头数
            num_layers (int): Transformer编码器层数
            dim_feedforward (int): 前馈网络隐藏层维度
            dropout (float): Dropout概率
        """
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.n_features = input_dim  # 记录特征数量
        
        # 输入投影层，将输入特征映射到模型维度
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码，为序列添加位置信息
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 批处理维度在前
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层，将模型输出映射到预测值
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # 输出单个预测值
        )
        
    def forward(self, src):
        """
        前向传播函数
        
        Args:
            src (torch.Tensor): 输入张量，形状为 (batch_size, features) 或 (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: 预测结果，形状为 (batch_size, 1)
        """
        # 检查输入维度
        if src.dim() == 2:
            # 如果是2D张量 (batch_size, features)，扩展为3D (batch_size, 1, features)
            src = src.unsqueeze(1)
        elif src.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, but got {src.dim()}D")
        
        # 输入投影
        src = self.input_projection(src)
        
        # 添加位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码
        output = self.transformer_encoder(src)
        
        # 全局平均池化，将序列维度压缩
        output = output.mean(dim=1)
        
        # 输出层
        output = self.output_layer(output)
        return output

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    为序列数据添加位置信息，使模型能够区分不同位置的元素
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        初始化位置编码
        
        Args:
            d_model (int): 模型隐藏层维度
            dropout (float): Dropout概率
            max_len (int): 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦
        pe = pe.unsqueeze(0)  # 添加批处理维度
        self.register_buffer('pe', pe)  # 注册为缓冲区，不参与参数更新

    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: 添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1)]  # 添加位置编码
        return self.dropout(x)

class LSTMPredictor(nn.Module):
    """
    LSTM预测模型
    基于长短期记忆网络的时间序列预测模型
    """
    def __init__(self, input_size=6, hidden_size=32, num_layers=5, dropout=0.1):
        """
        初始化LSTM预测模型
        
        Args:
            input_size (int): 输入特征维度
            hidden_size (int): LSTM隐藏层维度
            num_layers (int): LSTM层数
            dropout (float): Dropout概率
        """
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_features = input_size  # 记录特征数量
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 批处理维度在前
            dropout=dropout if num_layers > 1 else 0  # 只有多层时才使用dropout
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        # 输出层
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, features) 或 (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: 预测结果，形状为 (batch_size, 1)
        """
        # 检查输入维度
        if x.dim() == 2:
            # 如果是2D张量 (batch_size, features)，扩展为3D (batch_size, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, but got {x.dim()}D")
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        output = lstm_out[:, -1, :]
        
        # 应用dropout和全连接层
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

class Seq2SeqPredictor(nn.Module):
    """
    Seq2Seq预测模型
    基于编码器-解码器结构的序列到序列预测模型
    """
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.2):
        """
        初始化Seq2Seq预测模型
        
        Args:
            input_size (int): 输入特征维度
            hidden_size (int): 隐藏层维度
            num_layers (int): 层数
            dropout (float): Dropout概率
        """
        super(Seq2SeqPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_features = input_size  # 记录特征数量
        
        # 编码器
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, features) 或 (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: 预测结果，形状为 (batch_size, 1)
        """
        # 检查输入维度
        if x.dim() == 2:
            # 如果是2D张量 (batch_size, features)，扩展为3D (batch_size, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, but got {x.dim()}D")
        
        # 编码器，获取上下文向量
        _, (hidden, cell) = self.encoder(x)
        
        # 解码器，使用编码器的上下文向量进行解码
        decoder_output, _ = self.decoder(x, (hidden, cell))
        
        # 取最后一个时间步的输出
        output = decoder_output[:, -1, :]
        
        # 输出层
        output = self.output_layer(output)
        
        return output

class TFT(nn.Module):
    """
    Temporal Fusion Transformers (TFT) 预测模型
    融合了时间特征和注意力机制的时间序列预测模型
    """
    def __init__(self, input_dim=6, hidden_size=32, num_heads=8, num_layers=2, dropout=0.1):
        """
        初始化TFT预测模型
        
        Args:
            input_dim (int): 输入特征维度
            hidden_size (int): 隐藏层维度
            num_heads (int): 多头注意力头数
            num_layers (int): 注意力层数
            dropout (float): Dropout概率
        """
        super(TFT, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_features = input_dim  # 记录特征数量
        
        # 输入嵌入层，将输入特征映射到模型维度
        self.input_embedding = nn.Linear(input_dim, hidden_size)
        
        # 时间特征处理
        self.time_encoding = PositionalEncoding(hidden_size, dropout)
        
        # 门控机制，用于特征选择
        self.gating = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()  # 生成门控信号
        )
        
        # 多头注意力层，捕获特征间的依赖关系
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # 层归一化，稳定训练过程
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, features) 或 (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: 预测结果，形状为 (batch_size, 1)
        """
        # 检查输入维度
        if x.dim() == 2:
            # 如果是2D张量 (batch_size, features)，扩展为3D (batch_size, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, but got {x.dim()}D")
        
        # 输入嵌入
        embedded = self.input_embedding(x)
        
        # 添加时间编码
        embedded = self.time_encoding(embedded)
        
        # 应用门控机制，选择重要特征
        gate = self.gating(embedded)
        embedded = embedded * gate
        
        # 多头注意力，捕获特征间依赖
        attn_output, _ = self.attention(embedded, embedded, embedded)
        attn_output = self.layer_norm1(embedded + attn_output)  # 残差连接 + 层归一化
        
        # 前馈网络
        ff_output = self.feed_forward(attn_output)
        ff_output = self.layer_norm2(attn_output + ff_output)  # 残差连接 + 层归一化
        
        # 全局平均池化
        output = ff_output.mean(dim=1)
        
        # 输出层
        output = self.output_layer(output)
        return output

class EnsembleModel(nn.Module):
    """
    集成模型
    融合多个模型的预测结果，提高预测精度
    """
    def __init__(self, input_dim=6):
        """
        初始化集成模型
        
        Args:
            input_dim (int): 输入特征维度
        """
        super(EnsembleModel, self).__init__()
        self.n_features = input_dim  # 记录特征数量
        # 初始化各个基础模型
        self.transformer = TransformerPredictor(input_dim=input_dim)
        self.lstm = LSTMPredictor(input_size=input_dim)
        self.seq2seq = Seq2SeqPredictor(input_size=input_dim)
        self.tft = TFT(input_dim=input_dim)
        
    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, features) 或 (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: 集成预测结果，形状为 (batch_size, 1)
        """
        # 检查输入维度
        if x.dim() == 2:
            # 如果是2D张量 (batch_size, features)，扩展为3D (batch_size, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, but got {x.dim()}D")
        
        # 各个模型的预测
        transformer_pred = self.transformer(x)
        lstm_pred = self.lstm(x)
        seq2seq_pred = self.seq2seq(x)
        tft_pred = self.tft(x)
        
        # 简单平均集成
        ensemble_pred = (transformer_pred + lstm_pred + seq2seq_pred + tft_pred) / 4
        
        return ensemble_pred