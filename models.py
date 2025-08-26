import torch
import torch.nn as nn
import math

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=6, d_model=64, nhead=8, num_layers=5, dim_feedforward=64, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.n_features = input_dim  # 记录特征数量
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, src):
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
        
        # 全局平均池化
        output = output.mean(dim=1)
        
        # 输出层
        output = self.output_layer(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=5, dropout=0.1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_features = input_size  # 记录特征数量
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
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
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.1):
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
        # 检查输入维度
        if x.dim() == 2:
            # 如果是2D张量 (batch_size, features)，扩展为3D (batch_size, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, but got {x.dim()}D")
        
        # 编码器
        _, (hidden, cell) = self.encoder(x)
        
        # 解码器
        decoder_output, _ = self.decoder(x, (hidden, cell))
        
        # 取最后一个时间步的输出
        output = decoder_output[:, -1, :]
        
        # 输出层
        output = self.output_layer(output)
        
        return output

class EnsembleModel(nn.Module):
    def __init__(self, input_dim=6):
        super(EnsembleModel, self).__init__()
        self.n_features = input_dim  # 记录特征数量
        self.transformer = TransformerPredictor(input_dim=input_dim)
        self.lstm = LSTMPredictor(input_size=input_dim)
        self.seq2seq = Seq2SeqPredictor(input_size=input_dim)
        
    def forward(self, x):
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
        
        # 简单平均集成
        ensemble_pred = (transformer_pred + lstm_pred + seq2seq_pred) / 3
        
        return ensemble_pred