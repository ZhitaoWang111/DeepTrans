import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 特征提取模块
class FeatureExtractor(nn.Module):
    def __init__(self, d_model=64):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        ks = [3, 3, 3,    3, 3,   3, 3]
        ps = [0, 0, 0,    0, 0,   0, 0]
        ss = [2, 2, 2,    2, 2,   2, 1]
        nm = [8, 16, 64,  64, 64, 64, 64]
    
        cnn = nn.Sequential()
        def convRelu(i, cnn, batchNormalization=False):
            nIn = 4 if i == 0 else nm[i - 1]
            if i == 3: nIn = 64
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                            nn.Conv2d(nIn, nOut, (ks[i],1), (ss[i],1), (ps[i],0)))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0,cnn)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2,1), (2,1)))
        convRelu(1,cnn)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2,1), (2,1)))
        convRelu(2,cnn)
        cnn.add_module('pooling{0}'.format(2),
                        nn.MaxPool2d((2, 1), (2, 1), (0, 0)))
        self.conv = cnn

    def forward(self, x):
        conv = self.conv(x)  
        conv = conv.squeeze(2)  # (batch, d_model, 10)
        conv = conv.permute(0, 2, 1)
        conv = conv.reshape(-1, 64)  # (batch*10, d_model)
        return conv

# 2. 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(output)

# 3. Transformer Encoder 层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, positional_encoding):
        # 仅对 Query 和 Key 加入位置编码
        q = x + positional_encoding
        k = x + positional_encoding
        v = x  # Value 不加位置编码
        
        # 多头注意力
        attn_output = self.self_attn(q, k, v)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈神经网络
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        return x

# 4. Transformer 时间建模模块
class Transformer(nn.Module):
    def __init__(self, d_model=64, num_heads=8, num_layers=4, d_ff=256, dropout=0.1):
        super(Transformer, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, 10, d_model))
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (batch, 10, d_model)
        for layer in self.layers:
            x = layer(x, self.positional_encoding)
        return x

# 5. 回归预测模块
class RegressionHead(nn.Module):
    def __init__(self, d_model=64, soh_size=10, rul_size=1):
        super(RegressionHead, self).__init__()
        self.fc1 = nn.Linear(10 * d_model, 256)
        self.fc_soh = nn.Linear(256, soh_size)
        self.fc_rul = nn.Linear(soh_size, rul_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, 10, d_model)
        x = x.view(x.size(0), -1)  # (batch, 10 * d_model)
        x = self.relu(self.fc1(x))  # (batch, 256)
        soh = self.fc_soh(x)  # (batch, 10)
        rul = self.fc_rul(soh)
        return rul, soh

# 6. 完整模型
class BatteryLifeTransformer(nn.Module):
    def __init__(self, d_model=64, num_heads=8, num_layers=4, d_ff=256, dropout=0.1):
        super(BatteryLifeTransformer, self).__init__()
        self.feature_extractor = FeatureExtractor(d_model)
        self.transformer_time_model = Transformer(d_model, num_heads, num_layers, d_ff, dropout)
        self.regression_head = RegressionHead(d_model)

    def forward(self, x):
        # x: (batch, 4, 100, 10)
        batch_size, num_features, seq_len, num_cycles = x.size()
        x = self.feature_extractor(x)  # (batch*10, d_model)
        x = x.reshape(batch_size, num_cycles, -1)  # (batch, 10, d_model)
        x = self.transformer_time_model(x)  # (batch, 10, d_model)
        x = self.regression_head(x)  # (batch, 11)
        return x

# 7. 示例代码
if __name__ == "__main__":
    # 假设输入维度为 (batch, 4, 100, 10)
    batch_size = 32
    input_data = torch.randn(batch_size, 4, 100, 10)  # (batch, 4, 100, 10)

    # 初始化模型
    model = BatteryLifeTransformer()

    # 前向传播
    start_time = time.time()
    soh, rul = model(input_data)  # (batch, 11)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Forward pass took {elapsed_time:.6f} seconds")

    print(rul.shape)  # 输出: torch.Size([32, 11])
    print(soh.shape)