import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomTransformerExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        seq_len=5,
        input_dim=23,
    ):
        super().__init__(observation_space, features_dim=embed_dim)

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # 映射每步状态到嵌入维度
        self.embedding = nn.Linear(self.input_dim, embed_dim)

        # 可学习的位置编码 (1, seq_len, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 池化成固定向量
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, observations):
        """
        observations: shape = (batch_size, seq_len * input_dim)
        """
        B = observations.size(0)
        x = observations.view(B, self.seq_len, self.input_dim)  # (B, seq_len, input_dim)
        x = self.embedding(x)  # (B, seq_len, embed_dim)
        x = x + self.pos_embed[:, :self.seq_len, :]  # 位置编码
        x = self.transformer_encoder(x)  # (B, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (B, embed_dim, seq_len)
        x = self.pool(x).squeeze(-1)  # (B, embed_dim)
        return x
