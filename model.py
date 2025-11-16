# ==================== 高级扩散模型定义 ====================
import torch
import torch.nn as nn
import math


class SinusoidalEmbedding(nn.Module):
    """正弦位置编码用于时间步"""
    def __init__(self, dim):
        super(SinusoidalEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (batch_size,) 时间步
        Returns:
            embeddings: (batch_size, dim) 位置编码
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, hidden_dim, num_heads=8):
        super(AttentionBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, hidden_dim)
        Returns:
            out: (batch_size, hidden_dim)
        """
        batch_size = x.shape[0]
        
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)
        
        # 计算注意力
        scores = torch.einsum('bnh,bmh->bnm', Q, K) * self.scale
        attn = torch.softmax(scores, dim=-1)
        
        # 应用注意力
        out = torch.einsum('bnm,bmh->bnh', attn, V)
        out = out.contiguous().view(batch_size, -1)
        out = self.output(out)
        
        return out


class ResidualBlock(nn.Module):
    """ResNet50 风格的残差块 - 使用 Bottleneck 架构"""
    def __init__(self, hidden_dim, reduction=4, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.hidden_dim = hidden_dim
        reduced_dim = max(hidden_dim // reduction, 64)  # 压缩维度，最小 64
        
        # Bottleneck 架构：1x1 -> 3x3(通过MLP) -> 1x1
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dim, reduced_dim),
            nn.LayerNorm(reduced_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 中间层 - 包含注意力机制
        self.conv2_attention = AttentionBlock(reduced_dim, num_heads=4)
        self.conv2_mlp = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim * 2, reduced_dim),
            nn.Dropout(dropout)
        )
        
        # 输出层 - 扩展回原维度
        self.conv3 = nn.Sequential(
            nn.Linear(reduced_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 正则化层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(reduced_dim)
        self.norm3 = nn.LayerNorm(reduced_dim)

    def forward(self, x):
        # 保存输入用于残差连接
        identity = x
        
        # 1x1 卷积（降维）
        out = self.norm1(x)
        out = self.conv1(out)
        
        # 3x3 卷积等效（注意力 + MLP）
        out_norm2 = self.norm2(out)
        out_attn = self.conv2_attention(out_norm2)
        out = out + out_attn  # 残差连接
        
        out_norm3 = self.norm3(out)
        out_mlp = self.conv2_mlp(out_norm3)
        out = out + out_mlp  # 残差连接
        
        # 1x1 卷积（升维）
        out = self.conv3(out)
        
        # 主残差连接
        out = out + identity
        
        return out


class DiffusionPredictor(nn.Module):
    """
    高级扩散模型预测器
    使用更强大的条件扩散模型进行二分类预测
    特点:
    - 正弦位置编码用于时间步
    - 多头自注意力机制
    - 残差连接和层归一化
    - 更优的beta调度
    """

    def __init__(self, input_dim=2, seq_len=7, hidden_dim=2048, num_timesteps=1000, dropout=0.3):
        super(DiffusionPredictor, self).__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.input_dim = input_dim

        # 输入编码层 - 使用 Transformer 编码器
        self.input_proj = nn.Linear(input_dim * seq_len, hidden_dim)
        self.input_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        self.input_encoder_norm = nn.LayerNorm(hidden_dim)

        # 正弦时间步编码
        self.time_embedding = SinusoidalEmbedding(hidden_dim)
        
        # 时间步投影层
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 多层去噪网络with注意力和残差连接
        self.denoise_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(8)
        ])
        
        # 去噪网络的输入投影
        self.denoise_input_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.denoise_output_proj = nn.Linear(hidden_dim, hidden_dim)

        # 预测头 - 使用 Transformer 解码器
        self.classifier_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        self.classifier_decoder_norm = nn.LayerNorm(hidden_dim)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 改进的beta调度 - 使用cosine schedule
        self._setup_beta_schedule(num_timesteps)

    def _setup_beta_schedule(self, num_timesteps):
        """设置改进的beta调度"""
        # Cosine schedule
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def forward(self, x, inference_mode=True):
        """
        Args:
            x: (batch_size, input_dim, seq_len)
            inference_mode: 推理模式下直接预测，训练模式下使用扩散过程
        Returns:
            probs: (batch_size, 1) - 预测概率
        """
        batch_size = x.shape[0]

        # 扁平化输入
        x_flat = x.reshape(batch_size, -1)  # (batch_size, input_dim * seq_len)

        # 使用 Transformer 编码器编码输入特征
        x_proj = self.input_proj(x_flat)  # (batch_size, hidden_dim)
        x_proj = x_proj.unsqueeze(1)  # (batch_size, 1, hidden_dim) - Transformer 需要序列输入
        x_encoded = self.input_encoder(x_proj)  # (batch_size, 1, hidden_dim)
        x_encoded = self.input_encoder_norm(x_encoded)  # 正则化
        x_encoded = x_encoded.squeeze(1)  # (batch_size, hidden_dim) - 移除序列维度

        if inference_mode or not self.training:
            # 推理模式：直接输出预测
            # 使用 Transformer 解码器进行最终分类
            decoder_input = x_encoded.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            decoder_output = self.classifier_decoder(decoder_input, decoder_input)
            decoder_output = self.classifier_decoder_norm(decoder_output)
            decoder_output = decoder_output.squeeze(1)  # (batch_size, hidden_dim)
            logits = self.classifier_head(decoder_output)
            return logits
        else:
            # 训练模式：使用扩散过程
            # 随机采样时间步
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)

            # 获取时间编码
            time_emb = self.time_embedding(t.float())
            time_emb = self.time_proj(time_emb)  # (batch_size, hidden_dim)

            # 添加高斯噪声
            noise = torch.randn_like(x_encoded)
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t).view(batch_size, 1)
            sqrt_1_minus_alpha_t = torch.sqrt(1.0 - alpha_t).view(batch_size, 1)

            x_noisy = sqrt_alpha_t * x_encoded + sqrt_1_minus_alpha_t * noise

            # 去噪过程
            x_combined = torch.cat([x_noisy, time_emb], dim=1)
            x_denoised = self.denoise_input_proj(x_combined)
            
            # 通过多层残差块
            for layer in self.denoise_layers:
                x_denoised = layer(x_denoised)
            
            x_denoised = self.denoise_output_proj(x_denoised)
            x_denoised = x_denoised + x_encoded  # 跳跃连接

            # 使用 Transformer 解码器进行最终分类
            decoder_input = x_denoised.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            decoder_output = self.classifier_decoder(decoder_input, decoder_input)
            decoder_output = self.classifier_decoder_norm(decoder_output)
            decoder_output = decoder_output.squeeze(1)  # (batch_size, hidden_dim)
            logits = self.classifier_head(decoder_output)
            return logits
