# ==================== 高级扩散模型定义 (iTransformer架构) ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DataEmbedding_inverted(nn.Module):
    """iTransformer的倒置嵌入层"""
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_features) -> 需要转置为 (batch_size, num_features, seq_len)
        Returns:
            (batch_size, num_features, d_model)
        """
        # x: [Batch, Seq_len, Num_features] -> [Batch, Num_features, Seq_len]
        x = x.permute(0, 2, 1)
        # [Batch, Num_features, Seq_len] -> [Batch, Num_features, d_model]
        x = self.value_embedding(x)
        return self.dropout(x)


class FullAttention(nn.Module):
    """iTransformer的完整注意力机制"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Args:
            queries: (batch_size, num_tokens, num_heads, head_dim)
            keys: (batch_size, num_tokens, num_heads, head_dim)
            values: (batch_size, num_tokens, num_heads, head_dim)
            attn_mask: attention mask (optional)
        Returns:
            out: (batch_size, num_tokens, num_heads, head_dim)
            attn: attention weights (optional)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                # 创建因果掩码
                attn_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=queries.device), diagonal=1)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, S)

            scores.masked_fill_(attn_mask, float('-inf'))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    """iTransformer的注意力层包装器"""
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """iTransformer编码器层"""
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """iTransformer编码器"""
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class AttentionBlock(nn.Module):
    """自注意力块 (兼容iTransformer)"""
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
            x: (batch_size, num_tokens, hidden_dim) or (batch_size, hidden_dim)
        Returns:
            out: same shape as input
        """
        original_shape = x.shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        batch_size, num_tokens, _ = x.shape
        
        Q = self.query(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力 (batch_size, num_heads, num_tokens, num_tokens)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn, V)  # (batch_size, num_heads, num_tokens, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_tokens, -1)
        out = self.output(out)
        
        if len(original_shape) == 2:
            out = out.squeeze(1)
        
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
        self.use_norm = True  # 使用归一化（Non-stationary Transformer技术）

        # 输入编码层 - 使用 iTransformer 倒置嵌入
        # iTransformer: 将 (batch, seq_len, num_features) -> (batch, num_features, d_model)
        self.enc_embedding = DataEmbedding_inverted(seq_len, hidden_dim, dropout)
        
        # iTransformer 编码器 - 在变量维度上进行自注意力
        # 使用自定义的 Encoder 和 EncoderLayer
        self.input_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
                        hidden_dim, 8
                    ),
                    hidden_dim,
                    hidden_dim * 4,
                    dropout=dropout,
                    activation='gelu'
                ) for _ in range(2)
            ],
            norm_layer=nn.LayerNorm(hidden_dim)
        )
        self.output_attention = False  # 是否输出注意力权重
        
        # iTransformer 投影层：从 d_model 投影回序列长度维度
        self.inverted_projector = nn.Linear(hidden_dim, seq_len)

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

        # 预测头 - 使用 iTransformer 解码器
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
        
        # 最终分类头：汇聚所有变量的信息进行分类
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim * input_dim, hidden_dim // 2),
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
            x: (batch_size, input_dim, seq_len) or (batch_size, seq_len, input_dim)
            inference_mode: 推理模式下直接预测，训练模式下使用扩散过程
        Returns:
            probs: (batch_size, 1) - 预测概率
        """
        batch_size = x.shape[0]
        
        # 确保输入格式为 (batch_size, seq_len, input_dim)
        if x.shape[1] == self.input_dim and x.shape[2] == self.seq_len:
            x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len) -> (batch_size, seq_len, input_dim)
        
        # 归一化处理 (Non-stationary Transformer技术)
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()  # (batch_size, 1, input_dim)
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # iTransformer 编码：(batch_size, seq_len, input_dim) -> (batch_size, input_dim, hidden_dim)
        x_encoded = self.enc_embedding(x)  # 倒置嵌入
        
        # 在变量维度上应用自注意力（使用iTransformer编码器）
        x_encoded, attns = self.input_encoder(x_encoded)  # (batch_size, input_dim, hidden_dim)
        # 注：x_encoded已经过norm_layer处理，不需要额外的正则化
        
        # 保存用于后续处理的编码表示
        x_encoded_flat = x_encoded.reshape(batch_size, -1)  # (batch_size, input_dim * hidden_dim)

        if inference_mode or not self.training:
            # 推理模式：直接输出预测
            # 使用 iTransformer 解码器进行最终分类
            decoder_output = self.classifier_decoder(x_encoded, x_encoded)  # (batch_size, input_dim, hidden_dim)
            decoder_output = self.classifier_decoder_norm(decoder_output)
            decoder_output_flat = decoder_output.reshape(batch_size, -1)  # (batch_size, input_dim * hidden_dim)
            logits = self.classifier_head(decoder_output_flat)
            return logits
        else:
            # 训练模式：使用扩散过程
            # 随机采样时间步
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)

            # 获取时间编码
            time_emb = self.time_embedding(t.float())
            time_emb = self.time_proj(time_emb)  # (batch_size, hidden_dim)

            # 添加高斯噪声到编码表示
            noise = torch.randn_like(x_encoded_flat)
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t).view(batch_size, 1)
            sqrt_1_minus_alpha_t = torch.sqrt(1.0 - alpha_t).view(batch_size, 1)

            x_noisy = sqrt_alpha_t * x_encoded_flat + sqrt_1_minus_alpha_t * noise

            # 去噪过程
            x_combined = torch.cat([x_noisy, time_emb], dim=1)
            x_denoised = self.denoise_input_proj(x_combined)
            
            # 通过多层残差块
            for layer in self.denoise_layers:
                x_denoised = layer(x_denoised)
            
            x_denoised = self.denoise_output_proj(x_denoised)
            x_denoised = x_denoised + x_encoded_flat  # 跳跃连接
            
            # 重塑回 iTransformer 格式
            x_denoised_reshaped = x_denoised.reshape(batch_size, self.input_dim, self.hidden_dim)

            # 使用 iTransformer 解码器进行最终分类
            decoder_output = self.classifier_decoder(x_denoised_reshaped, x_encoded)  # (batch_size, input_dim, hidden_dim)
            decoder_output = self.classifier_decoder_norm(decoder_output)
            decoder_output_flat = decoder_output.reshape(batch_size, -1)  # (batch_size, input_dim * hidden_dim)
            logits = self.classifier_head(decoder_output_flat)
            return logits
