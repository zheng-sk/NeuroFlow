import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(int(channels) // max(int(reduction), 1), 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global avg/max descriptors summarize each channel's global importance.
        avg_att = self.mlp(self.avg_pool(x))
        max_att = self.mlp(self.max_pool(x))
        # Channel gate in [0,1], then feature-wise reweighting.
        att = self.sigmoid(avg_att + max_att)
        return x * att


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SpatialAttention3D kernel_size must be odd.")
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.last_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collapse channel dimension to two complementary spatial descriptors.
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        # Spatial gate in [0,1] highlights informative voxels.
        att = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        self.last_attention = att
        return x * att


class CBAM3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention3D(channels=channels, reduction=reduction)
        self.spatial_att = SpatialAttention3D(kernel_size=spatial_kernel)
        self.last_spatial_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sequential channel->spatial attention (standard CBAM pattern).
        x = self.channel_att(x)
        x = self.spatial_att(x)
        self.last_spatial_attention = self.spatial_att.last_attention
        return x


class CrossGatedFusion3D(nn.Module):
    """
    Lightweight cross-gating between phase and PC branches.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.phase_to_pc = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.pc_to_phase = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, phase: torch.Tensor, pc: torch.Tensor):
        # Each stream predicts a gate for the other stream.
        pc_gate = self.sigmoid(self.phase_to_pc(phase))
        phase_gate = self.sigmoid(self.pc_to_phase(pc))
        # Residual gating keeps identity path while allowing modulation.
        pc_out = pc + (pc * pc_gate)
        phase_out = phase + (phase * phase_gate)
        return phase_out, pc_out


class TokenMixingAttention3D(nn.Module):
    """
    Global-context token mixing on pooled 3D tokens, then reproject to voxel space.
    Keeps memory bounded by attending over pooled tokens (P^3), not all voxels.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        pool_size: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0.")

        hidden = max(int(channels * float(mlp_ratio)), channels)
        self.pool_size = int(pool_size)
        self.pool = nn.AdaptiveAvgPool3d((self.pool_size, self.pool_size, self.pool_size))
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, channels),
            nn.Dropout(float(dropout)),
        )
        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        # Start near identity for stable fine-tuning/warm start behavior.
        self.alpha = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, sx, sy, sz = x.shape
        # Pool first to keep token count tractable in 3D (P^3 tokens).
        pooled = self.pool(x)  # [B, C, P, P, P]
        tokens = pooled.flatten(2).transpose(1, 2)  # [B, P^3, C]

        # Self-attention over pooled 3D tokens captures long-range dependencies.
        attn_in = self.norm1(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out

        # Feed-forward refinement in token space.
        tokens = tokens + self.ffn(self.norm2(tokens))
        # Reproject token context back to dense voxel grid.
        context = tokens.transpose(1, 2).reshape(b, c, self.pool_size, self.pool_size, self.pool_size)
        context = F.interpolate(context, size=(sx, sy, sz), mode="trilinear", align_corners=False)
        context = self.proj(context)

        # Small learnable residual gain prevents destabilizing early training.
        return x + (torch.sigmoid(self.alpha) * context)


class CrossTokenAttentionFusion3D(nn.Module):
    """
    Bidirectional cross-attention between two 3D feature streams.
    Queries come from one stream and keys/values from the other stream.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        pool_size: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0.")

        hidden = max(int(channels * float(mlp_ratio)), channels)
        self.pool_size = int(pool_size)
        self.pool = nn.AdaptiveAvgPool3d((self.pool_size, self.pool_size, self.pool_size))

        self.phase_q_norm = nn.LayerNorm(channels)
        self.phase_kv_norm = nn.LayerNorm(channels)
        self.pc_q_norm = nn.LayerNorm(channels)
        self.pc_kv_norm = nn.LayerNorm(channels)

        self.phase_cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.pc_cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )

        self.phase_ffn_norm = nn.LayerNorm(channels)
        self.pc_ffn_norm = nn.LayerNorm(channels)
        self.phase_ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, channels),
            nn.Dropout(float(dropout)),
        )
        self.pc_ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, channels),
            nn.Dropout(float(dropout)),
        )

        self.phase_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.pc_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        # Residual gains start small so new cross-attention does not dominate at init.
        self.alpha_phase = nn.Parameter(torch.tensor(-4.0))
        self.alpha_pc = nn.Parameter(torch.tensor(-4.0))

    def forward(self, phase: torch.Tensor, pc: torch.Tensor):
        if phase.shape != pc.shape:
            raise ValueError(f"phase and pc must share shape, got {tuple(phase.shape)} vs {tuple(pc.shape)}")

        b, c, sx, sy, sz = phase.shape
        phase_pooled = self.pool(phase)  # [B, C, P, P, P]
        pc_pooled = self.pool(pc)  # [B, C, P, P, P]

        phase_tokens = phase_pooled.flatten(2).transpose(1, 2)  # [B, P^3, C]
        pc_tokens = pc_pooled.flatten(2).transpose(1, 2)  # [B, P^3, C]

        # phase <- attends to pc
        phase_q = self.phase_q_norm(phase_tokens)
        phase_kv = self.phase_kv_norm(pc_tokens)
        phase_cross, _ = self.phase_cross_attn(phase_q, phase_kv, phase_kv, need_weights=False)
        phase_tokens = phase_tokens + phase_cross
        phase_tokens = phase_tokens + self.phase_ffn(self.phase_ffn_norm(phase_tokens))

        # pc <- attends to phase
        pc_q = self.pc_q_norm(pc_tokens)
        pc_kv = self.pc_kv_norm(phase_tokens)
        pc_cross, _ = self.pc_cross_attn(pc_q, pc_kv, pc_kv, need_weights=False)
        pc_tokens = pc_tokens + pc_cross
        pc_tokens = pc_tokens + self.pc_ffn(self.pc_ffn_norm(pc_tokens))

        phase_ctx = phase_tokens.transpose(1, 2).reshape(b, c, self.pool_size, self.pool_size, self.pool_size)
        pc_ctx = pc_tokens.transpose(1, 2).reshape(b, c, self.pool_size, self.pool_size, self.pool_size)

        phase_ctx = F.interpolate(phase_ctx, size=(sx, sy, sz), mode="trilinear", align_corners=False)
        pc_ctx = F.interpolate(pc_ctx, size=(sx, sy, sz), mode="trilinear", align_corners=False)

        phase_ctx = self.phase_proj(phase_ctx)
        pc_ctx = self.pc_proj(pc_ctx)

        phase_out = phase + (torch.sigmoid(self.alpha_phase) * phase_ctx)
        pc_out = pc + (torch.sigmoid(self.alpha_pc) * pc_ctx)
        return phase_out, pc_out
