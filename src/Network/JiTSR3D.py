import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rounded_hidden_size(channel_nr: int) -> int:
    base = max(int(channel_nr) * 4, 192)
    return ((base + 11) // 12) * 12


def _build_1d_sincos_embedding(length: int, dim: int, device, dtype) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros((length, 0), device=device, dtype=dtype)
    if dim % 2 != 0:
        raise ValueError(f"Sin-cos embedding requires an even dim, got {dim}.")

    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(dim // 2, device=device, dtype=torch.float32) / max(dim // 2, 1)
    ).unsqueeze(0)
    angles = positions * freqs
    emb = torch.cat((angles.sin(), angles.cos()), dim=1)
    return emb.to(dtype=dtype)


def build_3d_sincos_pos_embed(grid_size: tuple[int, int, int], dim: int, device, dtype) -> torch.Tensor:
    if dim % 6 != 0:
        raise ValueError(f"3D sin-cos embedding requires dim divisible by 6, got {dim}.")

    grid_d, grid_h, grid_w = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    axis_dim = dim // 3

    emb_d = _build_1d_sincos_embedding(grid_d, axis_dim, device=device, dtype=dtype)
    emb_h = _build_1d_sincos_embedding(grid_h, axis_dim, device=device, dtype=dtype)
    emb_w = _build_1d_sincos_embedding(grid_w, axis_dim, device=device, dtype=dtype)

    pos_d = emb_d[:, None, None, :].expand(grid_d, grid_h, grid_w, axis_dim)
    pos_h = emb_h[None, :, None, :].expand(grid_d, grid_h, grid_w, axis_dim)
    pos_w = emb_w[None, None, :, :].expand(grid_d, grid_h, grid_w, axis_dim)
    pos = torch.cat((pos_d, pos_h, pos_w), dim=-1)
    return pos.reshape(1, grid_d * grid_h * grid_w, dim)


class ConditionProjector(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3, 4))
        rms = torch.sqrt((x.square()).mean(dim=(2, 3, 4)) + 1e-6)
        stats = torch.cat((mean, rms), dim=1)
        return self.net(stats)


class AdaLNZero(nn.Module):
    def __init__(self, hidden_size: int, n_params: int = 6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, n_params * hidden_size, bias=True),
        )
        self.proj[-1]._skip_jit_init = True
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        params = self.proj(cond).unsqueeze(1)
        chunks = params.chunk(6, dim=-1)
        x_norm = self.norm(x)
        return x_norm, chunks


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}.")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn_out = attn @ v

        out = attn_out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj(out)


class JiTBlock3D(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ada_ln = AdaLNZero(hidden_size, n_params=6)
        self.attn = SelfAttention(hidden_size, num_heads)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_norm, (shift_a, scale_a, gate_a, shift_m, scale_m, gate_m) = self.ada_ln(x, cond)

        attn_in = x_norm * (1 + scale_a) + shift_a
        x = x + gate_a.tanh() * self.attn(attn_in)

        mlp_norm = self.ada_ln.norm(x)
        mlp_in = mlp_norm * (1 + scale_m) + shift_m
        x = x + gate_m.tanh() * self.mlp(mlp_in)
        return x


class BottleneckPatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, hidden_size: int, bottleneck: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Sequential(
            nn.Conv3d(
                in_channels,
                bottleneck,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=False,
            ),
            nn.Conv3d(bottleneck, hidden_size, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x: torch.Tensor):
        out = self.proj(x)
        grid_size = (int(out.shape[2]), int(out.shape[3]), int(out.shape[4]))
        tokens = out.flatten(2).transpose(1, 2)
        return tokens, grid_size


class FinalLayer3D(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        patch_dim = self.patch_size ** 3 * self.out_channels

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        self.proj[-1]._skip_jit_init = True
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

        self.linear = nn.Linear(hidden_size, patch_dim)
        self.linear._skip_jit_init = True
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        shift, scale = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        x = self.linear(x)

        batch_size = x.shape[0]
        grid_d, grid_h, grid_w = grid_size
        patch = self.patch_size
        channels = self.out_channels

        x = x.reshape(batch_size, grid_d, grid_h, grid_w, patch, patch, patch, channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(batch_size, channels, grid_d * patch, grid_h * patch, grid_w * patch)
        return x


class JiTSR3D(nn.Module):
    """
    Transformer 3D variant compatible with the current SR pipeline.

    Key adaptations versus image-space JiT:
    - works on 3D volumes
    - keeps the current 6-input / 3-or-4-output interface
    - performs LR -> HR super-resolution by trilinear upsampling followed by
      patch-token processing on the HR lattice
    - predicts a residual over trilinear velocity upsampling
    """

    def __init__(
        self,
        res_increase,
        low_resblock=8,
        hi_resblock=4,
        channel_nr=64,
        predict_mag=False,
    ):
        super().__init__()
        self.res_increase = int(res_increase)
        self.low_resblock = int(low_resblock)
        self.hi_resblock = int(hi_resblock)
        self.channel_nr = int(channel_nr)
        self.predict_mag = bool(predict_mag)

        self.in_channels = 6
        self.out_channels = 4 if self.predict_mag else 3
        self.hidden_size = _rounded_hidden_size(self.channel_nr)
        self.depth = max(self.low_resblock + self.hi_resblock, 4)
        self.num_heads = 6
        self.mlp_ratio = 4.0
        self.token_patch_size = max(2, self.res_increase * 2)
        self.bottleneck = max(self.channel_nr * 2, 64)

        self.condition_projector = ConditionProjector(self.in_channels, self.hidden_size)
        self.patch_embed = BottleneckPatchEmbed3D(
            in_channels=self.in_channels,
            patch_size=self.token_patch_size,
            hidden_size=self.hidden_size,
            bottleneck=self.bottleneck,
        )
        self.blocks = nn.ModuleList(
            [JiTBlock3D(self.hidden_size, self.num_heads, self.mlp_ratio) for _ in range(self.depth)]
        )
        self.final = FinalLayer3D(self.hidden_size, self.token_patch_size, self.out_channels)

        self._init_weights()

    @staticmethod
    def _pad_to_patch_multiple(x: torch.Tensor, multiple: int):
        depth, height, width = int(x.shape[2]), int(x.shape[3]), int(x.shape[4])
        pad_d = (multiple - (depth % multiple)) % multiple
        pad_h = (multiple - (height % multiple)) % multiple
        pad_w = (multiple - (width % multiple)) % multiple
        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0)
        padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate")
        return padded, (pad_d, pad_h, pad_w)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if getattr(module, "_skip_jit_init", False):
                    continue
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def _upsample_to_hr(self, x: torch.Tensor) -> torch.Tensor:
        if self.res_increase == 1:
            return x
        return F.interpolate(
            x,
            scale_factor=self.res_increase,
            mode="trilinear",
            align_corners=False,
        )

    @staticmethod
    def _vector_norm(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.square().sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, u, v, w, u_mag, v_mag, w_mag):
        lr_input = torch.cat((u, v, w, u_mag, v_mag, w_mag), dim=1)
        hr_input = self._upsample_to_hr(lr_input)

        phase = hr_input[:, 0:3]
        speed = self._vector_norm(phase)
        # Default semantics: one shared magnitude channel, repeated for compatibility.
        mag = hr_input[:, 3:4]
        pcmr = mag * speed
        model_input = torch.cat((phase, pcmr, mag, speed), dim=1)

        cond = self.condition_projector(model_input)
        padded_input, pad_sizes = self._pad_to_patch_multiple(model_input, self.token_patch_size)

        tokens, grid_size = self.patch_embed(padded_input)
        tokens = tokens + build_3d_sincos_pos_embed(
            grid_size=grid_size,
            dim=self.hidden_size,
            device=tokens.device,
            dtype=tokens.dtype,
        )

        for block in self.blocks:
            tokens = block(tokens, cond)

        residual = self.final(tokens, cond, grid_size=grid_size)

        pad_d, pad_h, pad_w = pad_sizes
        if pad_d > 0:
            residual = residual[:, :, :-pad_d, :, :]
        if pad_h > 0:
            residual = residual[:, :, :, :-pad_h, :]
        if pad_w > 0:
            residual = residual[:, :, :, :, :-pad_w]

        outputs = [residual[:, 0:3] + phase]
        if self.predict_mag:
            outputs.append(residual[:, 3:4])
        return torch.cat(outputs, dim=1)
