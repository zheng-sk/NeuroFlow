import torch
import torch.nn as nn

from .SR4DFlowNet import Conv3dBlock, ResnetBlock, Upsample3D
from .attention_blocks import CBAM3D, CrossGatedFusion3D, TokenMixingAttention3D


class SR4DFlowNetPhase2Attention(nn.Module):
    """
    Phase 2 variant:
    - Includes Phase 1 local attention.
    - Adds cross-gating between phase/PC branches.
    - Adds pooled-token global context mixing before and after upsampling.
    """

    def __init__(
        self,
        res_increase,
        low_resblock=8,
        hi_resblock=4,
        channel_nr=64,
        predict_mag=False,
        attention_reduction=8,
        attention_heads=4,
        context_pool_size=4,
    ):
        super().__init__()
        self.res_increase = res_increase
        self.channel_nr = channel_nr
        self.predict_mag = bool(predict_mag)

        # Same dual-stream stem as baseline to keep phase and PC cues explicit.
        self.pc_path = nn.Sequential(
            Conv3dBlock(3, channel_nr, 3, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
        )
        self.phase_path = nn.Sequential(
            Conv3dBlock(3, channel_nr, 3, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
        )
        # Per-branch local attention (Phase 1 mechanism).
        self.pc_attn = CBAM3D(channel_nr, reduction=attention_reduction)
        self.phase_attn = CBAM3D(channel_nr, reduction=attention_reduction)
        # Cross-gating lets each branch modulate the other before merge.
        self.cross_gate = CrossGatedFusion3D(channel_nr)

        # Merge + local attention on fused representation.
        self.merge_path = nn.Sequential(
            Conv3dBlock(channel_nr * 2, channel_nr, 1, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
        )
        self.merge_attn = CBAM3D(channel_nr, reduction=attention_reduction)

        self.low_res_blocks = nn.ModuleList(
            [ResnetBlock(channel_nr=channel_nr, scale=1.0, pad="SYMMETRIC") for _ in range(low_resblock)]
        )
        # Local attention remains after each residual block.
        self.low_res_attn = nn.ModuleList(
            [CBAM3D(channel_nr, reduction=attention_reduction) for _ in range(low_resblock)]
        )
        # Global token-mixing at low resolution:
        # captures long-range vessel context with bounded memory cost.
        self.low_context = TokenMixingAttention3D(
            channels=channel_nr,
            num_heads=attention_heads,
            pool_size=context_pool_size,
            mlp_ratio=2.0,
            dropout=0.0,
        )

        self.upsample = Upsample3D(res_increase)
        self.hi_res_blocks = nn.ModuleList(
            [ResnetBlock(channel_nr=channel_nr, scale=1.0, pad="SYMMETRIC") for _ in range(hi_resblock)]
        )
        # Local refinement at HR scale.
        self.hi_res_attn = nn.ModuleList(
            [CBAM3D(channel_nr, reduction=attention_reduction) for _ in range(hi_resblock)]
        )
        # Second global pass after upsampling to harmonize large structures
        # with recovered fine details.
        self.hi_context = TokenMixingAttention3D(
            channels=channel_nr,
            num_heads=attention_heads,
            pool_size=context_pool_size,
            mlp_ratio=2.0,
            dropout=0.0,
        )
        # Final local reweighting before prediction heads.
        self.final_attn = CBAM3D(channel_nr, reduction=attention_reduction)

        self.u_path = nn.Sequential(
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, 1, 3, "SYMMETRIC", None),
        )
        self.v_path = nn.Sequential(
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, 1, 3, "SYMMETRIC", None),
        )
        self.w_path = nn.Sequential(
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, 1, 3, "SYMMETRIC", None),
        )
        if self.predict_mag:
            self.mag_path = nn.Sequential(
                Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
                Conv3dBlock(channel_nr, 1, 3, "SYMMETRIC", None),
            )

    def forward(self, u, v, w, u_mag, v_mag, w_mag):
        # Baseline physical helper channels.
        speed = torch.sqrt(u**2 + v**2 + w**2)
        mag = torch.sqrt(u_mag**2 + v_mag**2 + w_mag**2)
        pcmr = mag * speed

        phase = torch.cat([u, v, w], dim=1)
        pc = torch.cat([pcmr, mag, speed], dim=1)

        # Local branch attention first.
        pc = self.pc_attn(self.pc_path(pc))
        phase = self.phase_attn(self.phase_path(phase))
        # Cross-gating injects complementary cues across branches.
        phase, pc = self.cross_gate(phase, pc)

        rb = self.merge_attn(self.merge_path(torch.cat([phase, pc], dim=1)))

        for block, attn in zip(self.low_res_blocks, self.low_res_attn):
            rb = attn(block(rb))
        # Global low-res context aggregation.
        rb = self.low_context(rb)

        rb = self.upsample(rb)

        for block, attn in zip(self.hi_res_blocks, self.hi_res_attn):
            rb = attn(block(rb))
        # Global HR context aggregation.
        rb = self.hi_context(rb)
        rb = self.final_attn(rb)

        outputs = [self.u_path(rb), self.v_path(rb), self.w_path(rb)]
        if self.predict_mag:
            outputs.append(self.mag_path(rb))
        return torch.cat(outputs, dim=1)
