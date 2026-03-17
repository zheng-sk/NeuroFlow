import torch
import torch.nn as nn

from .SR4DFlowNet import Conv3dBlock, ResnetBlock, Upsample3D
from .attention_blocks import CBAM3D


class SR4DFlowNetPhase1Attention(nn.Module):
    """
    Phase 1 variant:
    - Keeps original dual-branch topology.
    - Adds lightweight CBAM attention in branch features, fusion, and residual stages.
    """

    def __init__(
        self,
        res_increase,
        low_resblock=8,
        hi_resblock=4,
        channel_nr=64,
        predict_mag=False,
        attention_reduction=8,
    ):
        super().__init__()
        self.res_increase = res_increase
        self.channel_nr = channel_nr
        self.predict_mag = bool(predict_mag)

        # Dual-stream feature extraction (same topology as baseline):
        # - phase_path keeps signed velocity cues
        # - pc_path keeps magnitude/speed context
        self.pc_path = nn.Sequential(
            Conv3dBlock(3, channel_nr, 3, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
        )
        self.phase_path = nn.Sequential(
            Conv3dBlock(3, channel_nr, 3, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
        )
        # Branch-wise local attention to suppress background-like activations early.
        self.pc_attn = CBAM3D(channel_nr, reduction=attention_reduction)
        self.phase_attn = CBAM3D(channel_nr, reduction=attention_reduction)

        # Merge both streams and immediately reweight merged features.
        self.merge_path = nn.Sequential(
            Conv3dBlock(channel_nr * 2, channel_nr, 1, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
        )
        self.merge_attn = CBAM3D(channel_nr, reduction=attention_reduction)

        self.low_res_blocks = nn.ModuleList(
            [ResnetBlock(channel_nr=channel_nr, scale=1.0, pad="SYMMETRIC") for _ in range(low_resblock)]
        )
        # Attention after each low-resolution residual block:
        # keeps local vessel patterns while filtering spurious textures.
        self.low_res_attn = nn.ModuleList(
            [CBAM3D(channel_nr, reduction=attention_reduction) for _ in range(low_resblock)]
        )
        self.upsample = Upsample3D(res_increase)
        self.hi_res_blocks = nn.ModuleList(
            [ResnetBlock(channel_nr=channel_nr, scale=1.0, pad="SYMMETRIC") for _ in range(hi_resblock)]
        )
        # Attention after each high-resolution residual block:
        # helps refine fine-grained lumen boundaries after upsampling.
        self.hi_res_attn = nn.ModuleList(
            [CBAM3D(channel_nr, reduction=attention_reduction) for _ in range(hi_resblock)]
        )
        # Final feature reweighting before output heads.
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
        # Build physically meaningful helper channels used in the baseline.
        speed = torch.sqrt(u**2 + v**2 + w**2)
        mag = u_mag
        pcmr = mag * speed

        phase = torch.cat([u, v, w], dim=1)
        pc = torch.cat([pcmr, mag, speed], dim=1)

        # Local attention is applied independently per branch before fusion.
        pc = self.pc_attn(self.pc_path(pc))
        phase = self.phase_attn(self.phase_path(phase))

        # Apply attention again after concatenation so the network can
        # prioritize fused vessel-consistent features.
        rb = self.merge_attn(self.merge_path(torch.cat([phase, pc], dim=1)))

        for block, attn in zip(self.low_res_blocks, self.low_res_attn):
            rb = attn(block(rb))

        rb = self.upsample(rb)

        for block, attn in zip(self.hi_res_blocks, self.hi_res_attn):
            rb = attn(block(rb))

        # Last cleanup pass before per-component prediction heads.
        rb = self.final_attn(rb)

        outputs = [self.u_path(rb), self.v_path(rb), self.w_path(rb)]
        if self.predict_mag:
            outputs.append(self.mag_path(rb))
        return torch.cat(outputs, dim=1)
