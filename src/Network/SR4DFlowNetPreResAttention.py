import torch
import torch.nn as nn

from .SR4DFlowNet import Conv3dBlock, ResnetBlock, Upsample3D
from .attention_blocks import CBAM3D


class SR4DFlowNetPreResAttention(nn.Module):
    """
    Baseline 4DFlowNet plus a single local-attention block before the residual stack.
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

        self.pc_path = nn.Sequential(
            Conv3dBlock(3, channel_nr, 3, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
        )
        self.phase_path = nn.Sequential(
            Conv3dBlock(3, channel_nr, 3, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
        )
        self.merge_path = nn.Sequential(
            Conv3dBlock(channel_nr * 2, channel_nr, 1, "SYMMETRIC", "relu"),
            Conv3dBlock(channel_nr, channel_nr, 3, "SYMMETRIC", "relu"),
        )
        # Single local attention pass on fused features before any residual processing.
        self.pre_res_attn = CBAM3D(channel_nr, reduction=attention_reduction)

        self.low_res_blocks = nn.ModuleList(
            [ResnetBlock(channel_nr=channel_nr, scale=1.0, pad="SYMMETRIC") for _ in range(low_resblock)]
        )
        self.upsample = Upsample3D(res_increase)
        self.hi_res_blocks = nn.ModuleList(
            [ResnetBlock(channel_nr=channel_nr, scale=1.0, pad="SYMMETRIC") for _ in range(hi_resblock)]
        )

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
        speed = torch.sqrt(u**2 + v**2 + w**2)
        mag = torch.sqrt(u_mag**2 + v_mag**2 + w_mag**2)
        pcmr = mag * speed

        phase = torch.cat([u, v, w], dim=1)
        pc = torch.cat([pcmr, mag, speed], dim=1)

        pc = self.pc_path(pc)
        phase = self.phase_path(phase)

        rb = self.merge_path(torch.cat([phase, pc], dim=1))
        rb = self.pre_res_attn(rb)

        for block in self.low_res_blocks:
            rb = block(rb)

        rb = self.upsample(rb)

        for block in self.hi_res_blocks:
            rb = block(rb)

        outputs = [self.u_path(rb), self.v_path(rb), self.w_path(rb)]
        if self.predict_mag:
            outputs.append(self.mag_path(rb))
        return torch.cat(outputs, dim=1)
