import torch
import torch.nn as nn
import torch.nn.functional as F

from .SR4DFlowNet import Conv3dBlock, ResnetBlock, Upsample3D
from .attention_blocks import CBAM3D


class SR4DFlowNetPreUpsampleAttention(nn.Module):
    """
    Baseline 4DFlowNet plus a single local-attention block right before upsampling.
    """

    def __init__(
        self,
        res_increase,
        low_resblock=8,
        hi_resblock=4,
        channel_nr=64,
        predict_mag=False,
        attention_reduction=8,
        use_seg_head=False,
        seg_head_bias_init=-4.0,
    ):
        super().__init__()
        self.res_increase = res_increase
        self.channel_nr = channel_nr
        self.predict_mag = bool(predict_mag)
        self.use_seg_head = bool(use_seg_head)
        self.seg_head_bias_init = float(seg_head_bias_init)
        self.last_attention_map_lr = None
        self.last_attention_map = None

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

        self.low_res_blocks = nn.ModuleList(
            [ResnetBlock(channel_nr=channel_nr, scale=1.0, pad="SYMMETRIC") for _ in range(low_resblock)]
        )
        # Single local attention pass right before the low-res to high-res transition.
        self.pre_upsample_attn = CBAM3D(channel_nr, reduction=attention_reduction)
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
        if self.use_seg_head:
            self.seg_head = nn.Sequential(
                nn.Conv3d(channel_nr, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 1, kernel_size=1),
            )
            nn.init.constant_(self.seg_head[-1].bias, self.seg_head_bias_init)
        else:
            self.seg_head = None

    def forward(self, u, v, w, u_mag, v_mag, w_mag):
        # Epsilon avoids undefined gradients at exactly-zero velocity vectors,
        # which become common when masked inputs are used in cascade training.
        speed = torch.sqrt(u**2 + v**2 + w**2 + 1e-12)
        mag = u_mag
        pcmr = mag * speed

        phase = torch.cat([u, v, w], dim=1)
        pc = torch.cat([pcmr, mag, speed], dim=1)

        pc = self.pc_path(pc)
        phase = self.phase_path(phase)

        rb = self.merge_path(torch.cat([phase, pc], dim=1))

        for block in self.low_res_blocks:
            rb = block(rb)

        rb = self.pre_upsample_attn(rb)
        self.last_attention_map_lr = getattr(self.pre_upsample_attn, "last_spatial_attention", None)
        rb = self.upsample(rb)

        for block in self.hi_res_blocks:
            rb = block(rb)

        seg_map = self.seg_head(rb) if self.use_seg_head else None
        outputs = [self.u_path(rb), self.v_path(rb), self.w_path(rb)]
        if self.predict_mag:
            outputs.append(self.mag_path(rb))
        velocity_output = torch.cat(outputs, dim=1)
        if self.last_attention_map_lr is not None:
            self.last_attention_map = F.interpolate(
                self.last_attention_map_lr,
                size=velocity_output.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        else:
            self.last_attention_map = None
        if self.use_seg_head:
            return velocity_output, seg_map
        return velocity_output
