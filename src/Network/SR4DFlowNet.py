import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample3D(nn.Module):
    def __init__(self, res_increase):
        super().__init__()
        self.res_increase = res_increase

    def forward(self, x):
        if self.res_increase == 1:
            return x
        return F.interpolate(
            x,
            scale_factor=self.res_increase,
            mode="trilinear",
            align_corners=False,
        )


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding="SYMMETRIC", activation=None, use_bias=True):
        super().__init__()
        self.padding = padding.upper()
        self.kernel_size = kernel_size

        if self.padding in ("SYMMETRIC", "REFLECT"):
            conv_padding = 0
        elif self.padding == "SAME":
            conv_padding = (kernel_size - 1) // 2
        elif self.padding == "VALID":
            conv_padding = 0
        else:
            raise ValueError(f"Unsupported padding type: {padding}")

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=conv_padding,
            bias=use_bias,
        )
        self.activation = nn.ReLU(inplace=True) if activation == "relu" else None

    def forward(self, x):
        if self.padding in ("SYMMETRIC", "REFLECT"):
            p = (self.kernel_size - 1) // 2
            # PyTorch has no "symmetric" mode; replicate is the closest border-preserving option.
            pad_mode = "replicate" if self.padding == "SYMMETRIC" else "reflect"
            x = F.pad(x, (p, p, p, p, p, p), mode=pad_mode)

        x = self.conv(x)

        if self.activation is not None:
            x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, channel_nr=64, scale=1.0, pad="SAME"):
        super().__init__()
        self.scale = scale
        self.conv1 = Conv3dBlock(channel_nr, channel_nr, kernel_size=3, padding=pad, activation=None, use_bias=False)
        self.conv2 = Conv3dBlock(channel_nr, channel_nr, kernel_size=3, padding=pad, activation=None, use_bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.act(tmp)
        tmp = self.conv2(tmp)
        tmp = x + (tmp * self.scale)
        return self.act(tmp)


class SR4DFlowNet(nn.Module):
    def __init__(self, res_increase, low_resblock=8, hi_resblock=4, channel_nr=64):
        super().__init__()
        self.res_increase = res_increase
        self.channel_nr = channel_nr

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

    def forward(self, u, v, w, u_mag, v_mag, w_mag):
        speed = torch.sqrt(u**2 + v**2 + w**2)
        mag = torch.sqrt(u_mag**2 + v_mag**2 + w_mag**2)
        pcmr = mag * speed

        phase = torch.cat([u, v, w], dim=1)
        pc = torch.cat([pcmr, mag, speed], dim=1)

        pc = self.pc_path(pc)
        phase = self.phase_path(phase)

        concat_layer = torch.cat([phase, pc], dim=1)
        rb = self.merge_path(concat_layer)

        for block in self.low_res_blocks:
            rb = block(rb)

        rb = self.upsample(rb)

        for block in self.hi_res_blocks:
            rb = block(rb)

        u_path = self.u_path(rb)
        v_path = self.v_path(rb)
        w_path = self.w_path(rb)
        return torch.cat([u_path, v_path, w_path], dim=1)
