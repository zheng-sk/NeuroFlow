import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_factory import build_sr_model


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


class EndToEndCascadeSRDenoise(nn.Module):
    """
    End-to-end cascade:
      Stage 1: SR x2 with Pre-Upsample CBAM
      Stage 2: Denoising x1 with Original 4DFlowNet

    The vessel mask is applied between stages so Stage 2 sees the same masked
    distribution family it was designed for, while gradients still propagate
    back through both stages.
    """

    requires_mask_input = True

    def __init__(
        self,
        res_increase,
        low_resblock=8,
        hi_resblock=4,
        channel_nr=64,
        predict_mag=False,
        stage1_checkpoint="",
        stage2_checkpoint="",
        freeze_stage1=False,
        freeze_stage2=False,
        apply_stage2_mask=True,
        apply_stage2_mask_to_magnitude=True,
    ):
        super().__init__()
        if not bool(predict_mag):
            raise ValueError("EndToEndCascadeSRDenoise currently requires predict_mag=True.")

        self.predict_mag = True
        self.res_increase = int(res_increase)
        self.apply_stage2_mask = bool(apply_stage2_mask)
        self.apply_stage2_mask_to_magnitude = bool(apply_stage2_mask_to_magnitude)

        self.stage1 = build_sr_model(
            model_variant="pre_upsample_attention",
            res_increase=int(res_increase),
            low_resblock=int(low_resblock),
            hi_resblock=int(hi_resblock),
            channel_nr=int(channel_nr),
            predict_mag=True,
        )
        self.stage2 = build_sr_model(
            model_variant="original",
            res_increase=1,
            low_resblock=int(low_resblock),
            hi_resblock=int(hi_resblock),
            channel_nr=int(channel_nr),
            predict_mag=True,
        )

        if stage1_checkpoint:
            checkpoint = torch.load(stage1_checkpoint, map_location="cpu")
            self.stage1.load_state_dict(_extract_state_dict(checkpoint))
        if stage2_checkpoint:
            checkpoint = torch.load(stage2_checkpoint, map_location="cpu")
            self.stage2.load_state_dict(_extract_state_dict(checkpoint))

        if freeze_stage1:
            for p in self.stage1.parameters():
                p.requires_grad = False
        if freeze_stage2:
            for p in self.stage2.parameters():
                p.requires_grad = False

    @staticmethod
    def _align_stage1_and_mask(stage1_out: torch.Tensor, mask: torch.Tensor):
        if mask is None:
            return stage1_out, mask
        if tuple(mask.shape[1:]) != tuple(stage1_out.shape[2:]):
            mask = F.interpolate(mask[:, None], size=stage1_out.shape[2:], mode="nearest")[:, 0]
        target_shape = tuple(int(s) for s in stage1_out.shape[2:])
        stage1_out = stage1_out[:, :, : target_shape[0], : target_shape[1], : target_shape[2]]
        mask = mask[:, : target_shape[0], : target_shape[1], : target_shape[2]]
        return stage1_out, mask

    def forward(self, u, v, w, u_mag, v_mag, w_mag, mask=None):
        stage1_out = self.stage1(u, v, w, u_mag, v_mag, w_mag)
        stage1_out, mask = self._align_stage1_and_mask(stage1_out, mask)

        sr_u = stage1_out[:, 0:1]
        sr_v = stage1_out[:, 1:2]
        sr_w = stage1_out[:, 2:3]
        sr_mag = stage1_out[:, 3:4]

        if self.apply_stage2_mask and mask is not None:
            sr_u = sr_u * mask[:, None]
            sr_v = sr_v * mask[:, None]
            sr_w = sr_w * mask[:, None]
            if self.apply_stage2_mask_to_magnitude:
                sr_mag = sr_mag * mask[:, None]

        return self.stage2(sr_u, sr_v, sr_w, sr_mag, sr_mag, sr_mag)
