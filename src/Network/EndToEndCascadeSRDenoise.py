import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_factory import build_sr_model


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _load_state_dict_with_seg_head_compat(module, state_dict, seg_prefix="seg_head."):
    module_keys = set(module.state_dict().keys())
    state_keys = set(state_dict.keys())
    missing_keys = sorted(module_keys - state_keys)
    unexpected_keys = sorted(state_keys - module_keys)

    only_seg_head_mismatch = (
        all(key.startswith(seg_prefix) for key in missing_keys)
        and all(key.startswith(seg_prefix) for key in unexpected_keys)
    )
    if missing_keys or unexpected_keys:
        if only_seg_head_mismatch:
            module.load_state_dict(state_dict, strict=False)
            print(
                "Info: loaded checkpoint with seg_head compatibility "
                f"(missing={missing_keys}, unexpected={unexpected_keys})."
            )
            return
    module.load_state_dict(state_dict)


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
        use_stage1_seg_head=False,
        use_seg_mask_at_inference=False,
        seg_head_bias_init=-4.0,
    ):
        super().__init__()
        if not bool(predict_mag):
            raise ValueError("EndToEndCascadeSRDenoise currently requires predict_mag=True.")

        self.predict_mag = True
        self.res_increase = int(res_increase)
        self.apply_stage2_mask = bool(apply_stage2_mask)
        self.apply_stage2_mask_to_magnitude = bool(apply_stage2_mask_to_magnitude)
        self.use_stage1_seg_head = bool(use_stage1_seg_head)
        self.use_seg_mask_at_inference = bool(use_seg_mask_at_inference)
        self.seg_head_bias_init = float(seg_head_bias_init)

        self.stage1 = build_sr_model(
            model_variant="pre_upsample_attention",
            res_increase=int(res_increase),
            low_resblock=int(low_resblock),
            hi_resblock=int(hi_resblock),
            channel_nr=int(channel_nr),
            predict_mag=True,
            cascade_use_seg_head=self.use_stage1_seg_head,
            cascade_seg_head_bias_init=self.seg_head_bias_init,
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
            _load_state_dict_with_seg_head_compat(self.stage1, _extract_state_dict(checkpoint))
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
        stage1_result = self.stage1(u, v, w, u_mag, v_mag, w_mag)
        if isinstance(stage1_result, tuple):
            stage1_out, seg_map = stage1_result
        else:
            stage1_out, seg_map = stage1_result, None
        stage1_out, mask = self._align_stage1_and_mask(stage1_out, mask)
        if seg_map is not None:
            seg_map = seg_map[
                :,
                :,
                : stage1_out.shape[2],
                : stage1_out.shape[3],
                : stage1_out.shape[4],
            ]

        sr_u = stage1_out[:, 0:1]
        sr_v = stage1_out[:, 1:2]
        sr_w = stage1_out[:, 2:3]
        sr_mag = stage1_out[:, 3:4].clamp_(0.0, 1.0)

        selected_mask = None
        if self.apply_stage2_mask:
            if mask is not None:
                selected_mask = mask
            elif self.use_seg_mask_at_inference and seg_map is not None:
                selected_mask = (torch.sigmoid(seg_map[:, 0]) > 0.5).float()

        if selected_mask is not None:
            sr_u = sr_u * selected_mask[:, None]
            sr_v = sr_v * selected_mask[:, None]
            sr_w = sr_w * selected_mask[:, None]
            if self.apply_stage2_mask_to_magnitude:
                sr_mag = sr_mag * selected_mask[:, None]

        stage2_out = self.stage2(sr_u, sr_v, sr_w, sr_mag, sr_mag, sr_mag)
        return stage2_out, seg_map
