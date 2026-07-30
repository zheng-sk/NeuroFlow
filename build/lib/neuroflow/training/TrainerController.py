"""
4DFlowNet: Super Resolution ResNet (PyTorch + MONAI data pipeline)
Author: Edward Ferdian, migrated to PyTorch/MONAI
"""
import datetime
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter

from neuroflow.models import h5util, loss_utils, utility
from neuroflow.models.model_factory import build_sr_model, normalize_model_variant


def _load_state_dict_with_seg_head_compat(module, state_dict):
    seg_prefixes = ("seg_head.",)
    module_keys = set(module.state_dict().keys())
    state_keys = set(state_dict.keys())
    missing_keys = sorted(module_keys - state_keys)
    unexpected_keys = sorted(state_keys - module_keys)

    def _is_seg_key(key):
        return any(key.startswith(prefix) for prefix in seg_prefixes)

    if missing_keys or unexpected_keys:
        only_seg_head_mismatch = all(_is_seg_key(key) for key in missing_keys) and all(
            _is_seg_key(key) for key in unexpected_keys
        )
        if only_seg_head_mismatch:
            module.load_state_dict(state_dict, strict=False)
            print(
                "Info: loaded checkpoint with seg_head compatibility "
                f"(missing={missing_keys}, unexpected={unexpected_keys})."
            )
            return
    module.load_state_dict(state_dict)


class TrainerController:
    # constructor
    def __init__(
        self,
        patch_size,
        res_increase,
        initial_learning_rate=1e-4,
        quicksave_enable=True,
        network_name="4DFlowNet",
        low_resblock=8,
        hi_resblock=4,
        model_variant="original",
        predict_mag=False,
        mag_loss_weight=1.0,
        non_fluid_loss_weight=0.1,
        outside_tv_weight=1e-5,
        tb_image_every_n_epochs=10,
        tb_image_axis=2,
        tb_image_batch_index=0,
        accuracy_include_mag=True,
        accuracy_mag_weight=1.0,
        lr_scheduler="reduce_on_plateau",
        lr_reduce_factor=0.5,
        lr_reduce_patience=8,
        lr_min=1e-6,
        early_stopping_patience=20,
        early_stopping_min_delta=0.0,
        overfit_patience=8,
        overfit_min_delta=0.0,
        val_full_volume=False,
        val_sw_patch_size=16,
        val_sw_batch_size=2,
        val_sw_overlap=0.25,
        use_seg_head=False,
        seg_loss_weight=0.0,
        seg_loss_dice_weight=1.0,
        seg_loss_bce_weight=1.0,
        seg_vesselness_loss_weight=0.0,
        seg_vesselness_loss_type="mse",
        attn_loss_weight=0.0,
        attn_loss_type="bce",
        attn_supervision_target="mask",
        seg_head_bias_init=-4.0,
        model_root_dir="",
    ):
        """
        TrainerController constructor.
        Setup model, loss functions and optimizer here.
        """
        self.div_weight = 0  # Weighting for divergence loss
        self.non_fluid_weight = max(float(non_fluid_loss_weight), 0.0)  # Weighting for non fluid region
        self.outside_tv_weight = max(float(outside_tv_weight), 0.0)

        # General params
        self.patch_size = patch_size
        self.res_increase = res_increase
        self.low_resblock = int(low_resblock)
        self.hi_resblock = int(hi_resblock)
        self.channel_nr = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network_src_dir = os.path.dirname(os.path.abspath(__file__))
        self.src_root_dir = os.path.dirname(self.network_src_dir)
        self.repo_root_dir = os.path.dirname(self.src_root_dir)
        if model_root_dir:
            if os.path.isabs(model_root_dir):
                self.model_root_dir = model_root_dir
            else:
                self.model_root_dir = os.path.join(self.repo_root_dir, model_root_dir)
        else:
            self.model_root_dir = os.path.join(self.repo_root_dir, "models")

        # Training params
        self.QUICKSAVE_ENABLED = quicksave_enable
        self.predict_mag = bool(predict_mag)
        self.mag_loss_weight = float(mag_loss_weight)
        self.model_variant = normalize_model_variant(model_variant)
        self.tb_image_every_n_epochs = max(int(tb_image_every_n_epochs), 0)
        self.tb_image_axis = int(tb_image_axis)
        self.tb_image_batch_index = max(int(tb_image_batch_index), 0)
        self.accuracy_include_mag = bool(accuracy_include_mag)
        self.accuracy_mag_weight = max(float(accuracy_mag_weight), 0.0)
        self.lr_scheduler_name = str(lr_scheduler).lower()
        self.lr_reduce_factor = float(lr_reduce_factor)
        self.lr_reduce_patience = int(lr_reduce_patience)
        self.lr_min = float(lr_min)
        self.early_stopping_patience = max(int(early_stopping_patience), 0)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.overfit_patience = max(int(overfit_patience), 0)
        self.overfit_min_delta = float(overfit_min_delta)
        self.val_full_volume = bool(val_full_volume)
        self.val_sw_patch_size = max(int(val_sw_patch_size), 1)
        self.val_sw_batch_size = max(int(val_sw_batch_size), 1)
        self.val_sw_overlap = float(val_sw_overlap)
        self.use_seg_head = bool(use_seg_head)
        self.seg_loss_weight = max(float(seg_loss_weight), 0.0)
        self.seg_loss_dice_weight = max(float(seg_loss_dice_weight), 0.0)
        self.seg_loss_bce_weight = max(float(seg_loss_bce_weight), 0.0)
        self.seg_vesselness_loss_weight = max(float(seg_vesselness_loss_weight), 0.0)
        self.seg_vesselness_loss_type = str(seg_vesselness_loss_type).strip().lower()
        self.attn_loss_weight = max(float(attn_loss_weight), 0.0)
        self.attn_loss_type = str(attn_loss_type).strip().lower()
        self.attn_supervision_target = str(attn_supervision_target).strip().lower()
        self.seg_head_bias_init = float(seg_head_bias_init)
        self.model_outputs_seg_map = self.use_seg_head and self.model_variant == "pre_upsample_attention"
        self.model_outputs_attn_map = (
            self.attn_loss_weight > 0.0 and self.model_variant == "pre_upsample_attention"
        )
        if self.seg_vesselness_loss_type not in {"bce", "mse", "l1", "smoothl1"}:
            raise ValueError(
                f"Unsupported seg_vesselness_loss_type={self.seg_vesselness_loss_type!r}. "
                "Use one of: bce, mse, l1, smoothl1."
            )
        if self.attn_loss_type not in {"bce", "mse", "l1", "smoothl1"}:
            raise ValueError(
                f"Unsupported attn_loss_type={self.attn_loss_type!r}. "
                "Use one of: bce, mse, l1, smoothl1."
            )
        if self.attn_supervision_target not in {"mask", "vesselness"}:
            raise ValueError(
                f"Unsupported attn_supervision_target={self.attn_supervision_target!r}. "
                "Use one of: mask, vesselness."
            )
        self.uses_vesselness_target = self.seg_vesselness_loss_weight > 0.0 or (
            self.attn_loss_weight > 0.0 and self.attn_supervision_target == "vesselness"
        )
        if not (0.0 <= self.val_sw_overlap < 1.0):
            raise ValueError("--val-sw-overlap must be in [0,1).")

        # Network
        self.network_name = network_name
        self.model = build_sr_model(
            model_variant=self.model_variant,
            res_increase=res_increase,
            low_resblock=low_resblock,
            hi_resblock=hi_resblock,
            channel_nr=self.channel_nr,
            predict_mag=self.predict_mag,
            use_seg_head=self.use_seg_head,
            seg_head_bias_init=self.seg_head_bias_init,
        ).to(self.device)
        self.metric_keys = [
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
            "train_mse",
            "val_mse",
            "train_div",
            "val_div",
            "l2_reg_loss",
        ]
        if self.predict_mag:
            self.metric_keys.extend(["train_mag_mse", "val_mag_mse"])
            if self.accuracy_include_mag:
                self.metric_keys.extend(["train_mag_accuracy", "val_mag_accuracy"])
        if self.seg_loss_weight > 0.0:
            self.metric_keys.extend(["train_seg_dice_loss", "val_seg_dice_loss"])
            if self.seg_loss_bce_weight > 0.0:
                self.metric_keys.extend(["train_seg_bce_loss", "val_seg_bce_loss"])
        if self.seg_vesselness_loss_weight > 0.0:
            self.metric_keys.extend(["train_seg_vesselness_loss", "val_seg_vesselness_loss"])
        if self.attn_loss_weight > 0.0:
            self.metric_keys.extend(["train_attn_loss", "val_attn_loss"])
        self.accuracy_metric = "val_loss"

        print(f"Divergence loss2 * {self.div_weight}")
        print(f"Non-fluid loss weight: {self.non_fluid_weight}")
        print(f"Outside TV weight: {self.outside_tv_weight}")
        print(f"Accuracy metric: {self.accuracy_metric}")
        print(f"Predict magnitude head: {self.predict_mag}")
        print(f"Model variant: {self.model_variant}")
        if self.predict_mag:
            print(f"Magnitude loss weight: {self.mag_loss_weight}")
        if self.use_seg_head:
            print("Stage-1 segmentation head: enabled")
        if self.seg_loss_weight > 0.0:
            print(
                f"Segmentation loss: weight={self.seg_loss_weight}, "
                f"dice_weight={self.seg_loss_dice_weight}, "
                f"bce_weight={self.seg_loss_bce_weight}"
            )
        if self.seg_vesselness_loss_weight > 0.0:
            print(
                f"Seg-head vesselness loss: weight={self.seg_vesselness_loss_weight}, "
                f"type={self.seg_vesselness_loss_type}"
            )
        if self.attn_loss_weight > 0.0:
            print(
                f"Attention supervision loss: weight={self.attn_loss_weight}, "
                f"type={self.attn_loss_type}, target={self.attn_supervision_target}"
            )
        if self.use_seg_head:
            print(f"Seg-head bias init: {self.seg_head_bias_init}")
        if self.seg_loss_weight > 0.0 and not self.model_outputs_seg_map:
            print("Warning: seg_loss_weight > 0 but the selected model configuration does not emit seg_map.")
        if self.seg_vesselness_loss_weight > 0.0 and not self.model_outputs_seg_map:
            print(
                "Warning: seg_vesselness_loss_weight > 0 but the selected model configuration does not emit seg_map."
            )
        if self.attn_loss_weight > 0.0 and not self.model_outputs_attn_map:
            print("Warning: attn_loss_weight > 0 but the selected model configuration does not expose attention maps.")
        if self.tb_image_every_n_epochs > 0:
            print(
                f"TensorBoard validation recon images every {self.tb_image_every_n_epochs} epochs "
                f"(axis={self.tb_image_axis}, batch_index={self.tb_image_batch_index})"
            )
        if self.predict_mag and self.accuracy_include_mag:
            print(f"Accuracy metric includes magnitude error (weight={self.accuracy_mag_weight})")
        print(f"LR scheduler: {self.lr_scheduler_name}")
        if self.val_full_volume:
            print(
                f"Validation mode: full-volume sliding window "
                f"(roi={self.val_sw_patch_size}, sw_batch={self.val_sw_batch_size}, overlap={self.val_sw_overlap})"
            )
        if self.early_stopping_patience > 0:
            print(
                f"Early stopping on val_loss: patience={self.early_stopping_patience}, "
                f"min_delta={self.early_stopping_min_delta}"
            )
        if self.overfit_patience > 0:
            print(f"Overfitting stop: patience={self.overfit_patience}, min_delta={self.overfit_min_delta}")
        print(f"Using device: {self.device}")

        # Learning rate and optimizer (weight_decay replicates L2 regularization)
        self.learning_rate = initial_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-7)
        self.scheduler = None
        if self.lr_scheduler_name == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.lr_reduce_factor,
                patience=self.lr_reduce_patience,
                min_lr=self.lr_min,
            )
        self._reset_metric_storage()
        self._shape_align_warned = False

    def _reset_metric_storage(self):
        self.metric_sums = {k: 0.0 for k in self.metric_keys}
        self.metric_counts = {k: 0 for k in self.metric_keys}

    def _update_metric(self, key, value, n=1):
        self.metric_sums[key] += float(value) * n
        self.metric_counts[key] += n

    def _log_metric(self, metric_name, value, phase, n=1):
        key = f"{phase}_{metric_name}"
        if key in self.metric_sums:
            self._update_metric(key, value, n)

    def _metric_value(self, key):
        if self.metric_counts[key] == 0:
            return 0.0
        return self.metric_sums[key] / self.metric_counts[key]

    @staticmethod
    def _crop_front_5d(tensor, target_shape):
        tx, ty, tz = target_shape
        return tensor[:, :, :tx, :ty, :tz]

    @staticmethod
    def _crop_front_4d(tensor, target_shape):
        tx, ty, tz = target_shape
        return tensor[:, :tx, :ty, :tz]

    def _align_spatial_shapes(self, y_true, y_pred, mask):
        """
        Align y_true/y_pred/mask spatial sizes by front-cropping to their common
        minimum size. This is needed when HR dims are not exact multiples of LR
        dims (e.g., 99 vs 100 with res_increase=2 in full-volume validation).
        """
        if y_true.ndim != 5 or y_pred.ndim != 5 or mask.ndim != 4:
            return y_true, y_pred, mask

        target_shape = (
            min(int(y_true.shape[2]), int(y_pred.shape[2]), int(mask.shape[1])),
            min(int(y_true.shape[3]), int(y_pred.shape[3]), int(mask.shape[2])),
            min(int(y_true.shape[4]), int(y_pred.shape[4]), int(mask.shape[3])),
        )
        if min(target_shape) <= 0:
            raise ValueError(
                f"Invalid tensor shapes for metric alignment: "
                f"y_true={tuple(y_true.shape)}, y_pred={tuple(y_pred.shape)}, mask={tuple(mask.shape)}"
            )

        needs_crop = (
            tuple(y_true.shape[2:5]) != target_shape
            or tuple(y_pred.shape[2:5]) != target_shape
            or tuple(mask.shape[1:4]) != target_shape
        )
        if not needs_crop:
            return y_true, y_pred, mask

        if not self._shape_align_warned:
            print(
                "Warning: aligning prediction/target/mask shapes by front-cropping "
                f"to {target_shape}. Original shapes: "
                f"y_true={tuple(y_true.shape)}, y_pred={tuple(y_pred.shape)}, mask={tuple(mask.shape)}"
            )
            self._shape_align_warned = True

        y_true = self._crop_front_5d(y_true, target_shape)
        y_pred = self._crop_front_5d(y_pred, target_shape)
        mask = self._crop_front_4d(mask, target_shape)
        return y_true, y_pred, mask

    def save_latest_model(self, epoch):
        if epoch > 0 and epoch % 10 == 0:
            checkpoint_path = f"{self.model_path}-latest.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                    "learning_rate": self.learning_rate,
                    "network_name": self.network_name,
                    "model_variant": self.model_variant,
                    "res_increase": self.res_increase,
                    "low_resblock": self.low_resblock,
                    "hi_resblock": self.hi_resblock,
                    "channel_nr": self.channel_nr,
                    "predict_mag": self.predict_mag,
                    "mag_loss_weight": self.mag_loss_weight,
                    "non_fluid_loss_weight": self.non_fluid_weight,
                    "outside_tv_weight": self.outside_tv_weight,
                    "use_seg_head": self.use_seg_head,
                    # legacy alias, kept so older tooling can still read the flag
                    "cascade_use_seg_head": self.use_seg_head,
                    "seg_loss_weight": self.seg_loss_weight,
                    "seg_loss_dice_weight": self.seg_loss_dice_weight,
                    "seg_loss_bce_weight": self.seg_loss_bce_weight,
                    "seg_vesselness_loss_weight": self.seg_vesselness_loss_weight,
                    "seg_vesselness_loss_type": self.seg_vesselness_loss_type,
                    "attn_loss_weight": self.attn_loss_weight,
                    "attn_loss_type": self.attn_loss_type,
                    "attn_supervision_target": self.attn_supervision_target,
                    "seg_head_bias_init": self.seg_head_bias_init,
                    "cascade_seg_head_bias_init": self.seg_head_bias_init,
                },
                checkpoint_path,
            )
            message = f"Saving current model - {time.ctime()}\n"
            print(message)

    def loss_function(self, y_true, y_pred, mask, seg_map=None, vesselness_target=None, attn_map=None):
        """
        Calculate Total Loss function:
        Loss = velocity_MSE + mag_weight * magnitude_MSE + weight * div_loss2
        """
        u, v, w = y_true[:, 0], y_true[:, 1], y_true[:, 2]
        u_pred, v_pred, w_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

        vel_mse_vox = self.calculate_mse(u, v, w, u_pred, v_pred, w_pred)
        vel_mse = self._masked_region_mse(vel_mse_vox, mask, non_fluid_weight=self.non_fluid_weight)

        mag_mse = torch.zeros_like(vel_mse)
        if self.predict_mag:
            if y_true.shape[1] < 4 or y_pred.shape[1] < 4:
                raise ValueError(
                    f"predict_mag=True requires 4 channels in y_true/y_pred, got "
                    f"{y_true.shape[1]} and {y_pred.shape[1]}."
                )
            mag_true = y_true[:, 3]
            mag_pred = y_pred[:, 3]
            mag_mse_vox = (mag_pred - mag_true) ** 2
            mag_mse = self._masked_region_mse(mag_mse_vox, mask, non_fluid_weight=self.non_fluid_weight)

        divergence_loss = torch.zeros_like(vel_mse)
        outside_tv = self._outside_tv_penalty(y_pred, mask)
        total_loss = vel_mse + (self.mag_loss_weight * mag_mse) + divergence_loss + (self.outside_tv_weight * outside_tv)
        seg_dice_loss = None
        seg_bce_loss = None
        seg_vesselness_loss = None
        attn_loss = None
        seg_logits = seg_map[:, 0] if seg_map is not None else None
        seg_prob = torch.sigmoid(seg_logits) if seg_logits is not None else None
        if self.seg_loss_weight > 0.0 and seg_map is not None and mask is not None:
            mask_aligned = self._align_mask_to_shape(mask, seg_logits.shape[-3:]).to(seg_logits.dtype)
            dice = self._dice_loss(seg_prob, mask_aligned)
            bce = F.binary_cross_entropy_with_logits(seg_logits, mask_aligned)
            seg_loss = (self.seg_loss_dice_weight * dice) + (self.seg_loss_bce_weight * bce)
            total_loss = total_loss + (self.seg_loss_weight * seg_loss)
            seg_dice_loss = dice
            seg_bce_loss = bce
        if self.seg_vesselness_loss_weight > 0.0 and seg_prob is not None and vesselness_target is not None:
            vesselness_aligned = self._align_continuous_to_shape(vesselness_target, seg_logits.shape[-3:]).to(
                seg_logits.dtype
            )
            seg_vesselness_loss = self._probability_target_loss(
                seg_prob,
                vesselness_aligned,
                self.seg_vesselness_loss_type,
                logits=seg_logits if self.seg_vesselness_loss_type == "bce" else None,
            )
            total_loss = total_loss + (self.seg_vesselness_loss_weight * seg_vesselness_loss)
        if self.attn_loss_weight > 0.0 and attn_map is not None:
            attn_prob = attn_map[:, 0]
            if self.attn_supervision_target == "vesselness":
                if vesselness_target is not None:
                    attn_target = self._align_continuous_to_shape(vesselness_target, attn_prob.shape[-3:])
                else:
                    attn_target = None
            else:
                attn_target = self._align_mask_to_shape(mask, attn_prob.shape[-3:]) if mask is not None else None

            if attn_target is not None:
                attn_target = attn_target.to(attn_prob.dtype)
                attn_loss = self._probability_target_loss(attn_prob, attn_target, self.attn_loss_type)
                total_loss = total_loss + (self.attn_loss_weight * attn_loss)

        return total_loss, vel_mse, divergence_loss, mag_mse, seg_dice_loss, seg_bce_loss, seg_vesselness_loss, attn_loss

    def accuracy_function(self, y_true, y_pred, mask):
        """
        Calculate relative speed error.
        """
        u, v, w = y_true[:, 0], y_true[:, 1], y_true[:, 2]
        u_pred, v_pred, w_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        vel_rel = loss_utils.calculate_relative_error(u_pred, v_pred, w_pred, u, v, w, mask)

        if self.predict_mag and self.accuracy_include_mag and y_true.shape[1] >= 4 and y_pred.shape[1] >= 4:
            mag_rel = loss_utils.calculate_relative_error_mag(y_pred[:, 3], y_true[:, 3], mask)
            w_mag = self.accuracy_mag_weight
            combined = (vel_rel + (w_mag * mag_rel)) / (1.0 + w_mag)
            return combined, vel_rel, mag_rel

        return vel_rel, vel_rel, None

    @staticmethod
    def calculate_mse(u, v, w, u_pred, v_pred, w_pred):
        """
        Calculate speed magnitude error.
        """
        return (u_pred - u) ** 2 + (v_pred - v) ** 2 + (w_pred - w) ** 2

    def _dice_loss(self, pred, target):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + 1e-6) / (
            pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
        )
        return (1.0 - dice).mean()

    @staticmethod
    def _masked_region_mse(voxel_error, mask, non_fluid_weight: float = 1.0):
        non_fluid_mask = (mask < 0.5).float()
        epsilon = 1.0  # minimum 1 pixel

        fluid_mse = voxel_error * mask
        fluid_mse = fluid_mse.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + epsilon)

        non_fluid_mse = voxel_error * non_fluid_mask
        non_fluid_mse = non_fluid_mse.sum(dim=(1, 2, 3)) / (non_fluid_mask.sum(dim=(1, 2, 3)) + epsilon)

        return fluid_mse + (float(non_fluid_weight) * non_fluid_mse)

    @staticmethod
    def _outside_tv_penalty(pred, mask):
        # pred: [B, C, X, Y, Z], mask: [B, X, Y, Z]
        outside = (mask < 0.5).float()
        if pred.ndim != 5 or outside.ndim != 4:
            return torch.zeros((pred.shape[0],), dtype=pred.dtype, device=pred.device)

        penalties = []
        epsilon = 1.0

        if pred.shape[2] > 1:
            dx = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
            wx = (outside[:, 1:, :, :] * outside[:, :-1, :, :]).unsqueeze(1)
            num = (dx * wx).sum(dim=(1, 2, 3, 4))
            den = (wx.sum(dim=(1, 2, 3, 4)) * pred.shape[1]) + epsilon
            penalties.append(num / den)
        if pred.shape[3] > 1:
            dy = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
            wy = (outside[:, :, 1:, :] * outside[:, :, :-1, :]).unsqueeze(1)
            num = (dy * wy).sum(dim=(1, 2, 3, 4))
            den = (wy.sum(dim=(1, 2, 3, 4)) * pred.shape[1]) + epsilon
            penalties.append(num / den)
        if pred.shape[4] > 1:
            dz = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
            wz = (outside[:, :, :, 1:] * outside[:, :, :, :-1]).unsqueeze(1)
            num = (dz * wz).sum(dim=(1, 2, 3, 4))
            den = (wz.sum(dim=(1, 2, 3, 4)) * pred.shape[1]) + epsilon
            penalties.append(num / den)

        if not penalties:
            return torch.zeros((pred.shape[0],), dtype=pred.dtype, device=pred.device)
        return torch.stack(penalties, dim=0).mean(dim=0)

    @staticmethod
    def _extract_slice_2d(vol_3d, axis, index):
        if axis == 0:
            return vol_3d[index, :, :]
        if axis == 1:
            return vol_3d[:, index, :]
        return vol_3d[:, :, index]

    @staticmethod
    def _resize_2d(img_2d, target_hw, mode="bilinear"):
        t = torch.as_tensor(img_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if mode == "nearest":
            out = F.interpolate(t, size=target_hw, mode="nearest")
        else:
            out = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
        return out.squeeze(0).squeeze(0).cpu().numpy()

    @staticmethod
    def _robust_limits(arrays, signed):
        vals = []
        for a in arrays:
            x = np.asarray(a, dtype=np.float32).ravel()
            x = x[np.isfinite(x)]
            if x.size > 0:
                vals.append(x)
        if not vals:
            return (-1.0, 1.0) if signed else (0.0, 1.0)
        merged = np.concatenate(vals, axis=0)
        if signed:
            vmax = float(np.percentile(np.abs(merged), 99.5))
            vmax = max(vmax, 1e-6)
            return -vmax, vmax
        lo = float(np.percentile(merged, 0.5))
        hi = float(np.percentile(merged, 99.5))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        return lo, hi

    @staticmethod
    def _normalize_signed(img, vabs):
        return np.clip((img / max(float(vabs), 1e-6) + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _normalize_unsigned(img, lo, hi):
        denom = max(float(hi - lo), 1e-6)
        return np.clip((img - lo) / denom, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _tile_images_h(tiles, sep=2):
        h = int(tiles[0].shape[0])
        separator = np.full((h, int(sep)), 0.15, dtype=np.float32)
        out = tiles[0]
        for t in tiles[1:]:
            out = np.concatenate((out, separator, t), axis=1)
        return out.astype(np.float32)

    def _log_validation_reconstructions(self, epoch_nr, u, v, w, u_mag, hires, mask, predictions):
        if self.tb_image_every_n_epochs <= 0:
            return

        if predictions.ndim != 5 or hires.ndim != 5 or mask.ndim != 4:
            return

        b_idx = min(int(self.tb_image_batch_index), int(predictions.shape[0] - 1))
        axis = int(np.clip(self.tb_image_axis, 0, 2))

        mask_np = mask[b_idx].detach().float().cpu().numpy()
        channel_specs = [
            ("u", u[b_idx, 0], predictions[b_idx, 0], hires[b_idx, 0], True),
            ("v", v[b_idx, 0], predictions[b_idx, 1], hires[b_idx, 1], True),
            ("w", w[b_idx, 0], predictions[b_idx, 2], hires[b_idx, 2], True),
        ]
        if self.predict_mag and predictions.shape[1] > 3 and hires.shape[1] > 3:
            channel_specs.append(("mag", u_mag[b_idx, 0], predictions[b_idx, 3], hires[b_idx, 3], False))

        for ch_name, inp_t, pred_t, gt_t, signed in channel_specs:
            inp_3d = inp_t.detach().float().cpu().numpy()
            pred_3d = pred_t.detach().float().cpu().numpy()
            gt_3d = gt_t.detach().float().cpu().numpy()

            gt_n = gt_3d.shape[axis]
            if gt_n <= 0:
                continue
            gt_idx = int(gt_n // 2)
            inp_n = inp_3d.shape[axis]
            if inp_n <= 1 or gt_n <= 1:
                inp_idx = 0
            else:
                inp_idx = int(round(gt_idx * (inp_n - 1) / float(gt_n - 1)))
            mask_idx = min(gt_idx, mask_np.shape[axis] - 1)

            inp_2d = self._extract_slice_2d(inp_3d, axis, inp_idx)
            pred_2d = self._extract_slice_2d(pred_3d, axis, gt_idx)
            gt_2d = self._extract_slice_2d(gt_3d, axis, gt_idx)
            mask_2d = self._extract_slice_2d(mask_np, axis, mask_idx)

            target_hw = gt_2d.shape
            if inp_2d.shape != target_hw:
                inp_2d = self._resize_2d(inp_2d, target_hw, mode="bilinear")
            if mask_2d.shape != target_hw:
                mask_2d = self._resize_2d(mask_2d, target_hw, mode="nearest")
            mask_2d = (mask_2d > 0.5).astype(np.float32)

            err_2d = np.abs(pred_2d - gt_2d).astype(np.float32)
            err_in_mask = err_2d[mask_2d > 0.5]
            if err_in_mask.size == 0:
                err_hi = float(np.percentile(err_2d, 99.5))
                mae_mask = float(np.mean(err_2d))
            else:
                err_hi = float(np.percentile(err_in_mask, 99.5))
                mae_mask = float(np.mean(err_in_mask))
            err_hi = max(err_hi, 1e-6)

            if signed:
                _, vmax = self._robust_limits([inp_2d, pred_2d, gt_2d], signed=True)
                vabs = abs(float(vmax))
                inp_img = self._normalize_signed(inp_2d, vabs)
                pred_img = self._normalize_signed(pred_2d, vabs)
                gt_img = self._normalize_signed(gt_2d, vabs)
            else:
                lo, hi = self._robust_limits([inp_2d, pred_2d, gt_2d], signed=False)
                inp_img = self._normalize_unsigned(inp_2d, lo, hi)
                pred_img = self._normalize_unsigned(pred_2d, lo, hi)
                gt_img = self._normalize_unsigned(gt_2d, lo, hi)

            err_img = np.clip(err_2d / err_hi, 0.0, 1.0).astype(np.float32)
            tiled = self._tile_images_h([inp_img, pred_img, gt_img, err_img, mask_2d], sep=2)

            self.val_writer.add_image(
                f"{self.network_name}/reconstruction/{ch_name}",
                tiled[None, ...],
                global_step=int(epoch_nr),
                dataformats="CHW",
            )
            self.val_writer.add_scalar(
                f"{self.network_name}/reconstruction_mae/{ch_name}",
                mae_mask,
                int(epoch_nr),
            )

        self.val_writer.flush()

    def init_model_dir(self):
        """
        Create model directory to save the weights with a [network_name]_[datetime] format.
        Also prepare logfile and tensorboard summary within the directory.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.unique_model_name = f"{self.network_name}_{timestamp}"

        self.model_dir = os.path.join(self.model_root_dir, self.unique_model_name)
        self.model_path = os.path.join(self.model_dir, self.network_name)

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        self._prepare_logfile_and_summary()

    def _prepare_logfile_and_summary(self):
        """
        Prepare csv logfile to keep track of the loss and TensorBoard summaries.
        """
        self.train_writer = SummaryWriter(log_dir=f"{self.model_dir}/tensorboard/train")
        self.val_writer = SummaryWriter(log_dir=f"{self.model_dir}/tensorboard/validate")

        # Prepare log file
        self.logfile = f"{self.model_dir}/loss.csv"

        utility.log_to_file(self.logfile, f"Network: {self.network_name}\n")
        utility.log_to_file(self.logfile, f"Model variant: {self.model_variant}\n")
        utility.log_to_file(self.logfile, f"Initial learning rate: {self.learning_rate}\n")
        utility.log_to_file(self.logfile, f"Accuracy metric: {self.accuracy_metric}\n")
        utility.log_to_file(
            self.logfile,
            f"Accuracy include magnitude: {self.predict_mag and self.accuracy_include_mag} "
            f"(weight={self.accuracy_mag_weight})\n",
        )
        utility.log_to_file(self.logfile, f"Divergence weight: {self.div_weight}\n")
        utility.log_to_file(self.logfile, f"Non-fluid loss weight: {self.non_fluid_weight}\n")
        utility.log_to_file(self.logfile, f"Outside TV weight: {self.outside_tv_weight}\n")
        utility.log_to_file(self.logfile, f"Use seg head: {self.use_seg_head}\n")
        utility.log_to_file(
            self.logfile,
            f"Seg loss weight: {self.seg_loss_weight}, seg loss dice weight: {self.seg_loss_dice_weight}, "
            f"seg loss bce weight: {self.seg_loss_bce_weight}\n",
        )
        utility.log_to_file(
            self.logfile,
            f"Seg vesselness loss weight: {self.seg_vesselness_loss_weight}, "
            f"seg vesselness loss type: {self.seg_vesselness_loss_type}\n",
        )
        utility.log_to_file(
            self.logfile,
            f"Attention loss weight: {self.attn_loss_weight}, "
            f"attention loss type: {self.attn_loss_type}, "
            f"attention supervision target: {self.attn_supervision_target}\n",
        )
        utility.log_to_file(self.logfile, f"Seg-head bias init: {self.seg_head_bias_init}\n")
        utility.log_to_file(
            self.logfile,
            f"LR scheduler: {self.lr_scheduler_name} "
            f"(factor={self.lr_reduce_factor}, patience={self.lr_reduce_patience}, min_lr={self.lr_min})\n",
        )
        utility.log_to_file(
            self.logfile,
            f"Early stopping patience: {self.early_stopping_patience}, "
            f"min_delta: {self.early_stopping_min_delta}, "
            f"overfit_patience: {self.overfit_patience}, "
            f"overfit_min_delta: {self.overfit_min_delta}\n",
        )

        stat_names = ",".join(self.metric_keys)
        utility.log_to_file(
            self.logfile,
            f"epoch, {stat_names}, learning rate, elapsed (sec), best_model, benchmark_err, benchmark_rel_err, benchmark_mse, benchmark_divloss\n",
        )

        print("Copying source code to model directory...")
        directory_to_backup = [
            (".", self.src_root_dir),
            ("Network", self.network_src_dir),
        ]
        for backup_subdir, source_dir in directory_to_backup:
            files = os.listdir(source_dir)
            for fname in files:
                if fname.endswith(".py") or fname.endswith(".ipynb"):
                    dest_fpath = os.path.join(self.model_dir, "backup_source", backup_subdir, fname)
                    os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
                    shutil.copy2(os.path.join(source_dir, fname), dest_fpath)

    def _to_device_batch(self, data_pairs):
        return [item.to(self.device, non_blocking=True) for item in data_pairs]

    def _prepare_batch(self, data_pairs):
        batch = self._to_device_batch(data_pairs)
        if self.predict_mag:
            if len(batch) not in {12, 13}:
                raise ValueError(
                    f"predict_mag=True expects 12 or 13 batch tensors (including optional vesselness), got {len(batch)}."
                )
            if len(batch) == 13:
                u, v, w, u_mag, v_mag, w_mag, u_hr, v_hr, w_hr, mag_hr, venc, mask, vesselness_target = batch
            else:
                u, v, w, u_mag, v_mag, w_mag, u_hr, v_hr, w_hr, mag_hr, venc, mask = batch
                vesselness_target = None
            hires = torch.cat((u_hr, v_hr, w_hr, mag_hr), dim=1)
            return u, v, w, u_mag, v_mag, w_mag, hires, venc, mask, mag_hr, vesselness_target

        if len(batch) == 12:
            u, v, w, u_mag, v_mag, w_mag, u_hr, v_hr, w_hr, venc, mask, vesselness_target = batch
        elif len(batch) == 11:
            u, v, w, u_mag, v_mag, w_mag, u_hr, v_hr, w_hr, venc, mask = batch
            vesselness_target = None
        else:
            raise ValueError(f"Unexpected batch length {len(batch)}.")
        hires = torch.cat((u_hr, v_hr, w_hr), dim=1)
        return u, v, w, u_mag, v_mag, w_mag, hires, venc, mask, None, vesselness_target

    def train_step(self, data_pairs):
        self.model.train()
        u, v, w, u_mag, v_mag, w_mag, hires, venc, mask, _, vesselness_target = self._prepare_batch(data_pairs)
        del venc

        self.optimizer.zero_grad(set_to_none=True)
        predictions, seg_map, attn_map = self._forward_model(u, v, w, u_mag, v_mag, w_mag, mask)
        loss = self.calculate_and_update_metrics(hires, predictions, mask, seg_map, attn_map, vesselness_target, "train")
        loss.backward()
        self.optimizer.step()

    def _forward_model(self, u, v, w, u_mag, v_mag, w_mag, mask=None):
        if getattr(self.model, "requires_mask_input", False):
            result = self.model(u, v, w, u_mag, v_mag, w_mag, mask=mask)
        else:
            result = self.model(u, v, w, u_mag, v_mag, w_mag)
        if isinstance(result, tuple):
            predictions, seg_map = result
        else:
            predictions, seg_map = result, None
        return predictions, seg_map, self._get_attention_map()

    @staticmethod
    def _align_mask_to_shape(mask, target_shape):
        if mask is None:
            return None
        if tuple(mask.shape[-3:]) == tuple(target_shape):
            return mask

        mx, my, mz = [int(v) for v in mask.shape[-3:]]
        tx, ty, tz = [int(v) for v in target_shape]
        fx = mx // tx if tx > 0 else 0
        fy = my // ty if ty > 0 else 0
        fz = mz // tz if tz > 0 else 0

        if (
            fx > 0
            and fy > 0
            and fz > 0
            and mx == tx * fx
            and my == ty * fy
            and mz == tz * fz
        ):
            pooled = mask.unfold(1, fx, fx).unfold(2, fy, fy).unfold(3, fz, fz)
            return pooled.amax(dim=(-1, -2, -3))

        return F.interpolate(mask[:, None], size=tuple(target_shape), mode="nearest")[:, 0]

    @staticmethod
    def _align_mask_to_lr_shape(mask, lr_shape):
        return TrainerController._align_mask_to_shape(mask, lr_shape)

    @staticmethod
    def _align_continuous_to_shape(volume, target_shape):
        if volume is None:
            return None
        if tuple(volume.shape[-3:]) == tuple(target_shape):
            return volume

        mx, my, mz = [int(v) for v in volume.shape[-3:]]
        tx, ty, tz = [int(v) for v in target_shape]
        fx = mx // tx if tx > 0 else 0
        fy = my // ty if ty > 0 else 0
        fz = mz // tz if tz > 0 else 0

        if (
            fx > 0
            and fy > 0
            and fz > 0
            and mx == tx * fx
            and my == ty * fy
            and mz == tz * fz
        ):
            pooled = volume.unfold(1, fx, fx).unfold(2, fy, fy).unfold(3, fz, fz)
            return pooled.mean(dim=(-1, -2, -3))

        return F.interpolate(volume[:, None], size=tuple(target_shape), mode="trilinear", align_corners=False)[:, 0]

    @staticmethod
    def _probability_target_loss(pred, target, loss_type, logits=None):
        loss_type = str(loss_type).strip().lower()
        if loss_type == "bce":
            if logits is not None:
                return F.binary_cross_entropy_with_logits(logits, target)
            pred = pred.clamp(1e-6, 1.0 - 1e-6)
            return F.binary_cross_entropy(pred, target)
        if loss_type == "mse":
            return F.mse_loss(pred, target)
        if loss_type == "l1":
            return F.l1_loss(pred, target)
        if loss_type == "smoothl1":
            return F.smooth_l1_loss(pred, target)
        raise ValueError(f"Unsupported probability target loss type: {loss_type}")

    def _get_attention_map(self):
        attn_map = getattr(self.model, "last_attention_map", None)
        if attn_map is not None:
            return attn_map
        if hasattr(self.model, "stage1"):
            return getattr(self.model.stage1, "last_attention_map", None)
        return None

    @torch.no_grad()
    def test_step(self, data_pairs, return_visuals=False):
        self.model.eval()
        u, v, w, u_mag, v_mag, w_mag, hires, venc, mask, _, vesselness_target = self._prepare_batch(data_pairs)
        del venc

        seg_map = None
        attn_map = None
        if self.val_full_volume:
            lr_input = torch.cat((u, v, w, u_mag, v_mag, w_mag), dim=1)
            lr_mask = self._align_mask_to_lr_shape(mask, lr_input.shape[-3:])
            if lr_mask is not None:
                lr_mask = lr_mask.to(lr_input.dtype)
            if getattr(self.model, "requires_mask_input", False):
                lr_input = torch.cat((lr_input, lr_mask[:, None]), dim=1)

            def _predictor(x):
                mask_patch = x[:, 6:7] if x.shape[1] > 6 else None
                if mask_patch is not None:
                    mask_patch = mask_patch[:, 0]
                pred_patch, seg_patch, attn_patch = self._forward_model(
                    x[:, 0:1],
                    x[:, 1:2],
                    x[:, 2:3],
                    x[:, 3:4],
                    x[:, 4:5],
                    x[:, 5:6],
                    mask_patch,
                )
                if self.model_outputs_seg_map:
                    if seg_patch is None:
                        seg_patch = torch.zeros_like(pred_patch[:, :1])
                outputs = [pred_patch]
                if self.model_outputs_seg_map:
                    outputs.append(seg_patch)
                if self.model_outputs_attn_map:
                    if attn_patch is None:
                        attn_patch = torch.zeros_like(pred_patch[:, :1])
                    outputs.append(attn_patch)
                return torch.cat(outputs, dim=1) if len(outputs) > 1 else pred_patch

            sw_output = sliding_window_inference(
                inputs=lr_input,
                roi_size=(self.val_sw_patch_size, self.val_sw_patch_size, self.val_sw_patch_size),
                sw_batch_size=self.val_sw_batch_size,
                predictor=_predictor,
                overlap=self.val_sw_overlap,
            )
            if self.model_outputs_seg_map:
                pred_channels = int(hires.shape[1])
                cursor = pred_channels
                predictions = sw_output[:, :pred_channels]
                seg_map = sw_output[:, cursor : cursor + 1]
                cursor += 1
                if self.model_outputs_attn_map:
                    attn_map = sw_output[:, cursor : cursor + 1]
            else:
                if self.model_outputs_attn_map:
                    pred_channels = int(hires.shape[1])
                    predictions = sw_output[:, :pred_channels]
                    attn_map = sw_output[:, pred_channels : pred_channels + 1]
                else:
                    predictions = sw_output
        else:
            predictions, seg_map, attn_map = self._forward_model(u, v, w, u_mag, v_mag, w_mag, mask)

        hires, predictions, mask = self._align_spatial_shapes(hires, predictions, mask)
        if seg_map is not None:
            seg_map = seg_map[:, :, : predictions.shape[2], : predictions.shape[3], : predictions.shape[4]]
        if attn_map is not None:
            attn_map = attn_map[:, :, : predictions.shape[2], : predictions.shape[3], : predictions.shape[4]]
        self.calculate_and_update_metrics(hires, predictions, mask, seg_map, attn_map, vesselness_target, "val")
        if return_visuals:
            return predictions, (u, v, w, u_mag, hires, mask)
        return predictions

    def calculate_and_update_metrics(self, hires, predictions, mask, seg_map, attn_map, vesselness_target, metric_set):
        hires, predictions, mask = self._align_spatial_shapes(hires, predictions, mask)
        if seg_map is not None:
            seg_map = seg_map[:, :, : predictions.shape[2], : predictions.shape[3], : predictions.shape[4]]
        if attn_map is not None:
            attn_map = attn_map[:, :, : predictions.shape[2], : predictions.shape[3], : predictions.shape[4]]
        if vesselness_target is not None:
            vesselness_target = vesselness_target[:, : predictions.shape[2], : predictions.shape[3], : predictions.shape[4]]
        total_loss, mse, divloss, mag_mse, seg_dice_loss, seg_bce_loss, seg_vesselness_loss, attn_loss = self.loss_function(
            hires, predictions, mask, seg_map, vesselness_target, attn_map
        )
        rel_error, _vel_rel_error, mag_rel_error = self.accuracy_function(hires, predictions, mask)

        batch_size = hires.shape[0]
        if metric_set == "train":
            self._update_metric("l2_reg_loss", 0.0, batch_size)

        self._update_metric(f"{metric_set}_loss", total_loss.mean().item(), batch_size)
        self._update_metric(f"{metric_set}_mse", mse.mean().item(), batch_size)
        self._update_metric(f"{metric_set}_div", divloss.mean().item(), batch_size)
        if self.predict_mag:
            self._update_metric(f"{metric_set}_mag_mse", mag_mse.mean().item(), batch_size)
            if self.accuracy_include_mag and mag_rel_error is not None:
                self._update_metric(f"{metric_set}_mag_accuracy", mag_rel_error.mean().item(), batch_size)
        if seg_dice_loss is not None:
            self._log_metric("seg_dice_loss", seg_dice_loss.item(), metric_set, batch_size)
        if seg_bce_loss is not None:
            self._log_metric("seg_bce_loss", seg_bce_loss.item(), metric_set, batch_size)
        if seg_vesselness_loss is not None:
            self._log_metric("seg_vesselness_loss", seg_vesselness_loss.item(), metric_set, batch_size)
        if attn_loss is not None:
            self._log_metric("attn_loss", attn_loss.item(), metric_set, batch_size)
        self._update_metric(f"{metric_set}_accuracy", rel_error.mean().item(), batch_size)
        return total_loss.mean()

    def reset_metrics(self):
        self._reset_metric_storage()

    def train_network(self, trainset, valset, n_epoch, testset=None):
        """
        Main training function. Receives trainining and validation DataLoaders.
        """
        print("==================== TRAINING =================", flush=True)
        print(f"Learning rate {self.optimizer.param_groups[0]['lr']:.7f}", flush=True)
        print(f"Start training at {time.ctime()} - {self.unique_model_name}\n", flush=True)
        start_time = time.time()

        previous_loss = np.inf
        no_improve_epochs = 0
        overfit_epochs = 0
        prev_epoch_train_loss = None
        prev_epoch_val_loss = None
        stopped_early = False
        total_batch_train = len(trainset)
        total_batch_val = len(valset)

        for epoch in range(n_epoch):
            self.reset_metrics()
            start_loop = time.time()

            # --- Training ---
            for i, data_pairs in enumerate(trainset):
                self.train_step(data_pairs)
                message = (
                    f"Epoch {epoch + 1} Train batch {i + 1}/{total_batch_train} | "
                    f"loss: {self._metric_value('train_loss'):.5f} "
                    f"({self._metric_value('train_accuracy'):.1f} %) - {time.time() - start_loop:.1f} secs"
                )
                print(f"\r{message}", end="", flush=True)

            # --- Validation ---
            log_recon_this_epoch = (
                self.tb_image_every_n_epochs > 0
                and ((epoch + 1) % self.tb_image_every_n_epochs == 0)
            )
            recon_logged = False
            for i, data_pairs in enumerate(valset):
                if log_recon_this_epoch and not recon_logged:
                    predictions, visual_tensors = self.test_step(data_pairs, return_visuals=True)
                    self._log_validation_reconstructions(epoch + 1, *visual_tensors, predictions)
                    recon_logged = True
                else:
                    self.test_step(data_pairs)
                message = (
                    f"Epoch {epoch + 1} Validation batch {i + 1}/{total_batch_val} | "
                    f"loss: {self._metric_value('val_loss'):.5f} "
                    f"({self._metric_value('val_accuracy'):.1f} %) - {time.time() - start_loop:.1f} secs"
                )
                print(f"\r{message}", end="", flush=True)

            # --- Epoch logging ---
            message = (
                f"\rEpoch {epoch + 1} Train loss: {self._metric_value('train_loss'):.5f} "
                f"({self._metric_value('train_accuracy'):.1f} %), "
                f"Val loss: {self._metric_value('val_loss'):.5f} "
                f"({self._metric_value('val_accuracy'):.1f} %) - {time.time() - start_loop:.1f} secs"
            )

            loss_values = [f"{self._metric_value(key):.5f}" for key in self.metric_keys]
            loss_str = ",".join(loss_values)
            lr = self.optimizer.param_groups[0]["lr"]
            log_line = f"{epoch + 1},{loss_str},{lr:.6f},{time.time() - start_loop:.1f}"

            train_loss_epoch = self._metric_value("train_loss")
            val_loss_epoch = self._metric_value("val_loss")

            if self.scheduler is not None:
                self.scheduler.step(val_loss_epoch)

            self._update_summary_logging(epoch)
            self.save_latest_model(epoch + 1)

            if self._metric_value(self.accuracy_metric) < (previous_loss - self.early_stopping_min_delta):
                self.save_best_model(epoch + 1)
                previous_loss = self._metric_value(self.accuracy_metric)
                no_improve_epochs = 0

                message += " **"
                log_line += ",**"

                if self.QUICKSAVE_ENABLED and testset is not None:
                    quick_loss, quick_accuracy, quick_mse, quick_div = self.quicksave(testset, epoch + 1)
                    quick_loss = np.mean(quick_loss)
                    quick_accuracy = np.mean(quick_accuracy)
                    quick_mse = np.mean(quick_mse)
                    quick_div = np.mean(quick_div)

                    message += f" Benchmark loss: {quick_loss:.5f} ({quick_accuracy:.1f} %)"
                    log_line += f", {quick_loss:.7f}, {quick_accuracy:.2f}%, {quick_mse:.7f}, {quick_div:.7f}"
            else:
                no_improve_epochs += 1

            if prev_epoch_train_loss is not None and prev_epoch_val_loss is not None:
                train_improved = train_loss_epoch < (prev_epoch_train_loss - 1e-12)
                val_worsened = val_loss_epoch > (prev_epoch_val_loss + self.overfit_min_delta)
                if train_improved and val_worsened:
                    overfit_epochs += 1
                else:
                    overfit_epochs = 0
            prev_epoch_train_loss = train_loss_epoch
            prev_epoch_val_loss = val_loss_epoch

            print(message, flush=True)
            utility.log_to_file(self.logfile, log_line + "\n")

            stop_due_to_no_improve = (
                self.early_stopping_patience > 0
                and no_improve_epochs >= self.early_stopping_patience
                and (val_loss_epoch >= previous_loss - self.early_stopping_min_delta)
            )
            stop_due_to_overfit = self.overfit_patience > 0 and overfit_epochs >= self.overfit_patience

            if stop_due_to_no_improve or stop_due_to_overfit:
                if stop_due_to_no_improve:
                    reason = (
                        f"Early stopping: no val_loss improvement for {no_improve_epochs} epochs "
                        f"(patience={self.early_stopping_patience})."
                    )
                else:
                    reason = (
                        f"Overfitting detected: val_loss worsened while train_loss improved for "
                        f"{overfit_epochs} consecutive epochs (patience={self.overfit_patience})."
                    )
                print(reason, flush=True)
                utility.log_to_file(self.logfile, reason + "\n")
                stopped_early = True
                break

        hrs, mins, secs = utility.calculate_time_elapsed(start_time)
        message = f"\nTraining {self.network_name} completed! - name: {self.unique_model_name}"
        if stopped_early:
            message += "\nStopped early."
        message += f"\nTotal training time: {hrs} hrs {mins} mins {secs} secs."
        message += f"\nFinished at {time.ctime()}"
        message += "\n==================== END TRAINING ================="
        utility.log_to_file(self.logfile, message)
        print(message, flush=True)

        self.train_writer.close()
        self.val_writer.close()

    def save_best_model(self, epoch):
        """
        Save model and optimizer state to continue training later.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "network_name": self.network_name,
            "model_variant": self.model_variant,
            "res_increase": self.res_increase,
            "low_resblock": self.low_resblock,
            "hi_resblock": self.hi_resblock,
            "channel_nr": self.channel_nr,
            "predict_mag": self.predict_mag,
            "mag_loss_weight": self.mag_loss_weight,
            "non_fluid_loss_weight": self.non_fluid_weight,
            "outside_tv_weight": self.outside_tv_weight,
            "use_seg_head": self.use_seg_head,
            # legacy alias, kept so older tooling can still read the flag
            "cascade_use_seg_head": self.use_seg_head,
            "seg_loss_weight": self.seg_loss_weight,
            "seg_loss_dice_weight": self.seg_loss_dice_weight,
            "seg_loss_bce_weight": self.seg_loss_bce_weight,
            "seg_vesselness_loss_weight": self.seg_vesselness_loss_weight,
            "seg_vesselness_loss_type": self.seg_vesselness_loss_type,
            "attn_loss_weight": self.attn_loss_weight,
            "attn_loss_type": self.attn_loss_type,
            "attn_supervision_target": self.attn_supervision_target,
            "seg_head_bias_init": self.seg_head_bias_init,
            "cascade_seg_head_bias_init": self.seg_head_bias_init,
        }
        torch.save(checkpoint, f"{self.model_path}-best.pt")

    def restore_model(self, old_model_dir, old_model_file):
        """
        Restore model and optimizer state.
        """
        model_weights_path = f"{old_model_dir}/{old_model_file}"
        checkpoint = torch.load(model_weights_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint_variant = checkpoint.get("model_variant", "original")
            checkpoint_variant = normalize_model_variant(checkpoint_variant)
            if checkpoint_variant != self.model_variant:
                print(
                    f"Warning: checkpoint model_variant={checkpoint_variant!r} "
                    f"differs from current model_variant={self.model_variant!r}."
                )
            _load_state_dict_with_seg_head_compat(self.model, checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except ValueError as exc:
                    print(f"Warning: optimizer state not restored due to parameter mismatch: {exc}")
            if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except ValueError as exc:
                    print(f"Warning: scheduler state not restored due to parameter mismatch: {exc}")
            if "learning_rate" in checkpoint:
                for group in self.optimizer.param_groups:
                    group["lr"] = checkpoint["learning_rate"]
        else:
            _load_state_dict_with_seg_head_compat(self.model, checkpoint)

    def _update_summary_logging(self, epoch):
        """
        Summary logging for epoch-level metrics.
        """
        lr = self.optimizer.param_groups[0]["lr"]
        self.train_writer.add_scalar(f"{self.network_name}/learning_rate", lr, epoch)

        train_metrics = {
            "loss": self._metric_value("train_loss"),
            "accuracy": self._metric_value("train_accuracy"),
            "mse": self._metric_value("train_mse"),
            "div": self._metric_value("train_div"),
            "l2_reg_loss": self._metric_value("l2_reg_loss"),
        }
        if self.predict_mag:
            train_metrics["mag_mse"] = self._metric_value("train_mag_mse")
            if self.accuracy_include_mag:
                train_metrics["mag_accuracy"] = self._metric_value("train_mag_accuracy")
        if "train_seg_dice_loss" in self.metric_sums:
            train_metrics["seg_dice_loss"] = self._metric_value("train_seg_dice_loss")
        if "train_seg_bce_loss" in self.metric_sums:
            train_metrics["seg_bce_loss"] = self._metric_value("train_seg_bce_loss")
        if "train_seg_vesselness_loss" in self.metric_sums:
            train_metrics["seg_vesselness_loss"] = self._metric_value("train_seg_vesselness_loss")
        if "train_attn_loss" in self.metric_sums:
            train_metrics["attn_loss"] = self._metric_value("train_attn_loss")
        for key, value in train_metrics.items():
            self.train_writer.add_scalar(f"{self.network_name}/{key}", value, epoch)

        val_metrics = {
            "loss": self._metric_value("val_loss"),
            "accuracy": self._metric_value("val_accuracy"),
            "mse": self._metric_value("val_mse"),
            "div": self._metric_value("val_div"),
        }
        if self.predict_mag:
            val_metrics["mag_mse"] = self._metric_value("val_mag_mse")
            if self.accuracy_include_mag:
                val_metrics["mag_accuracy"] = self._metric_value("val_mag_accuracy")
        if "val_seg_dice_loss" in self.metric_sums:
            val_metrics["seg_dice_loss"] = self._metric_value("val_seg_dice_loss")
        if "val_seg_bce_loss" in self.metric_sums:
            val_metrics["seg_bce_loss"] = self._metric_value("val_seg_bce_loss")
        if "val_seg_vesselness_loss" in self.metric_sums:
            val_metrics["seg_vesselness_loss"] = self._metric_value("val_seg_vesselness_loss")
        if "val_attn_loss" in self.metric_sums:
            val_metrics["attn_loss"] = self._metric_value("val_attn_loss")
        for key, value in val_metrics.items():
            self.val_writer.add_scalar(f"{self.network_name}/{key}", value, epoch)

    @torch.no_grad()
    def quicksave(self, testset, epoch_nr):
        """
        Predict a batch of data from the benchmark testset and save it in HDF5.
        """
        self.model.eval()
        for data_pairs in testset:
            u, v, w, u_mag, v_mag, w_mag, hires, venc, mask, mag_hr, vesselness_target = self._prepare_batch(data_pairs)
            preds, seg_map, attn_map = self._forward_model(u, v, w, u_mag, v_mag, w_mag, mask)

            loss_val, mse, divloss, _mag_mse, _seg_dice, _seg_bce, _seg_vesselness, _attn_loss = self.loss_function(
                hires,
                preds,
                mask,
                seg_map,
                vesselness_target,
                attn_map,
            )
            rel_loss, _vel_rel_loss, _mag_rel_loss = self.accuracy_function(hires, preds, mask)
            break

        quicksave_filename = f"quicksave_{self.network_name}.h5"
        h5util.save_predictions(self.model_dir, quicksave_filename, "epoch", np.asarray([epoch_nr]), compression="gzip")

        preds_np = preds.detach().cpu().numpy()
        h5util.save_predictions(self.model_dir, quicksave_filename, "u", preds_np[:, 0], compression="gzip")
        h5util.save_predictions(self.model_dir, quicksave_filename, "v", preds_np[:, 1], compression="gzip")
        h5util.save_predictions(self.model_dir, quicksave_filename, "w", preds_np[:, 2], compression="gzip")
        if self.predict_mag:
            h5util.save_predictions(self.model_dir, quicksave_filename, "mag", preds_np[:, 3], compression="gzip")

        if epoch_nr == 1:
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_u", u.detach().cpu().numpy().squeeze(1), compression="gzip")
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_v", v.detach().cpu().numpy().squeeze(1), compression="gzip")
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_w", w.detach().cpu().numpy().squeeze(1), compression="gzip")

            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_u", hires[:, 0].detach().cpu().numpy(), compression="gzip")
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_v", hires[:, 1].detach().cpu().numpy(), compression="gzip")
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_w", hires[:, 2].detach().cpu().numpy(), compression="gzip")
            if self.predict_mag and mag_hr is not None:
                h5util.save_predictions(
                    self.model_dir,
                    quicksave_filename,
                    "hr_mag",
                    mag_hr.detach().cpu().numpy().squeeze(1),
                    compression="gzip",
                )

            h5util.save_predictions(self.model_dir, quicksave_filename, "venc", venc.detach().cpu().numpy(), compression="gzip")
            h5util.save_predictions(self.model_dir, quicksave_filename, "mask", mask.detach().cpu().numpy(), compression="gzip")

        return (
            loss_val.detach().cpu().numpy(),
            rel_loss.detach().cpu().numpy(),
            mse.detach().cpu().numpy(),
            divloss.detach().cpu().numpy(),
        )
