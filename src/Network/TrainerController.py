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
from torch.utils.tensorboard import SummaryWriter

from . import h5util, loss_utils, utility
from .SR4DFlowNet import SR4DFlowNet


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
    ):
        """
        TrainerController constructor.
        Setup model, loss functions and optimizer here.
        """
        self.div_weight = 0  # Weighting for divergence loss
        self.non_fluid_weight = 1  # Weighting for non fluid region

        # General params
        self.patch_size = patch_size
        self.res_increase = res_increase
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training params
        self.QUICKSAVE_ENABLED = quicksave_enable

        # Network
        self.network_name = network_name
        self.model = SR4DFlowNet(
            res_increase=res_increase,
            low_resblock=low_resblock,
            hi_resblock=hi_resblock,
            channel_nr=64,
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
        self.accuracy_metric = "val_loss"

        print(f"Divergence loss2 * {self.div_weight}")
        print(f"Accuracy metric: {self.accuracy_metric}")
        print(f"Using device: {self.device}")

        # Learning rate and optimizer (weight_decay replicates L2 regularization)
        self.learning_rate = initial_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-7)
        self._reset_metric_storage()

    def _reset_metric_storage(self):
        self.metric_sums = {k: 0.0 for k in self.metric_keys}
        self.metric_counts = {k: 0 for k in self.metric_keys}

    def _update_metric(self, key, value, n=1):
        self.metric_sums[key] += float(value) * n
        self.metric_counts[key] += n

    def _metric_value(self, key):
        if self.metric_counts[key] == 0:
            return 0.0
        return self.metric_sums[key] / self.metric_counts[key]

    def save_latest_model(self, epoch):
        if epoch > 0 and epoch % 10 == 0:
            checkpoint_path = f"{self.model_path}-latest.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "learning_rate": self.learning_rate,
                    "network_name": self.network_name,
                },
                checkpoint_path,
            )
            message = f"Saving current model - {time.ctime()}\n"
            print(message)

    def loss_function(self, y_true, y_pred, mask):
        """
        Calculate Total Loss function:
        Loss = MSE + weight * div_loss2
        """
        u, v, w = y_true[:, 0], y_true[:, 1], y_true[:, 2]
        u_pred, v_pred, w_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

        mse = self.calculate_mse(u, v, w, u_pred, v_pred, w_pred)

        # === Separate mse ===
        non_fluid_mask = (mask < 0.5).float()
        epsilon = 1.0  # minimum 1 pixel

        fluid_mse = mse * mask
        fluid_mse = fluid_mse.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + epsilon)

        non_fluid_mse = mse * non_fluid_mask
        non_fluid_mse = non_fluid_mse.sum(dim=(1, 2, 3)) / (non_fluid_mask.sum(dim=(1, 2, 3)) + epsilon)

        mse = fluid_mse + non_fluid_mse

        divergence_loss = torch.zeros_like(mse)
        total_loss = mse + divergence_loss
        return total_loss, mse, divergence_loss

    def accuracy_function(self, y_true, y_pred, mask):
        """
        Calculate relative speed error.
        """
        u, v, w = y_true[:, 0], y_true[:, 1], y_true[:, 2]
        u_pred, v_pred, w_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        return loss_utils.calculate_relative_error(u_pred, v_pred, w_pred, u, v, w, mask)

    @staticmethod
    def calculate_mse(u, v, w, u_pred, v_pred, w_pred):
        """
        Calculate speed magnitude error.
        """
        return (u_pred - u) ** 2 + (v_pred - v) ** 2 + (w_pred - w) ** 2

    def init_model_dir(self):
        """
        Create model directory to save the weights with a [network_name]_[datetime] format.
        Also prepare logfile and tensorboard summary within the directory.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.unique_model_name = f"{self.network_name}_{timestamp}"

        self.model_dir = f"../models/{self.unique_model_name}"
        self.model_path = f"{self.model_dir}/{self.network_name}"

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
        utility.log_to_file(self.logfile, f"Initial learning rate: {self.learning_rate}\n")
        utility.log_to_file(self.logfile, f"Accuracy metric: {self.accuracy_metric}\n")
        utility.log_to_file(self.logfile, f"Divergence weight: {self.div_weight}\n")

        stat_names = ",".join(self.metric_keys)
        utility.log_to_file(
            self.logfile,
            f"epoch, {stat_names}, learning rate, elapsed (sec), best_model, benchmark_err, benchmark_rel_err, benchmark_mse, benchmark_divloss\n",
        )

        print("Copying source code to model directory...")
        directory_to_backup = [".", "Network"]
        for directory in directory_to_backup:
            files = os.listdir(directory)
            for fname in files:
                if fname.endswith(".py") or fname.endswith(".ipynb"):
                    dest_fpath = os.path.join(self.model_dir, "backup_source", directory, fname)
                    os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
                    shutil.copy2(f"{directory}/{fname}", dest_fpath)

    def _to_device_batch(self, data_pairs):
        return [item.to(self.device, non_blocking=True) for item in data_pairs]

    def train_step(self, data_pairs):
        self.model.train()
        u, v, w, u_mag, v_mag, w_mag, u_hr, v_hr, w_hr, venc, mask = self._to_device_batch(data_pairs)
        del venc

        hires = torch.cat((u_hr, v_hr, w_hr), dim=1)
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.model(u, v, w, u_mag, v_mag, w_mag)
        loss = self.calculate_and_update_metrics(hires, predictions, mask, "train")
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def test_step(self, data_pairs):
        self.model.eval()
        u, v, w, u_mag, v_mag, w_mag, u_hr, v_hr, w_hr, venc, mask = self._to_device_batch(data_pairs)
        del venc

        hires = torch.cat((u_hr, v_hr, w_hr), dim=1)
        predictions = self.model(u, v, w, u_mag, v_mag, w_mag)
        self.calculate_and_update_metrics(hires, predictions, mask, "val")
        return predictions

    def calculate_and_update_metrics(self, hires, predictions, mask, metric_set):
        total_loss, mse, divloss = self.loss_function(hires, predictions, mask)
        rel_error = self.accuracy_function(hires, predictions, mask)

        batch_size = hires.shape[0]
        if metric_set == "train":
            self._update_metric("l2_reg_loss", 0.0, batch_size)

        self._update_metric(f"{metric_set}_loss", total_loss.mean().item(), batch_size)
        self._update_metric(f"{metric_set}_mse", mse.mean().item(), batch_size)
        self._update_metric(f"{metric_set}_div", divloss.mean().item(), batch_size)
        self._update_metric(f"{metric_set}_accuracy", rel_error.mean().item(), batch_size)
        return total_loss.mean()

    def reset_metrics(self):
        self._reset_metric_storage()

    def train_network(self, trainset, valset, n_epoch, testset=None):
        """
        Main training function. Receives trainining and validation DataLoaders.
        """
        print("==================== TRAINING =================")
        print(f"Learning rate {self.optimizer.param_groups[0]['lr']:.7f}")
        print(f"Start training at {time.ctime()} - {self.unique_model_name}\n")
        start_time = time.time()

        previous_loss = np.inf
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
                print(f"\r{message}", end="")

            # --- Validation ---
            for i, data_pairs in enumerate(valset):
                self.test_step(data_pairs)
                message = (
                    f"Epoch {epoch + 1} Validation batch {i + 1}/{total_batch_val} | "
                    f"loss: {self._metric_value('val_loss'):.5f} "
                    f"({self._metric_value('val_accuracy'):.1f} %) - {time.time() - start_loop:.1f} secs"
                )
                print(f"\r{message}", end="")

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

            self._update_summary_logging(epoch)
            self.save_latest_model(epoch + 1)

            if self._metric_value(self.accuracy_metric) < previous_loss:
                self.save_best_model(epoch + 1)
                previous_loss = self._metric_value(self.accuracy_metric)

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

            print(message)
            utility.log_to_file(self.logfile, log_line + "\n")

        hrs, mins, secs = utility.calculate_time_elapsed(start_time)
        message = f"\nTraining {self.network_name} completed! - name: {self.unique_model_name}"
        message += f"\nTotal training time: {hrs} hrs {mins} mins {secs} secs."
        message += f"\nFinished at {time.ctime()}"
        message += "\n==================== END TRAINING ================="
        utility.log_to_file(self.logfile, message)
        print(message)

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
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "network_name": self.network_name,
        }
        torch.save(checkpoint, f"{self.model_path}-best.pt")

    def restore_model(self, old_model_dir, old_model_file):
        """
        Restore model and optimizer state.
        """
        model_weights_path = f"{old_model_dir}/{old_model_file}"
        checkpoint = torch.load(model_weights_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "learning_rate" in checkpoint:
                for group in self.optimizer.param_groups:
                    group["lr"] = checkpoint["learning_rate"]
        else:
            self.model.load_state_dict(checkpoint)

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
        for key, value in train_metrics.items():
            self.train_writer.add_scalar(f"{self.network_name}/{key}", value, epoch)

        val_metrics = {
            "loss": self._metric_value("val_loss"),
            "accuracy": self._metric_value("val_accuracy"),
            "mse": self._metric_value("val_mse"),
            "div": self._metric_value("val_div"),
        }
        for key, value in val_metrics.items():
            self.val_writer.add_scalar(f"{self.network_name}/{key}", value, epoch)

    @torch.no_grad()
    def quicksave(self, testset, epoch_nr):
        """
        Predict a batch of data from the benchmark testset and save it in HDF5.
        """
        self.model.eval()
        for data_pairs in testset:
            u, v, w, u_mag, v_mag, w_mag, u_hr, v_hr, w_hr, venc, mask = self._to_device_batch(data_pairs)
            hires = torch.cat((u_hr, v_hr, w_hr), dim=1)
            preds = self.model(u, v, w, u_mag, v_mag, w_mag)

            loss_val, mse, divloss = self.loss_function(hires, preds, mask)
            rel_loss = self.accuracy_function(hires, preds, mask)
            break

        quicksave_filename = f"quicksave_{self.network_name}.h5"
        h5util.save_predictions(self.model_dir, quicksave_filename, "epoch", np.asarray([epoch_nr]), compression="gzip")

        preds_np = preds.detach().cpu().numpy()
        h5util.save_predictions(self.model_dir, quicksave_filename, "u", preds_np[:, 0], compression="gzip")
        h5util.save_predictions(self.model_dir, quicksave_filename, "v", preds_np[:, 1], compression="gzip")
        h5util.save_predictions(self.model_dir, quicksave_filename, "w", preds_np[:, 2], compression="gzip")

        if epoch_nr == 1:
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_u", u.detach().cpu().numpy().squeeze(1), compression="gzip")
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_v", v.detach().cpu().numpy().squeeze(1), compression="gzip")
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_w", w.detach().cpu().numpy().squeeze(1), compression="gzip")

            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_u", u_hr.detach().cpu().numpy().squeeze(1), compression="gzip")
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_v", v_hr.detach().cpu().numpy().squeeze(1), compression="gzip")
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_w", w_hr.detach().cpu().numpy().squeeze(1), compression="gzip")

            h5util.save_predictions(self.model_dir, quicksave_filename, "venc", venc.detach().cpu().numpy(), compression="gzip")
            h5util.save_predictions(self.model_dir, quicksave_filename, "mask", mask.detach().cpu().numpy(), compression="gzip")

        return (
            loss_val.detach().cpu().numpy(),
            rel_loss.detach().cpu().numpy(),
            mse.detach().cpu().numpy(),
            divloss.detach().cpu().numpy(),
        )
