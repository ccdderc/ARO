from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb

from src.models import ModelConfig, build_model


def _plot_sample_comparison(
    actual: np.ndarray, predicted: np.ndarray, prefix: str, epoch: int
) -> wandb.Image:
    """Create a line plot comparing actual vs predicted, plus a residual plot."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1], sharex=True)
    x = np.arange(len(actual))

    axes[0].plot(x, actual, alpha=0.7, linewidth=0.5, label="Actual")
    axes[0].plot(x, predicted, alpha=0.7, linewidth=0.5, label="Predicted")
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"{prefix} — epoch {epoch}")
    axes[0].legend()

    residual = actual - predicted
    axes[1].plot(x, residual, alpha=0.7, linewidth=0.5, color="red")
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_xlabel("Feature index")
    axes[1].set_ylabel("Residual")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


class AEModule(pl.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        initial_lr: float,
        weight_decay: float = 0.0,
        mask_vals: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.initial_lr: float = initial_lr
        self.weight_decay: float = weight_decay
        self.model: torch.nn.Module = build_model(model_config)
        self.mask_vals = mask_vals

    def forward(self, data: torch.Tensor, training: bool = False):
        preds = self.model(data)
        return preds

    def compute_loss(
        self,
        preds: torch.Tensor,
        orig_vals: torch.Tensor,
        nan_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Exclude positions that were originally NaN from the loss
        valid = ~nan_mask
        loss = F.huber_loss(preds[valid], orig_vals[valid])
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, nan_mask = batch
        masked_data = data.clone()
        if self.mask_vals:
            masked_data[:, 50455] = 0
        out = self.forward(masked_data, training=True)

        ret = self.compute_loss(out, data, nan_mask)
        self.log("train_loss", ret)
        self.log("lr", self.lr_schedulers().get_last_lr()[0])
        return ret

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, nan_mask = batch
        masked_data = data.clone()
        if self.mask_vals:
            masked_data[:, 50455] = 0
        out = self.forward(masked_data, training=True)

        loss = self.compute_loss(out, data, nan_mask)
        self.log("val_loss", loss)
        if batch_idx == 0:
            self._log_predictions(data, out, "val")
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, nan_mask = batch
        masked_data = data.clone()
        if self.mask_vals:
            masked_data[:, 50455] = 0
        out = self.forward(masked_data, training=True)

        loss = self.compute_loss(out, data, nan_mask)
        self.log("test_loss", loss)
        if batch_idx == 0:
            self._log_predictions(data, out, "test")
        return loss

    def _log_predictions(
        self,
        data: torch.Tensor,
        preds: torch.Tensor,
        prefix: str,
    ) -> None:
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return
        actual = data[0].cpu().float().numpy()
        predicted = preds[0].cpu().float().numpy()
        img = _plot_sample_comparison(actual, predicted, prefix, self.current_epoch)
        self.logger.experiment.log({f"{prefix}/sample_plot": img})

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics["val_loss"])


class VAEModule(pl.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        initial_lr: float,
        weight_decay: float = 0.0,
        kl_weight: float = 1.0,
        kl_warmup_epochs: int = 750,
        mask_vals: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.initial_lr: float = initial_lr
        self.weight_decay: float = weight_decay
        self.kl_weight: float = kl_weight
        self.kl_warmup_epochs: int = kl_warmup_epochs
        self.model: torch.nn.Module = build_model(model_config)
        self.mask_vals = mask_vals

    def forward(self, data: torch.Tensor):
        return self.model(data)  # returns (recon, mu, log_var)

    def compute_loss(
        self,
        recon: torch.Tensor,
        orig: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        nan_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Exclude positions that were originally NaN from reconstruction loss
        valid = ~nan_mask
        recon_loss = F.huber_loss(recon[valid], orig[valid])
        # KL divergence: -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        # Linear warmup: scale KL weight from 0 to kl_weight over kl_warmup_epochs
        warmup_scale = min(1.0, self.current_epoch / max(1, self.kl_warmup_epochs))
        effective_kl_weight = self.kl_weight * warmup_scale
        total = recon_loss + effective_kl_weight * kl_loss
        return total, recon_loss, kl_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.01,
            patience=350,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_recon_loss",
            },
        }

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, nan_mask = batch
        recon, mu, log_var = self.forward(data)
        total, recon_loss, kl_loss = self.compute_loss(
            recon, data, mu, log_var, nan_mask
        )
        self.log("train_loss", total)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl_loss", kl_loss)
        self.log("lr", self.lr_schedulers().get_last_lr()[0])
        return total

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, nan_mask = batch
        recon, mu, log_var = self.forward(data)
        total, recon_loss, kl_loss = self.compute_loss(
            recon, data, mu, log_var, nan_mask
        )
        self.log("val_loss", total)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl_loss", kl_loss)
        if batch_idx == 0:
            self._log_predictions(data, recon, "val")
        return total

    def _log_predictions(
        self,
        data: torch.Tensor,
        preds: torch.Tensor,
        prefix: str,
    ) -> None:
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return
        actual = data[0].cpu().float().numpy()
        predicted = preds[0].cpu().float().numpy()
        img = _plot_sample_comparison(actual, predicted, prefix, self.current_epoch)
        self.logger.experiment.log({f"{prefix}/sample_plot": img})

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, nan_mask = batch
        recon, mu, log_var = self.forward(data)
        total, recon_loss, kl_loss = self.compute_loss(
            recon, data, mu, log_var, nan_mask
        )
        self.log("test_loss", total)
        self.log("test_recon_loss", recon_loss)
        self.log("test_kl_loss", kl_loss)
        return total

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics["val_loss"])
