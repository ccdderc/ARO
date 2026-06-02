from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.models import ModelConfig, build_model


class MaskedTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        initial_lr: float,
        weight_decay: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.initial_lr: float = initial_lr
        self.weight_decay: float = weight_decay
        self.model: torch.nn.Module = build_model(model_config)

    def forward(self, batch: torch.Tensor, training: bool = False):
        mask = self.generate_random_mask(batch)
        preds = self.model(batch, mask=mask)
        return preds, mask

    def compute_loss(
        self,
        model_out: Tuple[torch.Tensor, torch.Tensor],
        orig_vals: torch.Tensor,
    ) -> torch.Tensor:
        preds, mask = model_out
        masked_preds = preds[mask]
        masked_labels = orig_vals.data[mask]
        mse = F.huber_loss(masked_preds, masked_labels)
        # Penalize the model if the variance of predictions is much lower than the variance of the labels
        var_loss = torch.abs(masked_preds.var() - masked_labels.var())
        return mse + 0.1 * var_loss

    def generate_random_mask(
        self, batch: torch.Tensor, mask_prob: float = 0.15
    ) -> torch.Tensor:
        """Generate a random boolean mask over the input.

        Args:
            batch: Input batch with data of shape (batch_size, seq_len).
            mask_prob: Probability of masking each position.

        Returns:
            Boolean tensor of shape (batch_size, seq_len), True = masked.
        """
        probability_matrix = torch.full(batch.shape, mask_prob, device=batch.device)
        return torch.bernoulli(probability_matrix).bool()

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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        out = self.forward(batch, training=True)

        ret = self.compute_loss(out)
        self.log("train_loss", ret)
        self.log("lr", self.lr_schedulers().get_last_lr()[0])
        return ret

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        out = self.forward(batch, training=False)

        loss = self.compute_loss(out)
        self.log("val_loss", loss)
        self.test_step(batch, batch_idx)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        out = self.forward(batch)

        ret = self.compute_loss(out)
        self.log("test_loss", ret)
        return ret

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics["val_loss"])
