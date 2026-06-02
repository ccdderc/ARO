import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.masked_modules import MaskedTrainingModule
from src.models import ModelConfig

from omegaconf import DictConfig
import hydra
from src.omic_dataclass import OmicDataModule
from src.autoencoder_modules import AEModule, VAEModule


def build_model_config(cfg: DictConfig, inp_dim: int) -> ModelConfig:
    model_type = cfg.get("model")
    if model_type == "Linear":
        return {
            "type": "Linear",
            "inp_dim": inp_dim,
            "out_dim": inp_dim,
            "hidden_dim": cfg.get("hidden_dim", 512),
            "n_hidden_layers": cfg.get("n_hidden_layers", 3),
            "dropout": cfg.get("dropout", 0.0),
            "dropout_input": cfg.get("dropout_input", 0.0),
            "use_log_space": cfg.get("use_log_space", False),
            "use_batch_norm": cfg.get("use_batch_norm", False),
        }
    elif model_type == "Attention":
        return {
            "type": "Attention",
            "num_omics": cfg.get("num_omics", 74_002),
            "n_heads": cfg.get("n_heads", 8),
            "head_dim": cfg.get("head_dim", 64),
            "n_hidden_layers": cfg.get("n_hidden_layers", 3),
            "dropout": cfg.get("dropout", 0.1),
            "causal": False,
        }
    elif model_type == "LinearVAE":
        return {
            "type": "LinearVAE",
            "inp_dim": inp_dim,
            "hidden_dim": cfg.get("hidden_dim", 512),
            "dist_dim": cfg.get("dist_dim", 20),
            "n_enc": cfg.get("n_enc", 3),
            "n_dec": cfg.get("n_dec", 3),
            "res": cfg.get("res", False),
            "use_log_space": cfg.get("use_log_space", False),
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@hydra.main(version_base=None, config_path="../configs/", config_name="vae_linear")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    epochs = cfg.get("epochs", 20)

    run_name = f"Raw_{cfg.get('model')}"

    weight_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "logs", run_name
    )
    logger = WandbLogger(
        save_dir="logs",
        name=run_name,
        project="OmicBert",
        reinit=True,
        entity="OmicBertTeam",
    )

    data_path = "data/cleaned_data_matrix.pkl"

    dm = OmicDataModule(
        file_path=data_path,
        batch_size=cfg.get("batch_size", 8),
        use_log_space=cfg.get("use_log_space", False),
    )
    dm.setup()
    x_sample, _ = dm.train_ds[0]

    model_config = build_model_config(cfg, inp_dim=x_sample.shape[-1])
    module_type = cfg.get("module", "ae")
    module_kwargs = dict(
        model_config=model_config,
        initial_lr=cfg.get("lr", 0.01),
        weight_decay=cfg.get("weight_decay", 0),
        mask_vals=cfg.get("mask_vals", False),
    )
    match module_type:
        case "masked":
            model = MaskedTrainingModule(**module_kwargs)
        case "ae":
            model = AEModule(**module_kwargs)
        case "vae":
            model = VAEModule(**module_kwargs, kl_weight=cfg.get("kl_weight", 1.0))
        case _:
            raise ValueError(f"Unknown module type: {module_type}")
    model = model.to(device)

    # Compile model for faster execution on CUDA
    if device.type == "cuda":
        model.model = torch.compile(model.model)

    # Use bf16 on Ampere+ GPUs, fp16 on older CUDA GPUs
    precision = cfg.get("precision", None)
    if precision is None:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        elif device.type == "cuda":
            precision = "16-mixed"
        else:
            precision = "32"

    save_checkpoints = cfg.get("save_checkpoints", False)
    callbacks = []
    if save_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                save_weights_only=False,
                mode="min",
                monitor="val_loss",
                save_top_k=3,
                every_n_epochs=20,
                save_last="link",
            )
        )

    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_progress_bar=True,
        callbacks=callbacks,
        enable_checkpointing=save_checkpoints,
        precision=precision,
        default_root_dir=weight_dir,
        log_every_n_steps=1,
        accelerator=device.type,
        logger=logger,
    )

    trainer.fit(model, datamodule=dm)

    # Upload best checkpoint as a wandb artifact
    if save_checkpoints and trainer.checkpoint_callback:
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            artifact = wandb.Artifact(
                name=f"{run_name}-best-checkpoint",
                type="model",
                metadata={
                    "val_loss": float(trainer.checkpoint_callback.best_model_score)
                },
            )
            artifact.add_file(best_path)
            logger.experiment.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
