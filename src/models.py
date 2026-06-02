from __future__ import annotations

from typing import Literal, TypedDict

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, unpack

try:
    from flash_attn.cute import flash_attn_func
except ImportError:
    flash_attn_func = None


class LinearModelConfig(TypedDict):
    type: Literal["Linear"]
    inp_dim: int
    out_dim: int
    hidden_dim: int
    n_hidden_layers: int
    dropout: float
    dropout_input: float
    use_log_space: bool
    use_batch_norm: bool


class AttentionModelConfig(TypedDict):
    type: Literal["Attention"]
    num_omics: int
    n_heads: int
    head_dim: int
    n_hidden_layers: int
    dropout: float
    causal: bool


class LinearVAEModelConfig(TypedDict):
    type: Literal["LinearVAE"]
    inp_dim: int
    hidden_dim: int
    dist_dim: int
    n_enc: int
    n_dec: int
    res: bool
    use_log_space: bool
    dropout: float


ModelConfig = LinearModelConfig | AttentionModelConfig | LinearVAEModelConfig


def build_model(config: ModelConfig) -> nn.Module:
    match config["type"]:
        case "Linear":
            return LinearModel.build(config)
        case "Attention":
            return AttentionModel.build(config)
        case "LinearVAE":
            return LinearVAE.build(config)
        case _:
            raise ValueError(f"Unknown model type: {config['type']}")


class LinearVAE(nn.Module):
    def __init__(
        self,
        inp_dim,
        hidden_dim,
        dist_dim,
        n_enc=2,
        n_dec=2,
        res=False,
        dropout=0.2,
        use_log_space=False,
    ):
        super().__init__()

        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.dist_dim = dist_dim
        self.res = res
        self.use_log_space = use_log_space

        self.inp_lin = nn.Linear(self.inp_dim, self.hidden_dim)
        self.out_lin = nn.Linear(self.hidden_dim, self.inp_dim)
        self.enc = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_enc)]
        )
        self.mu_lin = nn.Linear(self.hidden_dim, self.dist_dim)
        self.log_var_lin = nn.Linear(self.hidden_dim, self.dist_dim)

        self.dec_lin = nn.Linear(self.dist_dim, self.hidden_dim)

        self.dec = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_dec)]
        )

        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h = self.dropout(F.relu(self.inp_lin(x)))
        for layer in self.enc:
            h_res = h
            h = self.dropout(F.relu(layer(h)))
            if self.res:
                h = h + h_res

        mu = self.mu_lin(h)
        log_var = self.log_var_lin(h).clamp(-10, 2)

        # Only sample during training; use mu directly at eval for clean reconstructions
        z = self.reparameterize(mu, log_var) if self.training else mu

        # Decode
        h = self.dropout(F.relu(self.dec_lin(z)))
        for layer in self.dec:
            h_res = h
            h = self.dropout(F.relu(layer(h)))
            if self.res:
                h = h + h_res
        recon = self.out_lin(h)
        if self.use_log_space:
            recon = F.softplus(recon)

        return recon, mu, log_var

    @classmethod
    def build(cls, config: LinearVAEModelConfig) -> LinearVAE:
        return cls(
            inp_dim=config["inp_dim"],
            hidden_dim=config.get("hidden_dim", 128),
            dist_dim=config.get("dist_dim", 20),
            n_enc=config.get("n_enc", 2),
            n_dec=config.get("n_dec", 2),
            res=config.get("res", False),
            dropout=config.get("dropout", 0.2),
            use_log_space=config.get("use_log_space", False),
        )


class LinearModel(nn.Module):
    def __init__(
        self,
        inp_dim,
        out_dim,
        hidden_dim=128,
        n_hidden_layers=2,
        dropout=0.0,
        dropout_input=0.0,
        use_log_space=False,
        use_batch_norm=False,
    ):
        super().__init__()

        self.use_log_space = use_log_space
        self.inp_lin = nn.Linear(inp_dim, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        )
        self.out_lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout_input = nn.Dropout(dropout_input)
        self.batch_norm = nn.BatchNorm1d(inp_dim) if use_batch_norm else nn.Identity()

    def forward(self, x, mask=None):
        # mask: (B, inp_dim) bool tensor, True = masked position
        if mask is not None:
            x = x.masked_fill(mask, 0.0)
        x = self.dropout_input(self.batch_norm(x))
        x = self.dropout(F.relu(self.inp_lin(x)))
        for layer in self.encoder_layers:
            x = self.dropout(F.relu(layer(x)))
        x = self.out_lin(x)
        if self.use_log_space:
            x = F.relu(x)
        return x

    @classmethod
    def build(cls, config: LinearModelConfig) -> LinearModel:
        return cls(
            inp_dim=config["inp_dim"],
            out_dim=config["out_dim"],
            hidden_dim=config.get("hidden_dim", 128),
            n_hidden_layers=config.get("n_hidden_layers", 2),
            dropout=config.get("dropout", 0.0),
            dropout_input=config.get("dropout_input", 0.0),
            use_log_space=config.get("use_log_space", False),
            use_batch_norm=config.get("use_batch_norm", False),
        )


class FourierFeatureEncoder(nn.Module):
    def __init__(self, d_model, sigma=1.0):
        super().__init__()
        self.d_model = d_model
        # Fixed random frequencies sampled from a Gaussian distribution
        B = torch.randn(1, d_model // 2) * sigma
        self.register_buffer("B", B)

    def forward(self, v):
        # v: (B, S, 1) -> (B, S, d_model)
        projected = 2 * math.pi * v @ self.B
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class AttentionModel(nn.Module):
    def __init__(
        self,
        num_omics=74_002,
        n_heads=8,
        head_dim=64,
        n_hidden_layers=4,
        dropout=0.1,
        causal=False,
    ):
        super().__init__()

        hidden_dim = n_heads * head_dim
        self.num_omics = num_omics
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.causal = causal

        # Value embedding: scalar per omic -> hidden_dim via random Fourier features
        self.inp_proj = FourierFeatureEncoder(hidden_dim, sigma=1.0)

        # Omic identity embedding: each position gets a learned embedding
        self.omic_embed = nn.Embedding(num_omics, hidden_dim)

        # Learned [MASK] token embedding — replaces value embedding at masked positions
        self.mask_emb = nn.Parameter(torch.randn(hidden_dim))

        self.layers = nn.ModuleList(
            [AttentionBlock(n_heads, head_dim, dropout) for _ in range(n_hidden_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection: hidden_dim -> scalar per omic via MLP
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        # x: (B, num_omics)
        # mask: (B, num_omics) bool tensor, True = masked position
        x = rearrange(x, "B N -> B N 1")

        # Value embeddings: (B, num_omics, hidden_dim)
        value_emb = self.inp_proj(x)

        # Replace value embeddings at masked positions with a learned [MASK] embedding
        if mask is not None:
            value_emb = torch.where(
                rearrange(mask, "B N -> B N 1"), self.mask_emb, value_emb
            )

        # Omic identity embeddings: (num_omics, hidden_dim) -> broadcast to (B, num_omics, hidden_dim)
        omic_ids = torch.arange(self.num_omics, device=x.device)
        omic_emb = self.omic_embed(omic_ids)

        # Masked positions still receive their omic identity, so the model
        # knows *which* omic to predict, just not the value
        x = value_emb + omic_emb

        for layer in self.layers:
            x = layer(x, causal=self.causal)

        x = self.norm(x)

        # Project back to scalar per omic: (B, num_omics, 1) -> (B, num_omics)
        x = rearrange(self.out_proj(x), "B N 1 -> B N")
        return x

    @classmethod
    def build(cls, config: AttentionModelConfig) -> AttentionModel:
        return cls(
            num_omics=config.get("num_omics", 74_002),
            n_heads=config.get("n_heads", 8),
            head_dim=config.get("head_dim", 64),
            n_hidden_layers=config.get("n_hidden_layers", 4),
            dropout=config.get("dropout", 0.1),
            causal=config.get("causal", False),
        )


class AttentionBlock(nn.Module):
    def __init__(self, n_heads, head_dim, dropout=0.1):
        super().__init__()

        hidden_dim = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal=False):
        # Pre-norm self-attention with residual
        h = self.norm1(x)

        qkv = rearrange(
            self.qkv_proj(h),
            "B S (three H D) -> three B S H D",
            three=3,
            H=self.n_heads,
        )
        q, k, v = unpack(qkv, [[], [], []], "* B S H D")  # each: (B, S, H, D)

        # FlashAttention-4: expects (B, S, H, D), returns (B, S, H, D)
        attn_out = flash_attn_func(q, k, v, causal=causal)[0]
        attn_out = rearrange(attn_out, "B S H D -> B S (H D)")

        x = x + self.out_proj(attn_out)

        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x
