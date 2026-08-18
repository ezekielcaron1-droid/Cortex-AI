"""
CORTEX — Attention multi-tête causale.

Implémente le mécanisme d'attention décrit en section 4.3 :
    Y_{n,i} = Attention(X_{n,i}, X_{n,i}, X_{n,i}) + X_{n,i}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.config import CortexConfig


class MultiHeadSelfAttention(nn.Module):
    """Attention multi-tête causale (self-attention du décodeur).

    Chaque couche d'attention dans les modules transformer du fractal
    utilise cette implémentation avec masque causal automatique.
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_embed = config.d_embed

        # Projections Q, K, V
        self.q_proj = nn.Linear(config.d_embed, config.d_embed)
        self.k_proj = nn.Linear(config.d_embed, config.d_embed)
        self.v_proj = nn.Linear(config.d_embed, config.d_embed)

        # Projection de sortie
        self.out_proj = nn.Linear(config.d_embed, config.d_embed)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D_embed) — tenseur d'entrée
            mask: (T, T) optionnel — masque booléen (True = masqué)

        Returns:
            (B, T, D_embed) — sortie de l'attention
        """
        B, T, _ = x.shape

        # Projections → (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scores d'attention : QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Masque causal (empêche de voir le futur)
        if mask is None:
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Sortie pondérée
        out = torch.matmul(attn_weights, v)                          # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_embed)  # (B, T, D)
        out = self.resid_dropout(self.out_proj(out))

        return out
