"""
CORTEX — Bloc transformer décodeur.

Section 4.1 du document :
    « Chaque modèle de niveau est un module transformer décodeur spécialisé,
    composé de couches d'attention multi-tête, de couches feed-forward
    positionnelles et de normalisation. »

Section 4.3 :
    Y_{n,i} = Attention(X_{n,i}, X_{n,i}, X_{n,i}) + X_{n,i}
    Z_{n,i} = FeedForward(Y_{n,i}) + Y_{n,i}
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig
from cortex.core.attention import MultiHeadSelfAttention
from cortex.core.feedforward import FeedForward


class TransformerDecoderBlock(nn.Module):
    """Bloc transformer décodeur avec pre-norm.

    Architecture (Pre-LayerNorm pour stabilité) :
        Y = X + Attention(LayerNorm(X))
        Z = Y + FeedForward(LayerNorm(Y))
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_embed)
        self.attention = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_embed)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D) — entrée
            mask: (T, T) optionnel — masque d'attention

        Returns:
            z: (B, T, D) — Z_{n,i}
        """
        # Y = X + Attention(LN(X))
        y = x + self.attention(self.ln1(x), mask)
        # Z = Y + FFN(LN(Y))
        z = y + self.feed_forward(self.ln2(y))
        return z
