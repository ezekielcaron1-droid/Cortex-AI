"""
CORTEX — Réseau feed-forward positionnel.

Couche FFN de chaque transformer décodeur (section 4.1) :
    Z_{n,i} = FeedForward(Y_{n,i}) + Y_{n,i}
"""

import torch.nn as nn

from cortex.config import CortexConfig


class FeedForward(nn.Module):
    """Réseau feed-forward positionnel avec activation GELU.

    Architecture : Linear(D → D_hidden) → GELU → Dropout → Linear(D_hidden → D) → Dropout
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_embed, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        """(B, T, D) → (B, T, D)"""
        return self.net(x)
