"""
CORTEX — Module de créativité.

Section 4.1 du document :
    « Le cerveau géant conserve des modules dédiés à la créativité,
    à la conception et à la visualisation mentale : ces modules génèrent
    des représentations internes, des métaphores et des projections
    visuelles avant que l'information ne soit traitée par le fractal réflexif. »

Ce module simule la pensée divergente du cerveau :
    - Phase divergente : exploration stochastique de l'espace latent
      (dropout élevé + injection de bruit)
    - Phase convergente : raffinage et sélection des meilleures alternatives
    - Fusion contrôlée avec l'entrée originale via un gate appris
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig


class CreativityModule(nn.Module):
    """Module de créativité inspiré du cerveau.

    Génère des représentations alternatives par exploration stochastique,
    simulant la pensée divergente puis convergente.
    Activé uniquement pour les tâches créatives (dessin, conception, etc.).
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        d = config.d_embed

        # ── Pensée divergente : exploration stochastique ────────────────
        self.divergent = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(0.3),  # Dropout élevé pour stochasticité créative
            nn.Linear(d * 2, d),
        )

        # ── Pensée convergente : raffinage des alternatives ─────────────
        self.convergent = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
        )

        # ── Gate de fusion : mélange original ↔ créatif ─────────────────
        self.blend_gate = nn.Sequential(
            nn.Linear(d * 2, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — représentations d'entrée

        Returns:
            (B, T, D) — représentations enrichies créativement
        """
        # Injection de bruit en mode entraînement (exploration)
        noise = torch.randn_like(x) * 0.1 if self.training else torch.zeros_like(x)

        # Divergence : générer des alternatives
        divergent = self.divergent(x + noise)

        # Convergence : raffiner
        convergent = self.convergent(divergent)

        # Fusion contrôlée avec l'original
        blend = self.blend_gate(torch.cat([x, convergent], dim=-1))
        x_creative = x + blend * convergent

        return self.layer_norm(x_creative)
