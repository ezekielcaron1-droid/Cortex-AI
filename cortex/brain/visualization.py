"""
CORTEX — Module de visualisation mentale.

Simule la capacité du cerveau humain à former des images mentales,
des projections spatiales et du raisonnement visuel.

Utilise une attention croisée entre l'espace linguistique et un espace
« visuel » interne projeté, permettant au modèle de « voir » mentalement
ce qu'il manipule (schémas, formes, relations spatiales).
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig


class VisualizationModule(nn.Module):
    """Module de visualisation mentale.

    Crée des représentations internes « visuelles » par projection spatiale
    et attention croisée. Activé pour les tâches nécessitant du raisonnement
    spatial (ASCII art, schémas, relations géométriques, etc.).
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        d = config.d_embed

        # ── Projection spatiale : espace linguistique → espace « visuel » ──
        self.spatial_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # ── Attention croisée : linguistique × visuel ──────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # ── Projection de sortie ───────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — représentations d'entrée

        Returns:
            (B, T, D) — représentations enrichies visuellement
        """
        # Créer la projection « visuelle » interne
        visual = self.spatial_proj(x)

        # Attention croisée : le linguistique attend le visuel
        attended, _ = self.cross_attn(query=x, key=visual, value=visual)

        # Projection et connexion résiduelle
        out = self.out_proj(attended)
        return self.layer_norm(x + out)
