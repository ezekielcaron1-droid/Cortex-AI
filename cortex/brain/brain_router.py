"""
CORTEX — Routeur des modules cerveau.

Analyse l'entrée pour déterminer quels modules du « grand cerveau »
doivent être activés :
    - CreativityModule  → tâches créatives (conception, invention)
    - VisualizationModule → tâches visuelles (schémas, ASCII art, formes)
    - ConceptualModule  → tâches abstraites (raisonnement, métaphores)

Logique (requise par l'utilisateur) :
    - Entrée simple ("bonjour") → AUCUN module cerveau activé
    - Entrée complexe ("concevoir un schéma en ASCII art avec des ronds")
      → modules créativité + visualisation activés
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig


class BrainRouter(nn.Module):
    """Routeur intelligent des modules cerveau.

    Analyse le contenu global de la séquence d'entrée pour
    décider quels modules cerveau activer. Chaque module reçoit
    un score d'activation ∈ [0, 1].
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        d = config.d_embed
        self.threshold = config.brain_activation_threshold

        # ── Réseau de routage : produit 3 scores (créativité, vision, concept) ──
        self.router = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, D) — embeddings d'entrée

        Returns:
            scores:      (B, 3) — scores d'activation [créativité, vision, concept]
            active_mask: (B, 3) — masque booléen des modules actifs
        """
        # Pooling global sur la séquence pour une décision par échantillon
        x_pooled = x.mean(dim=1)             # (B, D)

        # Scores d'activation par module
        scores = self.router(x_pooled)        # (B, 3)

        # Masque binaire d'activation
        active_mask = scores > self.threshold  # (B, 3)

        return scores, active_mask
