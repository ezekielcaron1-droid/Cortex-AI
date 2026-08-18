"""
CORTEX — Module d'abstraction conceptuelle.

Simule la capacité du cerveau à former des concepts abstraits et des
métaphores à partir d'entrées concrètes. Réalise une attention entre
concepts pour créer des associations de haut niveau.

Section 4.7 :
    « Les cinq niveaux peuvent correspondre à cinq échelles de traitement :
    sensoriel, perceptif, conceptuel, décisionnel et métacognitif. »

Ce module intervient à l'échelle conceptuelle, avant la descente
dans la hiérarchie fractale.
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig


class ConceptualModule(nn.Module):
    """Module d'abstraction conceptuelle.

    Effectue une projection vers un espace conceptuel abstrait,
    une auto-attention au niveau des concepts, puis un décodage
    vers l'espace original. Permet au modèle de raisonner
    sur des abstractions plutôt que sur des tokens bruts.
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        d = config.d_embed

        # ── Encodeur conceptuel : concret → abstrait ───────────────────
        self.concept_encoder = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # ── Attention entre concepts ───────────────────────────────────
        self.concept_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # ── Décodeur conceptuel : abstrait → concret enrichi ──────────
        self.concept_decoder = nn.Sequential(
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
            (B, T, D) — représentations enrichies conceptuellement
        """
        # Projection vers l'espace conceptuel
        concepts = self.concept_encoder(x)

        # Auto-attention au niveau des concepts
        attended, _ = self.concept_attn(concepts, concepts, concepts)

        # Décodage vers l'espace original
        decoded = self.concept_decoder(attended)

        # Connexion résiduelle + normalisation
        return self.layer_norm(x + decoded)
