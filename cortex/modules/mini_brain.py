"""
CORTEX — Mini-cerveaux de vérification.

Section 4.3 du document :
    « Après chaque transformer décodeur de niveau, un ensemble de mini-cerveaux
    de vérification interagit avec la sortie locale. Ces mini-cerveaux effectuent
    une évaluation contextuelle, une validation des hypothèses et une correction
    des représentations avant que les données ne remontent vers le niveau supérieur. »

Calcule :
    H_check_{n,i} : note de cohérence (scalaire)
    Q_{n,i}       : confiance (scalaire dans [0, 1])
    V_{n,i}       : vecteur de correction (même dimension que Z)

Sortie corrigée :
    Z_corrected = Z_{n,i} + V_{n,i} * (1 - Q_{n,i})
    → Plus la confiance est basse, plus la correction est forte.
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig


class MiniBrainVerifier(nn.Module):
    """Mini-cerveau de vérification post-décodeur.

    Simule les circuits d'évaluation interne du cerveau reproduit
    pour garantir la cohérence des sorties avant agrégation.
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        d = config.d_embed

        # ── Cohérence : H_check ∈ [0, 1] ───────────────────────────────
        self.coherence_net = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
            nn.Sigmoid(),
        )

        # ── Confiance : Q ∈ [0, 1] ─────────────────────────────────────
        self.confidence_net = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
            nn.Sigmoid(),
        )

        # ── Vecteur de correction : V ∈ R^D ────────────────────────────
        self.correction_net = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, T, D) — sortie brute du transformer décodeur.

        Returns:
            z_corrected: (B, T, D) — sortie corrigée
            h_check:     (B, T, 1) — score de cohérence
            q:           (B, T, 1) — score de confiance
            v:           (B, T, D) — vecteur de correction
        """
        h_check = self.coherence_net(z)    # (B, T, 1)
        q = self.confidence_net(z)          # (B, T, 1)
        v = self.correction_net(z)          # (B, T, D)

        # Correction pondérée par l'incertitude
        z_corrected = z + v * (1.0 - q)

        return z_corrected, h_check, q, v
