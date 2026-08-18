"""
cortex/sections/imagination.py
Section I : L'Imagination

S'exécute en continu tout le long du pipeline. Construit et affine
progressivement une « image mentale » interne au fur et à mesure
que les données arrivent des différentes sections (CO, RF, etc.).

Mécanisme :
    - L'image interne est initialisée à zéros au début de chaque inférence.
    - Chaque appel à update() intègre de nouvelles données via une
      attention croisée (image ← données), un gate d'intégration,
      et un réseau d'affinage.
    - verify() compare l'image à une référence pour évaluer la cohérence.
    - get_image() retourne l'état actuel de l'image.
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig


class Imagination(nn.Module):
    """Section I — Construction continue de l'image mentale.

    L'imagination reçoit des mises à jour des différentes sections
    et construit progressivement une représentation visuelle interne.
    Elle fonctionne comme un accumulateur à gate contrôlé.
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        d = config.d_embed
        d_h = config.d_hidden
        dropout = config.dropout

        # --- Attention croisée : image actuelle ← nouvelles données ---
        # Query = image interne, Key/Value = nouvelles données
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --- Gate d'intégration (contrôle le mélange) ---
        # Entrée : concat(image, attended) → coefficient σ ∈ [0, 1]
        self.gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.Sigmoid(),
        )

        # --- Réseau d'affinage (FFN profond) ---
        self.refiner = nn.Sequential(
            nn.Linear(d, d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, d),
        )

        # --- Vérificateur de cohérence ---
        # Compare l'image avec une référence → score ∈ [0, 1]
        self.verifier_net = nn.Sequential(
            nn.Linear(d * 2, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
            nn.Sigmoid(),
        )

        # --- Normalisation ---
        self.layer_norm = nn.LayerNorm(d)

        # --- État interne (initialisé par reset()) ---
        self._internal_image = None

    def reset(self, batch_size: int, seq_len: int, device: torch.device):
        """Réinitialise l'image mentale interne à zéros.

        Doit être appelé au début de chaque nouvelle inférence.

        Args:
            batch_size : taille du batch
            seq_len    : longueur de la séquence
            device     : device cible (cpu ou cuda)
        """
        d = self.layer_norm.normalized_shape[0]  # récupère d_embed depuis LayerNorm
        self._internal_image = torch.zeros(
            batch_size, seq_len, d,
            device=device,
        )

    def update(self, new_data: torch.Tensor, source: str) -> torch.Tensor:
        """Intègre de nouvelles données dans l'image mentale.

        Processus :
            1. Attention croisée : l'image « regarde » les nouvelles données
            2. Gate : contrôle combien d'information nouvelle est intégrée
            3. Mélange : interpolation gated entre l'image et les données attendues
            4. Affinage : passe dans un FFN pour lisser la représentation
            5. Normalisation : stabilise les valeurs

        Args:
            new_data : (B, T, D) — données de la section source
            source   : str — identifiant de la section ('CO', 'RF', etc.)

        Returns:
            image : (B, T, D) — image mentale mise à jour
        """
        assert self._internal_image is not None, \
            "Imagination.reset() doit être appelé avant update()."

        image = self._internal_image

        # 1. Attention croisée : Query = image, Key = Value = new_data
        attended, _ = self.cross_attn(image, new_data, new_data)  # (B, T, D)

        # 2. Gate : concat(image, attended) → coefficient d'intégration
        gate_input = torch.cat([image, attended], dim=-1)         # (B, T, 2D)
        g = self.gate(gate_input)                                  # (B, T, D)

        # 3. Mélange gated : interpolation image ← attended
        #    Si g ≈ 1 : on intègre fortement les nouvelles données
        #    Si g ≈ 0 : on conserve l'image existante
        image = image + g * (attended - image)                    # (B, T, D)

        # 4. Affinage + résiduel
        refined = self.refiner(image)                              # (B, T, D)
        image = image + refined                                    # (B, T, D)

        # 5. Normalisation
        image = self.layer_norm(image)                             # (B, T, D)

        # Sauvegarde de l'état interne
        self._internal_image = image
        return image

    def verify(self, reference_data: torch.Tensor) -> torch.Tensor:
        """Vérifie la cohérence de l'image mentale par rapport à une référence.

        Args:
            reference_data : (B, T, D) — données de référence (intention, réflexion, etc.)

        Returns:
            coherence : (B, T, 1) — score de cohérence ∈ [0, 1]
        """
        assert self._internal_image is not None, \
            "Imagination.reset() doit être appelé avant verify()."

        # Concat image + référence → vérificateur
        combined = torch.cat([self._internal_image, reference_data], dim=-1)  # (B, T, 2D)
        coherence = self.verifier_net(combined)                                # (B, T, 1)
        return coherence

    def get_image(self) -> torch.Tensor:
        """Retourne l'image mentale interne actuelle.

        Returns:
            image : (B, T, D) — état actuel de l'image mentale
        """
        assert self._internal_image is not None, \
            "Imagination.reset() doit être appelé avant get_image()."
        return self._internal_image
