"""
CORTEX — Gestion de la mémoire et des états cachés.

Section 4.5 du document :
    « Chaque module conserve un vecteur d'état caché H_{n,i} de dimension D_embed. »

    Mise à jour :
        H_{n,i}^{t+1} = LayerNorm(H_{n,i}^{t} + A_{n,i} + R_{n,i})

    « Un mécanisme de mémoire à court terme M_{n,i} peut stocker les résultats
    intermédiaires de chaque niveau pour les réutiliser sur plusieurs itérations. »
"""

import torch
import torch.nn as nn
from collections import deque

from cortex.config import CortexConfig


class ShortTermMemory:
    """Mémoire à court terme M_{n,i}.

    Buffer circulaire stockant les états intermédiaires pour
    réutilisation sur plusieurs itérations du fractal.
    """

    def __init__(self, capacity: int = 64):
        self.buffer: deque[torch.Tensor] = deque(maxlen=capacity)

    def store(self, state: torch.Tensor) -> None:
        """Stocke un état (détaché du graphe de calcul)."""
        self.buffer.append(state.detach())

    def retrieve(self, k: int | None = None) -> list[torch.Tensor]:
        """Récupère les k derniers états (tous si k=None)."""
        items = list(self.buffer)
        return items[-k:] if k is not None else items

    def get_latest(self) -> torch.Tensor | None:
        """Retourne l'état le plus récent, ou None si vide."""
        return self.buffer[-1] if self.buffer else None

    def clear(self) -> None:
        """Vide le buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class HiddenStateManager(nn.Module):
    """Gestion des états cachés H_{n,i} avec règle de mise à jour.

    H_{n,i}^{t+1} = LayerNorm(H_{n,i}^{t} + A_{n,i} + R_{n,i})
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.d_embed = config.d_embed
        self.layer_norm = nn.LayerNorm(config.d_embed)

    def init_state(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Crée un état caché initial à zéro."""
        return torch.zeros(batch_size, seq_len, self.d_embed, device=device)

    def update(
        self,
        h_prev: torch.Tensor,
        activation: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Mise à jour de l'état caché.

        H^{t+1} = LayerNorm(H^t + A + R)

        Args:
            h_prev:     (B, T, D) — état précédent H_{n,i}^{t}
            activation: (B, T, D) — activation A_{n,i}
            residual:   (B, T, D) — retour résiduel R_{n,i}

        Returns:
            h_new: (B, T, D) — état mis à jour H_{n,i}^{t+1}
        """
        return self.layer_norm(h_prev + activation + residual)
