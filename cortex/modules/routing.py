"""
CORTEX — Routing intelligent par complexité de token.

Section 4.3 du document :
    « Le modèle central intègre un mécanisme de routing intelligent et rapide :
    chaque mot ou token est évalué en temps réel pour lui attribuer un score
    de complexité Cx(token). »

    - Mots simples ("le", "de") → chemin court, niveaux peu profonds.
    - Séquences complexes / nouvelles → descente profonde dans la hiérarchie fractale.

Le routage est piloté par :
    Cx(token) : score de complexité ∈ [0, 1]
    Q_{n,i}   : métriques de confiance
    M_cache   : cache des complexités pré-calculées
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig


class ComplexityScorer(nn.Module):
    """Attribue un score de complexité Cx(token) à chaque token.

    Cx ∈ [0, 1] :
        0 → token très simple / fréquent  (chemin fractal court)
        1 → token très complexe / nouveau (descente profonde)
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.thresholds = config.complexity_thresholds
        self.n_levels = config.n_levels

        self.scorer = nn.Sequential(
            nn.Linear(config.d_embed, config.d_embed // 4),
            nn.GELU(),
            nn.Linear(config.d_embed // 4, config.d_embed // 8),
            nn.GELU(),
            nn.Linear(config.d_embed // 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_embed: (B, T, D) — embeddings d'entrée

        Returns:
            cx: (B,) — un seul score de complexité par prompt (pas par token).
                Le prompt entier est d'abord réduit à un vecteur unique
                (moyenne sur T), puisque le traducteur a déjà transformé
                le texte en une représentation numérique globale.
        """
        pooled = x_embed.mean(dim=1)          # (B, D) — un vecteur par prompt
        return self.scorer(pooled).squeeze(-1)  # (B,)

    def get_max_depth(
        self,
        cx_scores: torch.Tensor,
        override_level: int | None = None,
        cascade: bool = True,
    ) -> torch.Tensor:
        """Détermine la profondeur fractale pour tout le prompt (un seul
        niveau final par séquence, plus par token).

        Args:
            cx_scores: (B,) — score de complexité global du prompt
            override_level: si fourni, force ce niveau (1 à n_levels) au
                lieu de le calculer automatiquement à partir du score.
                Permet d'utiliser n'importe quel étage indépendamment.
            cascade: si True (défaut), le score automatique peut quand
                même pousser plus profond que override_level (celui-ci
                agit alors comme un plancher minimum). Si False, la
                profondeur reste strictement figée à override_level —
                aucune descente automatique supplémentaire, même si le
                score suggérerait d'aller plus loin. Ignoré si
                override_level est None.

        Returns:
            depths: (B,) — profondeur finale (1 à n_levels)
        """
        auto_depth = torch.ones_like(cx_scores, dtype=torch.long)
        for threshold in self.thresholds:
            auto_depth = auto_depth + (cx_scores > threshold).long()

        if override_level is None:
            depths = auto_depth
        elif cascade:
            # override_level = plancher minimum, le score peut pousser plus loin
            forced = torch.full_like(auto_depth, fill_value=override_level)
            depths = torch.maximum(forced, auto_depth)
        else:
            # Niveau figé : pas de descente automatique supplémentaire
            depths = torch.full_like(auto_depth, fill_value=override_level)

        return depths.clamp(min=1, max=self.n_levels)


class TokenCache:
    """Cache M_cache pour les complexités pré-calculées.

    « Si le mot a déjà été rencontré, le système utilise la complexité
    pré-calculée pour déterminer immédiatement le chemin fractal le plus
    adapté. »
    """

    def __init__(self, max_size: int = 100_000):
        self.cache: dict[int, float] = {}
        self.max_size = max_size

    def get(self, token_id: int) -> float | None:
        """Retourne la complexité en cache, ou None si inconnue."""
        return self.cache.get(token_id)

    def put(self, token_id: int, complexity: float) -> None:
        """Enregistre la complexité d'un token."""
        if len(self.cache) < self.max_size:
            self.cache[token_id] = complexity

    def batch_lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Recherche en batch. Retourne -1.0 pour les tokens non cachés.

        Args:
            token_ids: (B, T) — identifiants de tokens

        Returns:
            cached_cx: (B, T) — complexités (-1.0 si absent du cache)
        """
        result = torch.full_like(token_ids, fill_value=-1.0, dtype=torch.float)
        for b in range(token_ids.shape[0]):
            for t in range(token_ids.shape[1]):
                tid = token_ids[b, t].item()
                if tid in self.cache:
                    result[b, t] = self.cache[tid]
        return result

    def batch_update(self, token_ids: torch.Tensor, cx_scores: torch.Tensor) -> None:
        """Met à jour le cache en batch."""
        for b in range(token_ids.shape[0]):
            for t in range(token_ids.shape[1]):
                tid = token_ids[b, t].item()
                self.put(tid, cx_scores[b, t].item())

    def __len__(self) -> int:
        return len(self.cache)
