"""
cortex/sections/reflexion.py
Section RF : La Réflexion

Enveloppe l'arbre fractal récursif (FractalNode) et le système de
routing intelligent par complexité (ComplexityScorer / TokenCache).

Pipeline interne :
    comprehension_output (dict) + feedback (optionnel)
        → Projection (concat meaning + context → D)
        → Ajout du feedback résiduel (si retry > 0)
        → Calcul de complexité Cx par token
        → Détermination de la profondeur fractale max
        → Descente dans l'arbre fractal (3125 modules, offloading CPU↔GPU)
        → dict {'reasoning', 'confidence', 'depth_info'}
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig
from cortex.core.fractal_node import FractalNode
from cortex.modules.routing import ComplexityScorer, TokenCache


class Reflexion(nn.Module):
    """Section RF — Réflexion fractale profonde.

    Utilise l'arbre fractal à 5 niveaux × 5 modèles (3125 modules)
    pour effectuer un raisonnement multi-échelle. La profondeur de
    traitement est adaptée dynamiquement à la complexité de chaque
    token grâce au ComplexityScorer.
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.config = config
        d = config.d_embed

        # --- Scoring de complexité par token ---
        self.complexity_scorer = ComplexityScorer(config)

        # --- Cache des complexités déjà calculées ---
        self.token_cache = TokenCache()

        # --- Racine de l'arbre fractal (instanciation paresseuse des enfants) ---
        self.fractal_root = FractalNode(config, level=0)

        # --- Projection d'entrée : concat(meaning, context) → D ---
        # meaning (B, T, D) + context (B, T, D) → (B, T, 2D) → (B, T, D)
        self.input_proj = nn.Linear(d * 2, d)

        # --- Normalisation de sortie ---
        self.output_norm = nn.LayerNorm(d)

    def forward(
        self,
        comprehension_output: dict,
        feedback: torch.Tensor = None,
        override_level: int | None = None,
        cascade: bool = True,
    ) -> dict:
        """
        Args:
            comprehension_output : dict avec 'meaning' (B,T,D), 'intent' (B,D), 'context' (B,T,D)
            feedback : (B, T, D) optionnel — vecteur correctif issu de CA (si retry > 0)
            override_level : niveau fractal forcé (1 à n_levels), au lieu du
                calcul automatique par score de complexité. None = automatique.
            cascade : si True (défaut), le score automatique peut pousser plus
                profond que override_level. Si False, le niveau reste figé
                à override_level, sans descente automatique supplémentaire.

        Returns:
            dict avec :
                'reasoning'  : (B, T, D) — résultat du raisonnement fractal
                'confidence' : float     — confiance moyenne du noeud racine
                'depth_info' : dict      — métriques de diagnostic de l'arbre fractal
        """
        meaning = comprehension_output['meaning']   # (B, T, D)
        context = comprehension_output['context']    # (B, T, D)

        # 1. Fusion meaning + context → x
        combined = torch.cat([meaning, context], dim=-1)   # (B, T, 2D)
        x = self.input_proj(combined)                       # (B, T, D)

        # 2. Injection du feedback correctif (si disponible)
        if feedback is not None:
            x = x + feedback                                # (B, T, D)

        # 3. Calcul du score de complexité global du prompt
        cx_scores = self.complexity_scorer(x)               # (B,)

        # 4. Détermination de la profondeur fractale (un seul niveau par prompt)
        max_depth = self.complexity_scorer.get_max_depth(
            cx_scores, override_level=override_level, cascade=cascade
        )  # (B,)

        # 5. Descente dans l'arbre fractal
        #    FractalNode renvoie : (output, residual, hidden_state, info)
        fractal_out, _residual, _hidden, info = self.fractal_root(
            x,
            fractal_context=context,
            hidden_state=None,
            max_depth=max_depth,
            mask=None,
        )

        # 6. Normalisation de la sortie
        reasoning = self.output_norm(fractal_out)           # (B, T, D)

        return {
            'reasoning': reasoning,
            'confidence': info.get('confidence', 1.0),
            'depth_info': info,
        }
