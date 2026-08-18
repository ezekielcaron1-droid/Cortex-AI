"""
CORTEX v2 — Modèle complet.

Pipeline :
    input_ids (B, T)
        → TraducteurEntree  →  traduit (B, T, D) + lang_probs (B, num_langues)
        → CerveauGeneral    →  cortex_output (B, T, D) + métriques
        → TraducteurSortie  →  logits (B, T, vocab_size)

Ce fichier assemble les 3 grands blocs : T(entrée), CRVG, T(sortie).
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig
from cortex.sections.traducteur import TraducteurEntree, TraducteurSortie
from cortex.crvg.cerveau_general import CerveauGeneral


class CortexModel(nn.Module):
    """Modèle CORTEX complet v2.

    Assemblage final :
        TraducteurEntree → CerveauGeneral (CO → RF → I → CA ⟲ → E) → TraducteurSortie
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.config = config

        # ── Section T — Entrée ──────────────────────────────────────────
        self.traducteur_entree = TraducteurEntree(
            vocab_size=config.vocab_size,
            d_model=config.d_embed,
            max_seq_len=config.max_seq_len,
            num_langues=config.n_languages,
            n_heads=config.n_heads,
            ff_mult=config.d_hidden // config.d_embed,   # 2048 // 512 = 4
            n_traducteurs=config.n_translators,
        )

        # ── CRVG — Cerveau Général ──────────────────────────────────────
        self.cerveau = CerveauGeneral(config)

        # ── Section T — Sortie ──────────────────────────────────────────
        self.traducteur_sortie = TraducteurSortie(
            vocab_size=config.vocab_size,
            d_model=config.d_embed,
            num_langues=config.n_languages,
            n_heads=config.n_heads,
            ff_mult=config.d_hidden // config.d_embed,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        override_level: int | None = None,
        cascade: bool = True,
    ) -> dict:
        """
        Forward pass complet du modèle CORTEX.

        Args:
            input_ids : (B, T) — identifiants de tokens d'entrée
            override_level : niveau fractal forcé (1 à n_levels), au lieu du
                calcul automatique par complexité du prompt. None = auto.
            cascade : si True (défaut), le score de complexité peut pousser
                plus profond que override_level. Si False, la profondeur
                reste figée à override_level (utilisation d'un étage isolé).

        Returns:
            dict avec :
                'logits'        : (B, T, vocab_size) — logits de sortie
                'lang_probs'    : (B, n_languages)   — probabilités de langue détectées
                'evaluation'    : EvaluationResult    — métriques de qualité
                'retries'       : int                 — nombre d'essais CA
                'cortex_output' : (B, T, D)           — représentation interne finale
        """
        # ── 1. Traduction d'entrée ──────────────────────────────────────
        translated, lang_probs = self.traducteur_entree(input_ids)
        # translated : (B, T, D)
        # lang_probs : (B, n_languages)

        # ── 2. Traitement par le cerveau général ────────────────────────
        brain_output = self.cerveau(
            translated, override_level=override_level, cascade=cascade
        )
        # brain_output['cortex_output'] : (B, T, D)

        # ── 3. Traduction de sortie ─────────────────────────────────────
        logits = self.traducteur_sortie(
            brain_output['cortex_output'],
            lang_probs,
        )
        # logits : (B, T, vocab_size)

        return {
            'logits': logits,
            'lang_probs': lang_probs,
            'evaluation': brain_output['evaluation'],
            'retries': brain_output['retries'],
            'cortex_output': brain_output['cortex_output'],
            'invention': brain_output['invention'],
            'invention_stats': brain_output['invention_stats'],
        }

    def count_parameters(self) -> int:
        """Retourne le nombre total de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
