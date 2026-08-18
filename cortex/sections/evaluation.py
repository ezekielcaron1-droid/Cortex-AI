"""
cortex/sections/evaluation.py
Section E : L'Évaluation

S'exécute après le cycle de traitement. Fournit un bilan détaillé de la
performance. Contrairement à la comparaison (CA), elle n'est pas connectée
à la boucle de rétroaction et ne génère aucun vecteur correctif.
"""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class EvaluationResult:
    proximite: torch.Tensor          # (B,)
    completude: torch.Tensor         # (B,)
    scores_sections: Dict[str, torch.Tensor]  # {"CO": (B,), "RF": (B,), "I": (B,)}


class Evaluation(nn.Module):
    """
    Calcule des métriques de performance : proximité à la demande,
    complétude de la réponse, et qualité individuelle des sections.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model

        # --- Score de proximité (intention vs résultat final) ---
        self.proximity_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # --- Score de complétude (résultat final seul) ---
        self.completeness_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # --- Scoreurs individuels de section (concat 2*D -> MLP -> Sigmoid) ---
        def _make_scorer():
            return nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

        self.score_co = _make_scorer()
        self.score_rf = _make_scorer()
        self.score_i = _make_scorer()

    def forward(
        self,
        intention: torch.Tensor,      # (B, T, D)
        resultat_final: torch.Tensor, # (B, T, D)
        sortie_co: torch.Tensor,      # (B, T, D)
        sortie_rf: torch.Tensor,      # (B, T, D)
        sortie_i: torch.Tensor,       # (B, T, D)
    ) -> EvaluationResult:

        # --- Pooling temporel ---
        intention_m = intention.mean(dim=1)         # (B, D)
        resultat_m = resultat_final.mean(dim=1)     # (B, D)
        co_m = sortie_co.mean(dim=1)                # (B, D)
        rf_m = sortie_rf.mean(dim=1)                # (B, D)
        i_m = sortie_i.mean(dim=1)                  # (B, D)

        # --- Calcul de proximité ---
        prox_input = torch.cat([intention_m, resultat_m], dim=-1)  # (B, 2*D)
        proximite = self.proximity_scorer(prox_input).squeeze(-1)  # (B,)

        # --- Calcul de complétude ---
        completude = self.completeness_scorer(resultat_m).squeeze(-1)  # (B,)

        # --- Évaluation individuelle des sections ---
        s_co = self.score_co(torch.cat([intention_m, co_m], dim=-1)).squeeze(-1)  # (B,)
        s_rf = self.score_rf(torch.cat([intention_m, rf_m], dim=-1)).squeeze(-1)  # (B,)
        s_i = self.score_i(torch.cat([intention_m, i_m], dim=-1)).squeeze(-1)     # (B,)

        scores_sections = {"CO": s_co, "RF": s_rf, "I": s_i}

        return EvaluationResult(
            proximite=proximite,
            completude=completude,
            scores_sections=scores_sections,
        )
