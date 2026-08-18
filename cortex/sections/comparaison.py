"""
cortex/sections/comparaison.py
Section CA : La Comparaison

Valide la cohérence du travail du cerveau. Compare le prompt d'origine
avec les sorties intermédiaires (CO, RF, I) pour détecter les erreurs
et génère un vecteur de feedback correctif.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ComparaisonResult:
    similarite: torch.Tensor       # (B,)
    is_valid: torch.Tensor         # (B,) bool
    scores_sections: torch.Tensor  # (B, 3) -> [CO, RF, I]
    error_source: torch.Tensor     # (B,) index : 0=CO, 1=RF, 2=I
    feedback: torch.Tensor         # (B, T, D)


class Comparaison(nn.Module):
    """
    Compare l'intention d'origine au résultat actuel et localise les erreurs
    parmi les sections CO, RF et I.
    """

    def __init__(self, d_model: int = 512, seuil: float = 0.75):
        super().__init__()
        self.d_model = d_model
        self.seuil = seuil

        # --- Comparateurs par section (concat 2*D -> MLP -> Sigmoid) ---
        def _make_comparateur():
            return nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

        self.compare_co = _make_comparateur()
        self.compare_rf = _make_comparateur()
        self.compare_i = _make_comparateur()

        # --- Évaluateur global ---
        self.overall_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # --- Générateur de feedback (concat 2*D -> Linear -> Tanh) ---
        self.feedback_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )

    def forward(
        self,
        intention: torch.Tensor,       # (B, T, D)
        resultat_actuel: torch.Tensor, # (B, T, D)
        sortie_co: torch.Tensor,       # (B, T, D)
        sortie_rf: torch.Tensor,       # (B, T, D)
        sortie_i: torch.Tensor,        # (B, T, D)
    ) -> ComparaisonResult:

        # --- Étape 1 : Pooling temporel (moyenne sur T) ---
        intention_m = intention.mean(dim=1)             # (B, D)
        resultat_m = resultat_actuel.mean(dim=1)        # (B, D)
        co_m = sortie_co.mean(dim=1)                    # (B, D)
        rf_m = sortie_rf.mean(dim=1)                    # (B, D)
        i_m = sortie_i.mean(dim=1)                      # (B, D)

        # --- Étape 2 : Comparaison par section ---
        score_co = self.compare_co(torch.cat([intention_m, co_m], dim=-1))  # (B, 1)
        score_rf = self.compare_rf(torch.cat([intention_m, rf_m], dim=-1))  # (B, 1)
        score_i = self.compare_i(torch.cat([intention_m, i_m], dim=-1))     # (B, 1)

        scores_sections = torch.cat([score_co, score_rf, score_i], dim=-1)  # (B, 3)

        # --- Étape 3 : Localisation de l'erreur (argmin) ---
        error_source = torch.argmin(scores_sections, dim=-1)  # (B,)

        # --- Étape 4 : Score global & validation ---
        global_input = torch.cat([intention_m, resultat_m], dim=-1)  # (B, 2*D)
        similarite = self.overall_scorer(global_input).squeeze(-1)   # (B,)
        is_valid = similarite >= self.seuil                          # (B,) bool

        # --- Étape 5 : Vecteur de feedback ---
        # Concat sur la dimension feature, conserve la dimension temporelle.
        feedback_input = torch.cat([intention, resultat_actuel], dim=-1)  # (B, T, 2*D)
        feedback = self.feedback_net(feedback_input)                      # (B, T, D)

        return ComparaisonResult(
            similarite=similarite,
            is_valid=is_valid,
            scores_sections=scores_sections,
            error_source=error_source,
            feedback=feedback,
        )
