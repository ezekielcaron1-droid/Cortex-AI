"""
cortex/crvg/cerveau_general.py
CRVG : Le Cerveau Général

Chef d'orchestre du pipeline CORTEX. Gère le flux complet :
    CO → RF → I → INV → I.update(INV) → CA ⟲ → E → INV.record(E.score)

Avec la boucle de rétroaction :
    - Si CA détecte une incohérence (similarité < seuil), le feedback
      est renvoyé à RF (et l'imagination est mise à jour) pour un
      nouvel essai (max 3 retries).
    - L'imagination s'exécute en continu (mise à jour après CO, RF, et INV).
    - L'invention génère des idées nouvelles et apprend par l'expérience.
    - CA et E sont connectés à CO, RF, I mais PAS entre eux.

Pipeline complet du modèle :
    T(entrée) → CRVG → T(sortie)
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig
from cortex.sections.comprehension import Comprehension
from cortex.sections.reflexion import Reflexion
from cortex.sections.imagination import Imagination
from cortex.sections.invention import Invention
from cortex.sections.comparaison import Comparaison
from cortex.sections.evaluation import Evaluation
from cortex.brain import BrainRouter, CreativityModule, VisualizationModule, ConceptualModule
# HAI / Boîte Infinie : voir cortex/modules/hai_v2.py + boite_infinie.py.
# Pas encore importés/instanciés ici — branchement prévu quand CORTEX
# saura produire du texte cohérent (voir historique du 07/08/2026).


class CerveauGeneral(nn.Module):
    """Cerveau Général (CRVG) — Orchestrateur central de CORTEX.

    Encapsule les 6 sections internes (CO, RF, I, INV, CA, E) et gère
    la boucle de feedback entre CA et RF, ainsi que l'apprentissage
    par l'expérience de la section Invention.
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.config = config
        d = config.d_embed

        # --- Les 6 sections du cerveau ---
        self.comprehension = Comprehension(config)
        self.reflexion = Reflexion(config)
        self.imagination = Imagination(config)
        self.invention = Invention(config)
        self.comparaison = Comparaison(d_model=d, seuil=config.comparison_threshold)
        self.evaluation = Evaluation(d_model=d)

        # ── Modules « grand cerveau » (créativité, visualisation, concept) ──
        # Avant : codés mais jamais appelés nulle part -> jamais entraînés
        # par le gradient, même après un pré-entraînement complet. Branchés
        # ici, juste après CO, pour qu'ils participent réellement au calcul.
        self.brain_router = BrainRouter(config)
        self.creativity = CreativityModule(config)
        self.visualization = VisualizationModule(config)
        self.conceptual = ConceptualModule(config)

        # Nombre max de tentatives avant passage forcé
        self.max_retries = config.max_feedback_loops

        # ── HAI : non branchée pour l'instant (voir imports en tête de fichier) ──

    @staticmethod
    def _appliquer_si_actif(module: nn.Module, x: torch.Tensor, colonne_active: torch.Tensor) -> torch.Tensor:
        """Applique un module « grand cerveau » seulement aux échantillons
        du batch dont le score dépasse le seuil (colonne_active), pour
        respecter la logique du BrainRouter : entrée simple -> aucun
        module activé, entrée complexe -> modules pertinents activés.
        Les échantillons non concernés restent inchangés.

        Important pour l'entraînement : dès qu'AU MOINS un échantillon du
        batch active ce module, le gradient traverse bien ses poids -
        c'est ce qui garantit qu'il soit réellement entraîné, pas juste
        présent dans le code.
        """
        if not colonne_active.any():
            return x
        sortie = module(x)
        masque = colonne_active.view(-1, 1, 1)  # (B,1,1) -> broadcast sur (B,T,D)
        return torch.where(masque, sortie, x)

    def forward(
        self,
        translated_prompt: torch.Tensor,
        override_level: int | None = None,
        cascade: bool = True,
    ) -> dict:
        """
        Exécute le pipeline complet du cerveau.

        Pipeline :
            CO → I.update(CO)
            → RF → I.update(RF)
            → INV → I.update(INV)
            → CA (→ feedback → retry si échec)
            → E → INV.record(E.score)

        Args:
            translated_prompt : (B, T, D) — sortie du TraducteurEntree
            override_level : niveau fractal forcé (1 à n_levels), transmis à
                RF au lieu du calcul automatique par complexité. None = auto.
            cascade : si True (défaut), le score peut pousser plus profond que
                override_level. Si False, RF reste figé à ce niveau exact.

        Returns:
            dict avec :
                'cortex_output' : (B, T, D) — représentation finale traitée
                'evaluation'    : EvaluationResult — métriques de qualité
                'retries'       : int — nombre d'essais effectués
                'comparison'    : ComparaisonResult — dernier résultat de comparaison
                'co_output'     : dict — sortie brute de la compréhension
                'rf_output'     : dict — sortie brute de la réflexion
                'imagination'   : (B, T, D) — image mentale finale
                'invention'     : (B, T, D) — dernière idée inventée
                'invention_stats' : dict — statistiques de la banque d'expérience
        """
        B, T, D = translated_prompt.shape
        device = translated_prompt.device

        # ═══════════════════════════════════════════════════════════════
        # 1. Initialisation de l'imagination (image mentale à zéros)
        # ═══════════════════════════════════════════════════════════════
        self.imagination.reset(B, T, device)

        # Le feedback commence à None (pas de correction au 1er essai)
        feedback = None
        comparison = None
        co_out = None
        rf_out = None
        invention_out = None

        # ═══════════════════════════════════════════════════════════════
        # 2. Boucle de rétroaction (max_retries tentatives)
        # ═══════════════════════════════════════════════════════════════
        for retry in range(self.max_retries):

            # ── Étape CO : Compréhension ───────────────────────────────
            co_out = self.comprehension(translated_prompt)
            # co_out = {'meaning': (B,T,D), 'intent': (B,D), 'context': (B,T,D)}

            # ── Modules « grand cerveau » (créativité, visualisation, concept) ──
            # Entrée simple -> aucun module activé ; entrée complexe ->
            # modules pertinents activés (logique du BrainRouter).
            _, brain_mask = self.brain_router(co_out['meaning'])
            meaning_enrichi = co_out['meaning']
            meaning_enrichi = self._appliquer_si_actif(self.creativity, meaning_enrichi, brain_mask[:, 0])
            meaning_enrichi = self._appliquer_si_actif(self.visualization, meaning_enrichi, brain_mask[:, 1])
            meaning_enrichi = self._appliquer_si_actif(self.conceptual, meaning_enrichi, brain_mask[:, 2])
            co_out['meaning'] = meaning_enrichi

            # ── Mise à jour Imagination (après CO) ─────────────────────
            self.imagination.update(co_out['meaning'], source='CO')

            # ── Étape RF : Réflexion fractale ──────────────────────────
            rf_out = self.reflexion(
                co_out,
                feedback=feedback,
                override_level=override_level,
                cascade=cascade,
            )
            # rf_out = {'reasoning': (B,T,D), 'confidence': float, 'depth_info': dict}

            # ── Mise à jour Imagination (après RF) ─────────────────────
            self.imagination.update(rf_out['reasoning'], source='RF')

            # ── Étape INV : Invention ──────────────────────────────────
            # Génère une idée nouvelle via sélection darwinienne
            # à partir des sorties de CO, RF et I
            invention_out = self.invention(
                co_output=co_out['meaning'],
                rf_output=rf_out['reasoning'],
                i_output=self.imagination.get_image(),
            )

            # ── Mise à jour Imagination (après INV) ────────────────────
            # L'invention enrichit l'image mentale avec des idées nouvelles
            self.imagination.update(invention_out, source='INV')

            # ── Étape CA : Comparaison ─────────────────────────────────
            # L'intention est expansée de (B, D) à (B, T, D) pour la comparaison
            intent_expanded = co_out['intent'].unsqueeze(1).expand(B, T, D)

            comparison = self.comparaison(
                intention=intent_expanded,
                resultat_actuel=rf_out['reasoning'],
                sortie_co=co_out['meaning'],
                sortie_rf=rf_out['reasoning'],
                sortie_i=self.imagination.get_image(),
            )

            # ── Décision : valide ou on recommence ? ───────────────────
            if comparison.is_valid.all():
                # Toutes les séquences du batch sont valides → on sort
                break

            if retry < self.max_retries - 1:
                # On récupère le feedback correctif pour le prochain essai
                feedback = comparison.feedback
                # Reset imagination pour le prochain essai
                self.imagination.reset(B, T, device)

        # ═══════════════════════════════════════════════════════════════
        # 3. Étape E : Évaluation finale (hors boucle)
        # ═══════════════════════════════════════════════════════════════
        intent_expanded = co_out['intent'].unsqueeze(1).expand(B, T, D)
        eval_result = self.evaluation(
            intention=intent_expanded,
            resultat_final=rf_out['reasoning'],
            sortie_co=co_out['meaning'],
            sortie_rf=rf_out['reasoning'],
            sortie_i=self.imagination.get_image(),
        )

        # ═══════════════════════════════════════════════════════════════
        # 4. Feedback vers INV : apprentissage par l'expérience
        # ═══════════════════════════════════════════════════════════════
        # Le score de proximité de E est renvoyé à l'Invention
        # pour qu'elle apprenne ce qui a bien/mal marché.
        eval_score = eval_result.proximite.mean().item()
        self.invention.record_feedback(eval_score)

        # ═══════════════════════════════════════════════════════════════
        # 4b. HAI : non branchée pour l'instant (voir imports en tête de fichier)
        # ═══════════════════════════════════════════════════════════════

        # ═══════════════════════════════════════════════════════════════
        # 5. Sortie finale
        # ═══════════════════════════════════════════════════════════════
        # La sortie du cerveau combine le raisonnement, l'imagination
        # et l'invention pour une représentation enrichie
        cortex_output = rf_out['reasoning'] + self.imagination.get_image()

        return {
            'cortex_output': cortex_output,
            'evaluation': eval_result,
            'retries': retry + 1,
            'comparison': comparison,
            'co_output': co_out,
            'rf_output': rf_out,
            'imagination': self.imagination.get_image(),
            'invention': invention_out,
            'invention_stats': self.invention.get_stats(),
        }
