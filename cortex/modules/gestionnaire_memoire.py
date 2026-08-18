"""
cortex/modules/gestionnaire_memoire.py
Orchestre la mémoire à long terme de CORTEX : Compresseur (recherche O(k)
via l'index) + Vérificateur Web + Index Sémantique + Relations multi-sauts.

Point d'entrée unique : apprendre_async(texte, vecteur).
Tourne TOUJOURS en tâche de fond (thread séparé) — ne ralentit jamais une
réponse à l'utilisateur, même quand le Vérificateur Web interroge Wikipedia.
"""

import os
import re
import threading

import torch
import torch.nn.functional as F

from cortex.modules.index_semantique import IndexSemantique
from cortex.modules.verificateur_web import VerificateurWeb

SEUIL_FUSION = 0.95  # au-delà, on considère que c'est le même concept
SEUIL_RELATION = 0.55  # entre ce seuil et SEUIL_FUSION : concepts distincts mais liés

_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "est", "sont", "il", "elle", "que", "qui", "par", "sur", "pour",
    "ce", "cette", "ces", "je", "tu", "on", "nous", "vous", "ils",
    "the", "a", "an", "is", "are", "of", "in", "on", "at", "to",
}


def _categorie_depuis_texte(texte: str) -> str:
    """Dérive une catégorie simple à partir du texte (mot le plus long,
    hors mots vides). Heuristique volontairement légère pour rester rapide."""
    mots = re.findall(r"\b[a-zA-ZÀ-ÿ]{4,}\b", texte.lower())
    mots = [m for m in mots if m not in _STOPWORDS]
    if not mots:
        return "GENERAL"
    return max(mots, key=len).upper()


class GestionnaireMemoire:
    """Point d'entrée unique pour apprendre une nouvelle connaissance,
    de façon totalement asynchrone (ne bloque jamais l'appelant)."""

    def __init__(self, base_path: str = None):
        if base_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_path = os.path.join(base, "brain", "memoire")
        self.index = IndexSemantique(base_path)
        self.verificateur = VerificateurWeb()

    def apprendre_async(self, texte: str, vecteur: torch.Tensor):
        """Lance l'apprentissage en arrière-plan. Retourne immédiatement —
        aucun appel réseau ni calcul lourd ne se fait dans ce thread-ci."""
        if not texte or not texte.strip():
            return
        thread = threading.Thread(
            target=self._apprendre,
            args=(texte, vecteur.detach().cpu()),
            daemon=True,
        )
        thread.start()

    def _apprendre(self, texte: str, vecteur: torch.Tensor):
        """Wrapper de securite : ce thread tourne en arriere-plan, sans
        supervision. Sans ce try/except, la moindre erreur transitoire
        (panne reseau du verificateur, etc.) remontait comme une
        exception non geree dans le thread - invisible, sans aucune
        trace utile (observe en pratique : "Exception in thread
        Thread-N (_apprendre)"). Desormais, l'echec de CET apprentissage
        est simplement journalise, sans jamais rien casser d'autre."""
        try:
            self._apprendre_impl(texte, vecteur)
        except Exception as e:
            print(f"[MEMOIRE] Echec de l'apprentissage pour un texte (ignore) : {e}")

    def _apprendre_impl(self, texte: str, vecteur: torch.Tensor):
        categorie = _categorie_depuis_texte(texte)
        candidats = self.index.rechercher([categorie.lower()])

        candidats_a_lier = []  # (concept_id, force) — concepts proches mais distincts

        if candidats:
            matrice, ids_valides = self.index.charger_matrice(candidats)
            if len(ids_valides) > 0:
                sims = F.cosine_similarity(vecteur.unsqueeze(0), matrice)
                meilleur_sim, meilleur_idx = sims.max(dim=0)
                if meilleur_sim.item() > SEUIL_FUSION:
                    # Concept déjà connu → FUSION (compression infinie),
                    # pas besoin de revérifier les sources, on renforce juste.
                    concept_id = ids_valides[meilleur_idx.item()]
                    self.index.fusionner(concept_id, vecteur, score_nouveau=1.0)
                    self._verifier_garde_fou_si_besoin(concept_id)
                    return

                # Concepts proches (mais pas assez pour fusionner) → à relier
                for i, cid in enumerate(ids_valides):
                    sim_i = sims[i].item()
                    if SEUIL_RELATION <= sim_i < SEUIL_FUSION:
                        candidats_a_lier.append((cid, sim_i))

        # Nouveau concept → vérification par les sources (Wikipedia FR/EN),
        # puis indexation. C'est la seule étape qui touche le réseau.
        verdict = self.verificateur.verifier(texte)
        type_noeud = "racine" if (verdict["confirme"] and not candidats) else verdict["type_noeud"]

        nouveau_id = self.index.indexer(
            vecteur=vecteur,
            type_noeud=type_noeud,
            valeur=verdict["confiance"] if verdict["confirme"] else 0.5,
            tags=[categorie.lower()],
            categorie=categorie,
            texte_original=texte,
            source_url=verdict["sources"],
        )

        # Relie le nouveau concept aux concepts proches trouvés plus haut —
        # c'est ce qui permet plus tard un raisonnement multi-sauts entre
        # deux informations distinctes (voir IndexSemantique.chemin_entre).
        for concept_id, force in candidats_a_lier:
            self.index.lier_concepts(nouveau_id, concept_id, type_relation="lie", force=force)

    def _verifier_garde_fou_si_besoin(self, concept_id: str):
        """Vérifie l'invariant racine > branche après une fusion (arbitrage
        scientifique si une nuance menace de dépasser la racine)."""
        self.index.verifier_garde_fou(concept_id, self.verificateur)