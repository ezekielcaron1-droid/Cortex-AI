"""
cortex/modules/index_semantique.py
Index sémantique de la mémoire de CORTEX.

Rôle : Naviguer dans les connaissances via des mots-clés et synonymes.
Structure sur disque :
    cortex/brain/memoire/
    ├── sommaire.json          → { "CODE.PYTHON.AFFICHAGE": ["id1", "id2", ...] }
    ├── synonymes.json         → { "afficher": ["print", "output", "log"] }
    ├── concepts_vecteurs.pt   → { "id1": Tensor, "id2": Tensor, ... } (TOUS les concepts)
    └── concepts_meta.json     → { "id1": {...}, "id2": {...}, ... }  (TOUTES les metadonnees)

NOTE : avant, chaque concept creait 2 fichiers separes ({id}.pt +
{id}.json) - avec des centaines/milliers de concepts sur une session
longue, ca finissait par exploser en un nombre ingerable de petits
fichiers (lent a lister, mauvais pour les sauvegardes, limites pratiques
du systeme de fichiers). Desormais, TOUS les concepts vivent dans 2
fichiers uniques, charges en RAM au demarrage (comme sommaire/synonymes
deja) et reecrits entierement a chaque modification - largement
suffisant tant que le nombre de concepts reste de l'ordre de quelques
milliers. Si ca devait un jour monter a des dizaines de milliers, il
faudrait passer a une strategie d'ecriture differee (flush periodique
au lieu d'un rewrite complet a chaque appel) plutot que revenir a un
fichier par concept.

Recherche ultra-rapide : O(k) au lieu de O(n) car on ne touche
qu'au secteur sémantique pertinent (k << n).
"""

import os
import json
import threading
import uuid
import torch
import torch.nn.functional as F
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
#  SEUILS DE L'AUTO-ENRICHISSEMENT DES SYNONYMES (3 conditions)
# ═══════════════════════════════════════════════════════════════════

SEUIL_SIM_SYNONYME     = 0.92   # Similarité cosinus minimum entre deux vecteurs
SEUIL_FREQ_SYNONYME    = 50     # Chaque mot doit avoir été vu au moins 50 fois
SEUIL_COOCCURRENCE     = 0.80   # Co-occurrence dans 80% des contextes communs


# ═══════════════════════════════════════════════════════════════════
#  POIDS PAR TYPE DE NŒUD (Invariant : racine > branche > feuille)
# ═══════════════════════════════════════════════════════════════════

POIDS_INITIAL = {
    "racine":  1.0,   # Connaissance certifiée
    "branche": 0.5,   # Critique sourcée
    "feuille": 0.2,   # Critique sans source
}

# Seuil d'alerte : si nuance.poids > racine.poids * ce ratio → arbitrage
SEUIL_ALERTE_CONFLIT = 0.80


class IndexSemantique:
    """
    Index sémantique : sommaire + synonymes + recherche rapide.

    L'index est chargé en RAM au démarrage (légère JSON) et sauvegardé
    sur disque après chaque modification. Les tenseurs des vecteurs restent
    sur disque et sont chargés uniquement quand nécessaires.
    """

    def __init__(self, base_path: str):
        self.base_path    = Path(base_path)
        self.concepts_dir = self.base_path / "concepts"  # conserve pour l'archive de migration
        self.sommaire_path = self.base_path / "sommaire.json"
        self.synonymes_path = self.base_path / "synonymes.json"
        self.relations_path = self.base_path / "relations.json"
        self.vecteurs_path = self.base_path / "concepts_vecteurs.pt"
        self.meta_path = self.base_path / "concepts_meta.json"

        # Création des dossiers si nécessaires
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Chargement en RAM
        self.sommaire  = self._charger_json(self.sommaire_path, default={})
        self.synonymes = self._charger_json(self.synonymes_path, default={})
        self.relations = self._charger_json(self.relations_path, default={})

        # Tous les concepts (vecteurs + metadonnees) - voir docstring en
        # tete de fichier : ancien format = 1 fichier par concept, remplace
        # par ces 2 dictionnaires uniques.
        self._vecteurs: dict[str, torch.Tensor] = (
            torch.load(self.vecteurs_path, weights_only=True)
            if self.vecteurs_path.exists() else {}
        )
        self._metas: dict = self._charger_json(self.meta_path, default={})

        # Compteur de fréquence des mots (pour l'auto-enrichissement)
        self._freq_mots: dict[str, int] = {}

        # Verrou : plusieurs generations peuvent tourner en meme temps
        # (batching, voir bridge.py), chacune declenchant son propre
        # thread d'apprentissage (GestionnaireMemoire.apprendre_async).
        # Sans ce verrou, des ecritures concurrentes sur les memes
        # dictionnaires (sommaire/synonymes/relations) et les memes
        # fichiers JSON peuvent se marcher dessus ou corrompre le fichier
        # (meme risque que la corruption d'experience_bank.pt observee
        # plus tot, mais ici via une vraie race condition, pas juste un
        # Ctrl+C mal tombe).
        self._verrou = threading.Lock()

    # ── Recherche ────────────────────────────────────────────────────

    def rechercher(self, mots: list[str]) -> list[str]:
        """
        Retourne les concept_ids correspondant aux mots (+ synonymes).

        Args:
            mots : liste de mots-clés (ex: ["code", "python", "afficher"])

        Returns:
            Liste de concept_ids du secteur sémantique concerné.
        """
        # Résolution des synonymes
        mots_etendus = set(mots)
        for mot in mots:
            mot_lower = mot.lower()
            mots_etendus.update(self.synonymes.get(mot_lower, []))

        # Recherche dans toutes les catégories
        ids_trouves = set()
        for categorie, ids in self.sommaire.items():
            cat_mots = set(categorie.lower().split("."))
            if mots_etendus & cat_mots:  # Intersection
                ids_trouves.update(ids)

        return list(ids_trouves)

    # ── Indexation ────────────────────────────────────────────────────

    def indexer(
        self,
        vecteur: torch.Tensor,
        type_noeud: str,
        valeur: float,
        poids: int = 1,
        tags: list[str] | None = None,
        categorie: str = "GENERAL",
        texte_original: str = "",
        source_url: list[str] | None = None,
        nuances: list | None = None,
    ) -> str:
        """
        Crée et indexe un nouveau concept en mémoire.

        Returns:
            concept_id : identifiant unique du concept créé
        """
        concept_id = uuid.uuid4().hex[:12]
        tags = tags or []
        source_url = source_url or []
        nuances = nuances or []

        meta = {
            "id":             concept_id,
            "type":           type_noeud,
            "valeur":         valeur,
            "poids":          poids,
            "tags":           tags,
            "categorie":      categorie,
            "texte_original": texte_original[:200],  # Limité pour économiser l'espace
            "source_url":     source_url,
            "nuances":        nuances,
        }

        with self._verrou:
            self._vecteurs[concept_id] = vecteur.detach().cpu()
            self._metas[concept_id] = meta
            self._sauver_vecteurs()
            self._sauver_json(self.meta_path, self._metas)
            # Mise à jour du sommaire
            if categorie not in self.sommaire:
                self.sommaire[categorie] = []
            if concept_id not in self.sommaire[categorie]:
                self.sommaire[categorie].append(concept_id)
            self._sauver_json(self.sommaire_path, self.sommaire)

            # Mise à jour de la fréquence des mots
            for tag in tags:
                self._freq_mots[tag.lower()] = self._freq_mots.get(tag.lower(), 0) + 1

        return concept_id

    def fusionner(self, concept_id: str, vecteur_nouveau: torch.Tensor, score_nouveau: float):
        """
        Fusionne un nouveau vecteur dans un concept existant (compression infinie).

        La valeur augmente sans créer de nouvelle entrée.
        Le garde-fou vérifie ensuite l'invariant racine > branche > feuille.
        """
        meta = self.charger_meta(concept_id)
        if meta is None:
            return

        with self._verrou:
            w = meta["poids"]

            # Mise à jour de la valeur (moyenne pondérée)
            meta["valeur"] = (meta["valeur"] * w + score_nouveau) / (w + 1)
            meta["poids"]  = w + 1

            # Mise à jour du vecteur (moyenne pondérée)
            vecteur_ancien = self.charger_vecteur(concept_id)
            if vecteur_ancien is not None:
                vecteur_fusionne = (vecteur_ancien * w + vecteur_nouveau.detach().cpu()) / (w + 1)
                self._vecteurs[concept_id] = vecteur_fusionne
                self._sauver_vecteurs()

            self._sauver_meta(concept_id, meta)

    def ajouter_nuance(
        self,
        concept_id: str,
        vecteur_nuance: torch.Tensor,
        score: float,
        type_noeud: str = "feuille",
        source_url: list[str] | None = None,
    ):
        """
        Ajoute une nuance (critique) à un concept existant.
        Si une nuance identique existe déjà, on la renforce (poids++).
        """
        meta = self.charger_meta(concept_id)
        if meta is None:
            return

        vecteur_np = vecteur_nuance.detach().cpu()

        with self._verrou:
            # Chercher si une nuance similaire existe déjà
            for nuance in meta.get("nuances", []):
                if nuance.get("type") == type_noeud:
                    n_w = nuance.get("poids", 1)
                    nuance["valeur"] = (nuance["valeur"] * n_w + score) / (n_w + 1)
                    nuance["poids"]  = n_w + 1
                    self._sauver_meta(concept_id, meta)
                    return

            # Nouvelle nuance
            meta.setdefault("nuances", []).append({
                "type":       type_noeud,
                "valeur":     score,
                "poids":      1,
                "source_url": source_url or [],
            })
            self._sauver_meta(concept_id, meta)

    def verifier_garde_fou(self, concept_id: str, verificateur) -> str:
        """
        Garde-fou du compresseur infini.

        Si une nuance devient trop lourde (80% du poids de la racine),
        on appelle le Vérificateur Web pour arbitrage scientifique.

        Returns:
            'stable'  — rien n'a changé
            'reduit'  — poids de la nuance réduit (racine confirme par science)
            'inverse' — racine et nuance échangent leurs rôles (rare)
        """
        meta = self.charger_meta(concept_id)
        if meta is None or meta["type"] != "racine":
            return "stable"

        poids_racine = meta["poids"]
        nuances = meta.get("nuances", [])

        for nuance in nuances:
            if nuance["poids"] >= poids_racine * SEUIL_ALERTE_CONFLIT:
                # La nuance menace de dépasser la racine → arbitrage scientifique
                texte_racine = meta.get("texte_original", "")
                texte_nuance = nuance.get("texte_original", "")

                if not texte_racine:
                    return "stable"  # Pas assez d'info pour arbitrer

                verdict = verificateur.arbitrer_conflit(texte_racine, texte_nuance)
                # Le verrou n'englobe QUE la modification+sauvegarde, jamais
                # l'appel reseau ci-dessus - sinon toute la memoire se
                # retrouverait bloquee pendant l'attente de Wikipedia.
                with self._verrou:
                    if verdict == "racine" or verdict == "inconnu":
                        # La science confirme la racine → on réduit la nuance de 20%
                        nuance["poids"] = max(1, int(nuance["poids"] * 0.80))
                        self._sauver_meta(concept_id, meta)
                        return "reduit"

                    elif verdict == "nuance":
                        # La science confirme la nuance → inversion rare mais légitime
                        ancien_type = meta["type"]
                        meta["type"] = nuance["type"]
                        nuance["type"] = ancien_type
                        self._sauver_meta(concept_id, meta)
                        return "inverse"

        return "stable"

    # ── Auto-enrichissement des synonymes ────────────────────────────

    def tenter_synonyme(
        self,
        mot_a: str,
        mot_b: str,
        vecteur_a: torch.Tensor,
        vecteur_b: torch.Tensor,
    ):
        """
        Tente d'ajouter mot_b comme synonyme de mot_a.

        Les 3 gardes-fous SIMULTANÉS doivent être vrais :
        1. Similarité cosinus > 0.92
        2. Fréquence de chaque mot > 50
        3. (fréquence vérifiée via _freq_mots)
        """
        mot_a, mot_b = mot_a.lower(), mot_b.lower()

        if mot_a == mot_b:
            return

        # Condition 1 : Similarité cosinus
        sim = F.cosine_similarity(
            vecteur_a.detach().cpu().unsqueeze(0),
            vecteur_b.detach().cpu().unsqueeze(0),
        ).item()
        if sim < SEUIL_SIM_SYNONYME:
            return

        # Condition 2 : Fréquence minimale des deux mots
        if (self._freq_mots.get(mot_a, 0) < SEUIL_FREQ_SYNONYME or
                self._freq_mots.get(mot_b, 0) < SEUIL_FREQ_SYNONYME):
            return

        # Toutes les conditions sont remplies → ajout croisé des synonymes
        with self._verrou:
            syns_a = self.synonymes.setdefault(mot_a, [])
            syns_b = self.synonymes.setdefault(mot_b, [])

            if mot_b not in syns_a:
                syns_a.append(mot_b)
            if mot_a not in syns_b:
                syns_b.append(mot_a)

            self._sauver_json(self.synonymes_path, self.synonymes)

    # ── Relations multi-sauts ─────────────────────────────────────────

    def lier_concepts(
        self,
        id_a: str,
        id_b: str,
        type_relation: str = "lie",
        force: float = 0.5,
    ):
        """
        Crée un lien explicite entre deux concepts (bidirectionnel).

        Contrairement à une fusion (même concept) ou une nuance (critique
        d'un même concept), une relation connecte DEUX concepts DIFFÉRENTS
        — la brique de base du raisonnement multi-sauts (lier une info de
        H+2 avec une info de H+22, par exemple).
        """
        if id_a == id_b:
            return

        with self._verrou:
            liste_a = self.relations.setdefault(id_a, [])
            if not any(r["vers"] == id_b for r in liste_a):
                liste_a.append({"vers": id_b, "type": type_relation, "force": force})

            liste_b = self.relations.setdefault(id_b, [])
            if not any(r["vers"] == id_a for r in liste_b):
                liste_b.append({"vers": id_a, "type": type_relation, "force": force})

            self._sauver_json(self.relations_path, self.relations)

    def concepts_lies(self, concept_id: str) -> list[dict]:
        """Retourne les voisins directs d'un concept (1 saut)."""
        return self.relations.get(concept_id, [])

    def chemin_entre(
        self,
        id_depart: str,
        id_arrivee: str,
        profondeur_max: int = 3,
    ) -> list[str] | None:
        """
        Recherche en largeur (BFS) le chemin le plus court entre deux
        concepts dans le graphe de relations, jusqu'à profondeur_max sauts.
        """
        if id_depart == id_arrivee:
            return [id_depart]

        file_attente = [[id_depart]]
        visites = {id_depart}

        while file_attente:
            chemin = file_attente.pop(0)
            if len(chemin) - 1 >= profondeur_max:
                continue

            dernier = chemin[-1]
            for relation in self.concepts_lies(dernier):
                voisin = relation["vers"]
                if voisin == id_arrivee:
                    return chemin + [voisin]
                if voisin not in visites:
                    visites.add(voisin)
                    file_attente.append(chemin + [voisin])

        return None

    # ── I/O Utilitaires ──────────────────────────────────────────────

    def charger_vecteur(self, concept_id: str) -> torch.Tensor | None:
        return self._vecteurs.get(concept_id)

    def charger_meta(self, concept_id: str) -> dict | None:
        return self._metas.get(concept_id)

    def charger_matrice(self, concept_ids: list[str]) -> tuple[torch.Tensor, list[str]]:
        """
        Charge les vecteurs de plusieurs concepts en un seul Tensor (B, D).
        Retourne aussi les IDs valides (certains concepts peuvent être supprimés).
        """
        vecteurs, ids_valides = [], []
        for cid in concept_ids:
            v = self.charger_vecteur(cid)
            if v is not None:
                vecteurs.append(v)
                ids_valides.append(cid)
        if not vecteurs:
            return torch.empty(0), []
        return torch.stack(vecteurs), ids_valides

    def _sauver_meta(self, concept_id: str, meta: dict):
        """Met a jour UNE entree en RAM puis reecrit le fichier consolide
        entier (concepts_meta.json) - voir docstring en tete de fichier."""
        self._metas[concept_id] = meta
        self._sauver_json(self.meta_path, self._metas)

    def _sauver_vecteurs(self):
        """Reecrit le fichier consolide de TOUS les vecteurs. Appele par
        les methodes qui modifient self._vecteurs, sous self._verrou."""
        torch.save(self._vecteurs, self.vecteurs_path)

    @staticmethod
    def _charger_json(path: Path, default) -> dict:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _sauver_json(self, path: Path, data: dict):
        # Pas d'indentation ici : ce fichier (sommaire/synonymes/relations)
        # est reecrit EN ENTIER a chaque mise a jour et grandit sur toute
        # une session - l'indentation coute cher en I/O sur un gros
        # fichier reecrit des milliers de fois, sans vraie utilite (ce
        # n'est pas fait pour etre lu/edite a la main regulierement).
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
