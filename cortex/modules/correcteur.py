"""
Correcteur Orthographique Ultra-Rapide pour CORTEX.

Combine deux approches :
  - Solution 1 : Dictionnaire avec VRAIES frequences d'usage
  - Solution 2 : Algorithme de Peter Norvig (distance 2, substitutions, inversions)

Chargement Paresseux : 0 RAM au demarrage. Le dictionnaire ne se charge
que lors du PREMIER appel a corriger_phrase().
"""

import os
import re
import unicodedata
from collections import Counter
from typing import Optional, Set


class CorrecteurRapide:
    """
    Correcteur orthographique base sur l'algorithme de Peter Norvig.
    Gere les erreurs de distance 2 (deux fautes par mot).
    Utilise les frequences reelles pour choisir le meilleur candidat.
    """

    LETTRES = 'abcdefghijklmnopqrstuvwxyz'

    def __init__(self, dico_path: Optional[str] = None):
        if dico_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.dico_path = os.path.join(base_dir, 'data', 'dico_francais.txt')
        else:
            self.dico_path = dico_path

        self._is_loaded = False
        self.MOTS: Counter = Counter()

    # ── Chargement Paresseux ─────────────────────────────────────────
    def _load(self):
        """Charge le dictionnaire en memoire (une seule fois)."""
        if self._is_loaded:
            return

        print("[Correcteur] Chargement paresseux active...")

        if not os.path.exists(self.dico_path):
            print("[Correcteur] Dictionnaire introuvable, creation d'un dico de secours.")
            self._creer_dico_secours()

        with open(self.dico_path, 'r', encoding='utf-8', errors='ignore') as f:
            for ligne in f:
                parts = ligne.strip().split()
                if not parts:
                    continue
                mot = parts[0].lower()
                freq = int(parts[1]) if len(parts) > 1 else 1

                # Normaliser (retirer accents)
                mot = self._normaliser(mot)

                if mot and mot.isalpha():
                    self.MOTS[mot] = max(self.MOTS[mot], freq)

        self._is_loaded = True
        print(f"[Correcteur] {len(self.MOTS)} mots indexes (frequences reelles). Pret.")

    def _creer_dico_secours(self):
        """Cree un mini dictionnaire si le vrai n'existe pas."""
        os.makedirs(os.path.dirname(self.dico_path), exist_ok=True)
        mots = [
            "bonjour 100000", "cortex 100000", "comment 90000", "ca 80000",
            "va 80000", "est 200000", "le 500000", "la 500000", "les 400000",
            "un 300000", "une 300000", "des 400000", "et 400000", "en 300000",
            "de 600000", "je 200000", "tu 150000", "il 200000", "nous 100000",
            "vous 100000", "ils 100000", "pour 150000", "avec 120000",
            "dans 130000", "que 200000", "qui 180000", "sur 100000",
            "pas 150000", "mais 120000", "tout 100000", "bien 100000",
            "monde 80000", "beau 60000", "vrai 70000", "pensee 50000",
            "veritable 40000", "heure 80000", "cree 50000", "creer 50000",
        ]
        with open(self.dico_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(mots))

    # ── Utilitaires ──────────────────────────────────────────────────
    @staticmethod
    def _normaliser(mot: str) -> str:
        """Retire les accents d'un mot."""
        return unicodedata.normalize('NFKD', mot).encode('ASCII', 'ignore').decode('utf-8')

    def _P(self, mot: str) -> float:
        """Probabilite du mot (basee sur sa frequence reelle)."""
        N = sum(self.MOTS.values())
        return self.MOTS[mot] / N if N > 0 else 0

    def _connus(self, mots: Set[str]) -> Set[str]:
        """Filtre : ne garde que les mots qui existent dans le dictionnaire."""
        return {m for m in mots if m in self.MOTS}

    # ── Generateur d'erreurs (Norvig) ────────────────────────────────
    def _edits1(self, mot: str) -> Set[str]:
        """
        Genere TOUTES les variantes a 1 erreur de distance :
        - Suppressions  : une lettre en moins
        - Transpositions: deux lettres adjacentes inversees
        - Remplacements : une lettre changee par une autre
        - Insertions    : une lettre ajoutee
        """
        splits = [(mot[:i], mot[i:]) for i in range(len(mot) + 1)]
        suppressions  = [L + R[1:]             for L, R in splits if R]
        transpositions = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        remplacements = [L + c + R[1:]         for L, R in splits if R for c in self.LETTRES]
        insertions    = [L + c + R             for L, R in splits for c in self.LETTRES]
        return set(suppressions + transpositions + remplacements + insertions)

    def _edits2(self, mot: str) -> Set[str]:
        """Genere toutes les variantes a 2 erreurs de distance."""
        return {e2 for e1 in self._edits1(mot) for e2 in self._edits1(e1)}

    # ── Correction d'un seul mot ─────────────────────────────────────
    def _corriger_mot(self, mot: str) -> str:
        """
        Algorithme de Peter Norvig :
        1. Si le mot existe tel quel -> on le garde
        2. Sinon, on cherche parmi les mots a distance 1
        3. Sinon, on cherche parmi les mots a distance 2
        4. Sinon, on renvoie le mot tel quel (mot inconnu)
        
        A chaque etape, on choisit le candidat le PLUS FREQUENT.
        """
        if not mot:
            return mot

        mot_original = mot
        mot = mot.lower()

        # Ne pas corriger les mots tres courts (c', l', d', etc.)
        if len(mot) <= 2:
            return mot_original

        mot = self._normaliser(mot)

        # Etape 1 : le mot existe deja ?
        if mot in self.MOTS:
            return mot

        # Etape 1b : Le mot est-il un prefixe tronque d'un mot connu ?
        # Ex: "heur" est un prefixe de "heure" -> on complete
        completions = {m for m in self.MOTS if m.startswith(mot) and len(m) <= len(mot) + 2}
        
        # Etape 2 : candidats a distance 1
        candidats_d1 = self._connus(self._edits1(mot))
        
        # Fusionner les completions avec les candidats d1, en boostant les completions
        if completions or candidats_d1:
            tous = completions | candidats_d1
            # Les completions de prefixe ont un bonus x10 car elles sont plus probables
            def score(m):
                p = self._P(m)
                if m in completions:
                    p *= 10
                return p
            return max(tous, key=score)

        # Etape 3 : candidats a distance 2
        candidats_d2 = self._connus(self._edits2(mot))
        if candidats_d2:
            return max(candidats_d2, key=self._P)

        # Etape 4 : aucune correction trouvee
        return mot

    # ── Correction d'une phrase entiere ──────────────────────────────
    def corriger_phrase(self, phrase: str) -> str:
        """
        Corrige une phrase entiere.
        Declenche le chargement en RAM si c'est la premiere utilisation.
        """
        if not phrase.strip():
            return phrase

        if not self._is_loaded:
            self._load()

        # Decouper la phrase en mots et non-mots (ponctuation, espaces)
        tokens = re.findall(r"[a-zA-ZÀ-ÿ]+|[^a-zA-ZÀ-ÿ]+", phrase)

        resultat = []
        for token in tokens:
            if re.match(r'^[a-zA-ZÀ-ÿ]+$', token):
                resultat.append(self._corriger_mot(token))
            else:
                resultat.append(token)

        return "".join(resultat)
