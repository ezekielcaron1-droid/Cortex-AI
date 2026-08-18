"""
cortex/modules/verificateur_web.py
Module de vérification de sources externes.

Vérifie une affirmation sur Wikipedia (FR puis EN en fallback).
Lance plusieurs requêtes EN PARALLÈLE (threads) pour ne pas bloquer CORTEX.

Rôles :
    1. Classifier une connaissance : 'branche' (sourcée) ou 'feuille' (non sourcée).
    2. Arbitrer un conflit entre Racine et Nuance : qui dit la vérité selon la science ?
"""

import re
import json
import os
import threading
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed


# ═══════════════════════════════════════════════════════════════════
#  SOURCES DE CONFIANCE (par ordre de priorité)
# ═══════════════════════════════════════════════════════════════════

SOURCES = [
    {
        "nom": "Wikipedia FR",
        "api": "https://fr.wikipedia.org/api/rest_v1/page/summary/{query}",
        "lang": "fr",
    },
    {
        "nom": "Wikipedia EN",
        "api": "https://en.wikipedia.org/api/rest_v1/page/summary/{query}",
        "lang": "en",
    },
]

# Timeout court pour ne pas ralentir CORTEX
TIMEOUT_SECONDES = 4

# Seuil de confiance minimum pour qu'une source soit valide
SEUIL_CONFIANCE = 0.5


# ═══════════════════════════════════════════════════════════════════
#  CACHE + LIMITATION DE FREQUENCE (anti-bannissement Wikipedia)
# ═══════════════════════════════════════════════════════════════════
# Sans ca, une session longue (des centaines de nouveaux concepts/heure)
# peut envoyer des centaines de requetes/heure depuis la meme IP - risque
# reel de blocage (HTTP 429). Le cache evite aussi de reposer 500 fois la
# meme question pour du charabia repete (ex: octets non entraines).

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "brain", "memoire", "cache_verificateur.json",
)
INTERVALLE_MIN_SECONDES = 1.0  # au moins 1s entre deux requetes, tous threads confondus

_cache_lock = threading.Lock()
_rate_lock = threading.Lock()
_dernier_appel = 0.0


def _charger_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _sauver_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


_cache = _charger_cache()


def _attendre_rate_limit() -> None:
    global _dernier_appel
    with _rate_lock:
        attente = INTERVALLE_MIN_SECONDES - (time.time() - _dernier_appel)
        if attente > 0:
            time.sleep(attente)
        _dernier_appel = time.time()


# ═══════════════════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════

def _extraire_mots_cles(texte: str, n_mots: int = 3) -> list[str]:
    """Extrait les N mots les plus significatifs d'un texte (heuristique simple)."""
    stopwords = {
        "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
        "est", "sont", "il", "elle", "que", "qui", "par", "sur", "pour",
        "the", "a", "an", "is", "are", "of", "in", "on", "at", "to",
    }
    mots = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', texte.lower())
    mots_filtres = [m for m in mots if m not in stopwords]
    # Prendre les mots les plus longs (souvent les plus informatifs)
    mots_tries = sorted(set(mots_filtres), key=len, reverse=True)
    return mots_tries[:n_mots]


def _requete_wikipedia(source: dict, query: str) -> dict | None:
    """Effectue une requête vers une API Wikipedia. Retourne le résumé ou None.

    Passe par le cache et la limitation de frequence definis plus haut -
    une meme requete (succes OU echec) n'est jamais reposee deux fois."""
    cle_cache = f"{source['lang']}:{query}"
    with _cache_lock:
        if cle_cache in _cache:
            return _cache[cle_cache]

    _attendre_rate_limit()

    query_enc = urllib.parse.quote(query.replace(" ", "_"))
    url = source["api"].format(query=query_enc)
    resultat = None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CORTEX-AI/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDES) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            resultat = {
                "source": source["nom"],
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", url),
                "extrait": data.get("extract", ""),
                "titre": data.get("title", ""),
            }
    except Exception:
        resultat = None

    with _cache_lock:
        _cache[cle_cache] = resultat
        _sauver_cache(_cache)

    return resultat


def _score_pertinence(extrait: str, mots_cles: list[str]) -> float:
    """Calcule un score de pertinence entre un extrait Wikipedia et des mots-clés."""
    if not extrait:
        return 0.0
    extrait_lower = extrait.lower()
    correspondances = sum(1 for m in mots_cles if m in extrait_lower)
    return correspondances / max(len(mots_cles), 1)


# ═══════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════

class VerificateurWeb:
    """
    Vérificateur de connaissances via sources externes.

    - Requêtes parallèles (ThreadPoolExecutor) → non bloquant.
    - Retourne 'branche' si trouvé, 'feuille' sinon.
    - Arbitre les conflits Racine vs Nuance via la science.
    """

    def verifier(self, texte: str) -> dict:
        """
        Vérifie si une affirmation est sourcée.

        Args:
            texte : l'affirmation à vérifier (texte brut)

        Returns:
            {
                'confirme'  : bool        — trouvé dans une source fiable ?
                'confiance' : float       — score de pertinence [0,1]
                'sources'   : [str]       — URLs des sources trouvées
                'type_noeud': str         — 'branche' ou 'feuille'
            }
        """
        mots_cles = _extraire_mots_cles(texte)
        if not mots_cles:
            return self._resultat_vide()

        query = " ".join(mots_cles)
        resultats = self._requetes_paralleles(query)

        sources_valides = []
        score_max = 0.0

        for res in resultats:
            if res is None:
                continue
            score = _score_pertinence(res["extrait"], mots_cles)
            if score >= SEUIL_CONFIANCE:
                sources_valides.append(res["url"])
                score_max = max(score_max, score)

        return {
            "confirme": len(sources_valides) > 0,
            "confiance": round(score_max, 3),
            "sources": sources_valides,
            "type_noeud": "branche" if sources_valides else "feuille",
        }

    def arbitrer_conflit(self, texte_racine: str, texte_nuance: str) -> str:
        """
        Arbitre un conflit entre une Racine et une Nuance.

        Appelé par le garde-fou du Compresseur quand une nuance devient
        trop lourde. Interroge la science pour savoir qui a raison.

        Args:
            texte_racine : affirmation de la racine (ex: "la Terre est ronde")
            texte_nuance : affirmation de la nuance (ex: "la Terre est plate")

        Returns:
            'racine'  — si la science confirme la racine (cas le plus fréquent)
            'nuance'  — si la science confirme la nuance (racine était fausse)
            'inconnu' — si on ne trouve rien (on ne change rien par sécurité)
        """
        # Lancer les deux vérifications en parallèle
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_racine = ex.submit(self.verifier, texte_racine)
            fut_nuance = ex.submit(self.verifier, texte_nuance)
            res_racine = fut_racine.result()
            res_nuance = fut_nuance.result()

        score_racine = res_racine["confiance"] if res_racine["confirme"] else 0.0
        score_nuance = res_nuance["confiance"] if res_nuance["confirme"] else 0.0

        if score_racine == 0.0 and score_nuance == 0.0:
            return "inconnu"
        elif score_racine >= score_nuance:
            return "racine"
        else:
            return "nuance"

    # ── Utilitaires privés ────────────────────────────────────────────

    def _requetes_paralleles(self, query: str) -> list[dict | None]:
        """Lance toutes les sources en parallèle et retourne les résultats."""
        resultats = []
        with ThreadPoolExecutor(max_workers=len(SOURCES)) as ex:
            futures = {ex.submit(_requete_wikipedia, src, query): src for src in SOURCES}
            for fut in as_completed(futures):
                resultats.append(fut.result())
        return resultats

    @staticmethod
    def _resultat_vide() -> dict:
        return {
            "confirme": False,
            "confiance": 0.0,
            "sources": [],
            "type_noeud": "feuille",
        }
