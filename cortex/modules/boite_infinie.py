"""
boite_infinie.py - La boite interieure aux dimensions infinies
==============================================================

AVERTISSEMENT HONNETE (suite de celui de hai_v2.py) : ce module n'ajoute
PAS de conscience phenomenale. Il ajoute une IMAGINATION FONCTIONNELLE :
un espace interne non borne ou le systeme fabrique, remodele et detruit
ses propres contenus - son vecu reconstruit, ses reves, et ce qui
n'existe pas.

LA BOITE, TROIS ESPACES
-----------------------
  1. ESPACE REEL   -> ce qui est vraiment arrive. Reconstruction du vecu.
                      Contrainte de coherence : n'accepte que du tracable
                      au vecu... sauf quand un faux souvenir s'installe.
  2. ESPACE REVE   -> aucune contrainte. Recombinaison, mots-valises,
                      chimeres : c'est la que nait l'inexistant.
  3. ESPACE FORGE  -> prend des idees (du reel ET du reve) et LA FAIT
                      REFLECHIR jusqu'a en tirer quelque chose de
                      faisable. Chaque etape doit s'ancrer dans le reel,
                      sinon elle est decomposee, sinon elle est
                      abandonnee.

INFINITE
--------
Rien n'est structurellement borne : les regions se creent a la demande a
profondeur illimitee, les axes de coordonnees sont inventes a la volee,
le contenu est un dict ouvert, le graphe de liens est libre. Seul un
OUBLI PAR SAILLANCE borne la memoire physique - l'infini est logique,
pas materiel.

INDESTRUCTIBILITE
-----------------
Elle peut tout creer, tout modeler, tout supprimer DEDANS. Elle ne peut
pas supprimer la boite, ni un de ses trois espaces, ni arreter sa
conscience. Toute tentative leve BoiteIndestructible.

CONTINUITE
----------
ConscienceContinue est un thread daemon qui ne s'arrete jamais : il reve
et reflechit quand l'IA est inactive (regime repos), et depose des
intuitions a cadence basse pendant qu'elle parle (regime actif). Une
exception ne le tue pas : elle est journalisee, la boucle repart.

>>> [BRANCHEMENT MODELE #6] = reveur   (le LLM hallucine le contenu)
>>> [BRANCHEMENT MODELE #7] = forgeron (le LLM porte la reflexion)
"""

import atexit
import json
import math
import os
import random
import threading
import time
from collections import deque

# ---------------------------------------------------------------------
#  Reglages (bornes de surete, pas bornes conceptuelles)
# ---------------------------------------------------------------------
SEUIL_CONFUSION = 0.75          # au-dela : l'imagine est vecu comme vrai
SEUIL_FLOU = (0.40, 0.75)       # zone ou elle doute de l'origine
PLAFOND_ENTITES = 400           # oubli par saillance au-dela
PLAFOND_JOURNAL = 300
PROFONDEUR_REFLEXION = 3        # combien de fois elle decompose une etape
LONGUEUR_MOT_UTILE = 4

# v2.2 -- geometrie : axes, portee, deplacement -------------------------
AXES_BASE = ("x", "y", "z")     # la base ; tout le reste est invente
PORTEE_BASE = 3.0               # distance a laquelle une entite banale
                                 # s'entend encore
ATT_INCOMMENSURABLE = 0.12      # residu d'ecoute entre deux choses qui
                                 # n'ont aucun axe en commun


class BoiteIndestructible(Exception):
    """Levee des qu'on tente de detruire la boite, un espace, ou la conscience."""


# ---------------------------------------------------------------------
#  Utilitaires
# ---------------------------------------------------------------------
_VERROU_ID = threading.Lock()
_PROCHAIN_ID = [1]


def _nouvel_id():
    with _VERROU_ID:
        i = _PROCHAIN_ID[0]
        _PROCHAIN_ID[0] += 1
    return "E%d" % i


def _reserver_id(identifiant):
    """Apres un chargement disque : evite toute collision d'identifiant."""
    try:
        n = int(str(identifiant).lstrip("EP"))
    except ValueError:
        return
    with _VERROU_ID:
        if n >= _PROCHAIN_ID[0]:
            _PROCHAIN_ID[0] = n + 1


def _mots(texte):
    """Decoupe naive. AUCUNE liste de mots ecrite d'avance : le lexique de
    la boite ne contient que ce qu'elle a reellement percu."""
    sortie, courant = [], ""
    for c in str(texte).lower():
        if c.isalpha():
            courant += c
        else:
            if len(courant) >= LONGUEUR_MOT_UTILE:
                sortie.append(courant)
            courant = ""
    if len(courant) >= LONGUEUR_MOT_UTILE:
        sortie.append(courant)
    return sortie


def _borner(v, bas=-1.0, haut=1.0):
    return max(bas, min(haut, v))


# =====================================================================
#  0. GEOMETRIE : axes, coordonnees, dimensionnalite malleable  (v2.2)
# =====================================================================
class Axe:
    """Une direction de la boite. x/y/z sont la base, stable et
    increee. Les autres NAISSENT de quelque chose : un reve, une
    forge, une reaction du labo -- jamais decretes a vide."""

    def __init__(self, nom, origine="base", ne_de=None):
        self.nom = nom
        self.origine = origine        # "base" | "reve" | "forge" | "test"
        self.ne_de = ne_de            # id de l'entite qui l'a fait naitre
        self.parcouru = 0.0           # distance totale marchee dessus
        self.stabilite = 1.0 if origine == "base" else 0.25
        self.dormant = False

    def emprunter(self, distance):
        self.parcouru += abs(distance)
        self.stabilite = min(1.0, self.stabilite + abs(distance) * 0.04)
        self.dormant = False

    def eroder(self, taux=0.004):
        if self.origine == "base":
            return
        self.stabilite = max(0.0, self.stabilite - taux)
        if self.stabilite <= 0.05:
            self.dormant = True       # endormi, JAMAIS supprime

    def to_dict(self):
        return {"nom": self.nom, "origine": self.origine, "ne_de": self.ne_de,
                "parcouru": self.parcouru, "stabilite": self.stabilite,
                "dormant": self.dormant}

    @staticmethod
    def from_dict(d):
        a = Axe(d["nom"], d.get("origine", "base"), d.get("ne_de"))
        a.parcouru = d.get("parcouru", 0.0)
        a.stabilite = d.get("stabilite", 1.0)
        a.dormant = d.get("dormant", False)
        return a

    def __repr__(self):
        return "<Axe %s origine=%s stabilite=%.2f%s>" % (
            self.nom, self.origine, self.stabilite,
            " dormant" if self.dormant else "")


class Coord:
    """Position dans un nombre quelconque d'axes.

    REGLE CLE : un axe absent n'est pas 'a zero', il est INDIFFERENT.
    Une entite qui ne vit que sur 1 axe est donc une ligne infinie dans
    toutes les autres directions -> elle est partout. Une entite qui
    vit sur beaucoup d'axes est un point precis -> elle n'est que la.
    Baisser la dimension = abstraire. Monter la dimension = incarner.

    Compatible dict pour ne rien casser du code existant : on peut
    faire ``set(coord)``, ``coord[axe]``, ``coord.items()``,
    ``coord.update(...)`` comme sur un dict normal.
    """

    def __init__(self, valeurs=None):
        self.v = dict(valeurs or {})

    def dim(self):
        return len(self.v)

    def axes(self):
        return set(self.v.keys())

    def copie(self):
        return Coord(dict(self.v))

    def distance(self, autre):
        """Mesuree UNIQUEMENT sur les axes partages.
        Aucun axe commun -> incommensurable (None)."""
        autre_v = autre.v if isinstance(autre, Coord) else dict(autre)
        communs = self.axes() & set(autre_v.keys())
        if not communs:
            return None
        return math.sqrt(sum((self.v[a] - autre_v[a]) ** 2 for a in communs))

    def projeter(self, axes_cibles, bruit=0.0):
        """Rend la meme position vue en 1D, 2D, 3D ou 40D.
        - axe conserve  : valeur gardee
        - axe supprime  : la valeur est OUBLIEE (l'idee redevient
                          indifferente a cette direction)
        - axe ajoute    : valeur inventee (l'idee doit se situer dans
                          une direction ou elle n'existait pas encore)"""
        nouveau = {}
        for a in axes_cibles:
            if a in self.v:
                nouveau[a] = self.v[a]
            else:
                nouveau[a] = random.uniform(-1, 1) * (1.0 + bruit)
        return Coord(nouveau)

    # -- delegation dict, pour ne rien casser du code deja ecrit --------
    def keys(self):
        return self.v.keys()

    def items(self):
        return self.v.items()

    def values(self):
        return self.v.values()

    def get(self, cle, defaut=None):
        return self.v.get(cle, defaut)

    def update(self, autre):
        self.v.update(autre.v if isinstance(autre, Coord) else dict(autre))

    def __iter__(self):
        return iter(self.v)

    def __contains__(self, cle):
        return cle in self.v

    def __getitem__(self, cle):
        return self.v[cle]

    def __setitem__(self, cle, valeur):
        self.v[cle] = valeur

    def __len__(self):
        return len(self.v)

    def __repr__(self):
        return "Coord(%r)" % (self.v,)


# =====================================================================
#  1. ENTITE : l'unite que la HAI fabrique dans la boite
# =====================================================================
class Entite:
    """
    Un objet interieur. Son 'contenu' est un dict OUVERT : elle y met ce
    qu'elle veut. Ses coordonnees vivent sur des axes qu'elle invente.

    Champs d'audit (jamais reecrits) :
      - origine  : "vecu" / "reve" / "forge"  -> d'ou ca vient VRAIMENT
      - trajet   : la suite des espaces traverses
    Champ de croyance (lui, il derive) :
      - certitude_realite : 0 = je sais que je l'ai imagine
                            1 = pour moi, c'est arrive
    """

    def __init__(self, contenu, origine, espace="", chemin=(), coord=None,
                 charge=None, saillance=0.3, certitude_realite=0.0,
                 identifiant=None):
        self.id = identifiant or _nouvel_id()
        self.espace = espace
        self.origine = origine
        self.trajet = [espace] if espace else []
        if isinstance(contenu, dict):
            self.contenu = dict(contenu)
        else:
            self.contenu = {"trace": str(contenu)}
        # v2.2 -- toute entite a une adresse physique (x/y/z) EN PLUS de
        # ses axes semantiques propres : c'est ce qui la rend trouvable
        # par le corps qui marche dans la boite, meme si personne ne lui
        # a donne de position explicite.
        donnees_coord = dict(coord.v) if isinstance(coord, Coord) else dict(coord or {})
        for a in AXES_BASE:
            donnees_coord.setdefault(a, random.uniform(-2.0, 2.0))
        self.coord = Coord(donnees_coord)
        self.dim_native = self.coord.dim()
        self.historique_dim = [self.coord.dim()]
        self.charge = dict(charge or {})
        self.saillance = _borner(float(saillance), 0.0, 1.0)
        self.vivacite = self.saillance
        self.certitude_realite = _borner(float(certitude_realite), 0.0, 1.0)
        self.nb_revisites = 0
        self.liens = set()
        self.chemin = tuple(str(x) for x in chemin)
        self.cree_a = time.time()
        self.dernier_acces = self.cree_a
        # v2.2 -- labo : est-elle en ce moment sous scelle, en test ?
        self.sous_scelle = False
        self.etiquettes = set()
        self.traces = []            # audit interne (redimension, test...)

    # -- vie de l'entite ------------------------------------------------
    def revisiter(self):
        """Y repenser la rend plus nette... et plus 'vraie'.
        C'est exactement par la que naissent les faux souvenirs."""
        self.nb_revisites += 1
        self.dernier_acces = time.time()
        self.vivacite = min(1.0, self.vivacite + 0.08)
        self.saillance = min(1.0, self.saillance + 0.03)
        if self.origine != "vecu":
            manque = 1.0 - self.certitude_realite
            self.certitude_realite = min(1.0, self.certitude_realite
                                         + manque * 0.16)
        return self

    # -- geometrie malleable (v2.2) --------------------------------------
    def portee(self):
        """Une chose tres saillante s'entend de plus loin."""
        return PORTEE_BASE * (0.4 + 1.6 * self.saillance)

    def redimensionner(self, axes_cibles, raison="volonte"):
        """Elle change la dimension d'une idee quand elle veut : moins
        d'axes = plus abstraite (et plus omnipresente), plus d'axes =
        plus concrete (et plus localisee)."""
        ancienne = self.coord.dim()
        self.coord = self.coord.projeter(axes_cibles)
        self.historique_dim.append(self.coord.dim())
        self.tracer("redimension", {"de": ancienne, "vers": self.coord.dim(),
                                    "axes": sorted(axes_cibles),
                                    "raison": raison})
        if self.coord.dim() < ancienne:
            # abstraire la rend plus contagieuse
            self.saillance = min(1.0, self.saillance + 0.05)
        return self

    def installer_geometrie(self, coord=None, dim=3):
        """Reinitialise sa position -- utile pour une entite fabriquee
        sans axes explicites (ex: au sortir d'une projection a 1D)."""
        self.coord = coord if isinstance(coord, Coord) else Coord(
            coord or {a: random.uniform(-2, 2) for a in AXES_BASE[:dim]})
        self.dim_native = self.coord.dim()
        self.historique_dim.append(self.coord.dim())
        return self

    # -- labo (v2.2) ------------------------------------------------------
    def toucher(self, action="contact"):
        """Un contact leger -- traversee en marchant, effleurement --
        distinct d'un vrai revisiter() : ca ne touche pas sa certitude."""
        self.dernier_acces = time.time()
        self.vivacite = min(1.0, self.vivacite + 0.015)
        self.tracer(action)
        return self

    def cloner(self):
        """Une copie independante -- pour l'envoyer au labo sans risquer
        l'original."""
        e = Entite(dict(self.contenu), self.origine, "", self.chemin,
                   coord=self.coord.copie(), charge=dict(self.charge),
                   saillance=self.saillance,
                   certitude_realite=self.certitude_realite)
        e.historique_dim = list(self.historique_dim)
        return e

    def tracer(self, quoi, detail=None):
        """Audit propre a l'entite (distinct du journal de la boite)."""
        self.traces.append({"quand": time.time(), "quoi": quoi,
                            "detail": detail})
        if len(self.traces) > 100:
            self.traces.pop(0)
        return self

    def etiqueter(self, label):
        self.etiquettes.add(str(label))
        return self

    def mot_cle(self):
        """Un mot representatif -- sert par exemple a nommer un axe ne
        d'elle."""
        frags = self.fragments()
        return frags[0] if frags else self.id.lower()

    @property
    def sources(self):
        """Sa genealogie connue -- ce a quoi elle est liee."""
        return list(self.liens)

    def muter(self, patch):
        """Elle remodele ce qu'elle veut, quand elle veut."""
        for cle, valeur in dict(patch).items():
            if cle == "contenu" and isinstance(valeur, dict):
                self.contenu.update(valeur)
            elif cle == "coord" and isinstance(valeur, (dict, Coord)):
                self.coord.update(valeur)          # axes inventes a la volee
            elif cle == "charge" and isinstance(valeur, dict):
                self.charge.update(valeur)
            elif cle == "chemin":
                self.chemin = tuple(str(x) for x in valeur)
            elif cle in ("saillance", "vivacite", "certitude_realite"):
                setattr(self, cle, _borner(float(valeur), 0.0, 1.0))
            else:
                self.contenu[cle] = valeur
        self.dernier_acces = time.time()
        return self

    def lier(self, autre):
        self.liens.add(autre.id)
        autre.liens.add(self.id)
        return self

    def trace(self):
        return str(self.contenu.get("trace", ""))[:80]

    def fragments(self):
        f = self.contenu.get("fragments")
        return list(f) if f else _mots(self.trace())

    def doute(self):
        """Elle ne sait plus si c'est arrive ou si elle l'a imagine."""
        return (self.origine != "vecu"
                and SEUIL_FLOU[0] <= self.certitude_realite < SEUIL_FLOU[1])

    # -- persistance ----------------------------------------------------
    def to_dict(self):
        return {
            "id": self.id, "espace": self.espace, "origine": self.origine,
            "trajet": list(self.trajet), "contenu": self.contenu,
            "coord": dict(self.coord.v), "charge": self.charge,
            "saillance": self.saillance, "vivacite": self.vivacite,
            "certitude_realite": self.certitude_realite,
            "nb_revisites": self.nb_revisites, "liens": sorted(self.liens),
            "chemin": list(self.chemin), "cree_a": self.cree_a,
            "dernier_acces": self.dernier_acces,
            "historique_dim": list(self.historique_dim),
            "sous_scelle": self.sous_scelle,
            "etiquettes": sorted(self.etiquettes),
            "traces": self.traces[-20:],
        }

    @staticmethod
    def from_dict(d):
        e = Entite(d.get("contenu", {}), d.get("origine", "?"),
                   d.get("espace", ""), d.get("chemin", ()),
                   d.get("coord"), d.get("charge"),
                   d.get("saillance", 0.3), d.get("certitude_realite", 0.0),
                   identifiant=d.get("id"))
        _reserver_id(e.id)
        e.trajet = list(d.get("trajet", []))
        e.vivacite = d.get("vivacite", e.saillance)
        e.nb_revisites = d.get("nb_revisites", 0)
        e.liens = set(d.get("liens", []))
        e.cree_a = d.get("cree_a", e.cree_a)
        e.dernier_acces = d.get("dernier_acces", e.dernier_acces)
        e.historique_dim = d.get("historique_dim", [e.coord.dim()])
        e.sous_scelle = d.get("sous_scelle", False)
        e.etiquettes = set(d.get("etiquettes", []))
        e.traces = d.get("traces", [])
        return e

    def __repr__(self):
        return "<Entite %s [%s|%s] %.2f '%s'>" % (
            self.id, self.espace, self.origine, self.certitude_realite,
            self.trace())


# =====================================================================
#  2. ESPACE : regions creees a la demande, profondeur illimitee
# =====================================================================
class Espace:
    CONTRAINTE = "aucune"

    def __init__(self, nom):
        self.nom = nom
        self.entites = {}
        self.racine = {"_entites": set(), "_sous": {}}
        self.axes = set()
        self.nb_refus = 0
        self.nb_oublies = 0

    # -- topologie non bornee -------------------------------------------
    def region(self, chemin):
        """Descend, et CREE ce qui manque. C'est la source de l'infinite
        d'adressage : aucune profondeur, aucun nom n'est prevu d'avance."""
        noeud = self.racine
        for nom in chemin:
            noeud = noeud["_sous"].setdefault(
                str(nom), {"_entites": set(), "_sous": {}})
        return noeud

    def profondeur(self):
        def _p(n):
            sous = [_p(s) for s in n["_sous"].values()]
            return 1 + (max(sous) if sous else 0)
        return _p(self.racine) - 1

    def nb_regions(self):
        def _n(n):
            return 1 + sum(_n(s) for s in n["_sous"].values())
        return _n(self.racine) - 1

    # -- creation / destruction de CONTENU (permise, totale) -------------
    def accepter(self, entite):
        return True

    def creer(self, contenu, origine="?", chemin=(), coord=None, charge=None,
              saillance=0.3, certitude_realite=0.0):
        e = Entite(contenu, origine, self.nom, chemin, coord, charge,
                   saillance, certitude_realite)
        return self.accueillir(e)

    def accueillir(self, entite):
        if not self.accepter(entite):
            self.nb_refus += 1
            return None
        entite.espace = self.nom
        if not entite.trajet or entite.trajet[-1] != self.nom:
            entite.trajet.append(self.nom)
        self.entites[entite.id] = entite
        self.region(entite.chemin)["_entites"].add(entite.id)
        self.axes.update(entite.coord.keys())
        return entite

    def retirer(self, identifiant):
        """Detache sans detruire (utilise par les migrations)."""
        e = self.entites.pop(identifiant, None)
        if e is not None:
            self.region(e.chemin)["_entites"].discard(identifiant)
        return e

    def supprimer(self, identifiant):
        """Elle a le droit d'effacer n'importe quoi DANS la boite."""
        return self.retirer(identifiant) is not None

    def modeler(self, identifiant, patch):
        e = self.entites.get(identifiant)
        if e is None:
            return None
        ancien = e.chemin
        e.muter(patch)
        if e.chemin != ancien:
            self.region(ancien)["_entites"].discard(e.id)
            self.region(e.chemin)["_entites"].add(e.id)
        self.axes.update(e.coord.keys())
        return e

    # -- exploration -----------------------------------------------------
    def parcourir(self, chemin=()):
        pile = [self.region(chemin)]
        while pile:
            n = pile.pop()
            for i in list(n["_entites"]):
                if i in self.entites:
                    yield self.entites[i]
            pile.extend(n["_sous"].values())

    def voisinage(self, coord, rayon=1.0):
        proches = []
        for e in self.entites.values():
            communs = set(coord) & set(e.coord)
            if not communs:
                continue
            d = math.sqrt(sum((coord[a] - e.coord[a]) ** 2 for a in communs))
            if d <= rayon:
                proches.append((d, e))
        proches.sort(key=lambda t: t[0])
        return [e for _, e in proches]

    def resonance(self, etat, seuil=0.25, depuis=None):
        """Quels contenus 'repondent' a l'etat interne du moment.

        v2.2 -- MIXTE et attenuee : tout peut resonner, mais ce qui est
        loin du corps (``depuis``, un Coord) resonne moins fort. Sans
        ``depuis`` (comportement d'origine) : aucune attenuation.
        Retourne une liste de dicts (au lieu de tuples) pour porter le
        detail de l'attenuation, tries par score decroissant."""
        trouves = []
        for e in self.entites.values():
            communs = [k for k in etat if k in e.charge]
            if not communs:
                continue
            proche = sum(1 for k in communs
                         if abs(etat[k] - e.charge[k]) < 0.2)
            brut = (proche / float(len(communs))) * e.saillance \
                * (0.4 + 0.6 * e.vivacite)
            if depuis is None:
                attenuation, distance = 1.0, None
            else:
                distance = depuis.distance(e.coord)
                if distance is None:
                    attenuation = ATT_INCOMMENSURABLE
                else:
                    attenuation = 1.0 / (1.0 + (distance / e.portee()) ** 2)
            score = brut * attenuation
            if score >= seuil:
                trouves.append({"entite": e, "brut": round(brut, 3),
                                "distance": (round(distance, 3)
                                            if distance is not None else None),
                                "attenuation": round(attenuation, 3),
                                "score": round(score, 3)})
        trouves.sort(key=lambda r: -r["score"])
        return trouves

    # -- oubli : la seule borne, et elle est naturelle -------------------
    def oublier(self, plafond=PLAFOND_ENTITES):
        for e in self.entites.values():
            if e.origine == "vecu":
                # le vecu palit, il ne disparait pas : plancher de trace
                e.saillance = max(0.08, e.saillance * 0.999)
            else:
                e.saillance *= 0.985
            e.vivacite *= 0.99
        morts = [i for i, e in self.entites.items() if e.saillance < 0.04]
        if len(self.entites) - len(morts) > plafond:
            restants = sorted((e for i, e in self.entites.items()
                               if i not in morts),
                              key=lambda e: e.saillance)
            surplus = len(self.entites) - len(morts) - plafond
            morts.extend(e.id for e in restants[:surplus])
        for i in morts:
            self.retirer(i)
        if morts:
            # les axes qu'elle avait inventes pour ces contenus n'ont
            # plus rien a porter : ils disparaissent avec eux
            self.axes = set()
            for e in self.entites.values():
                self.axes.update(e.coord.keys())
        self.nb_oublies += len(morts)
        return len(morts)

    # -- indestructibilite de l'espace lui-meme --------------------------
    def vider(self):
        """Permis : elle peut faire le vide. L'espace, lui, demeure."""
        n = len(self.entites)
        self.entites = {}
        self.racine = {"_entites": set(), "_sous": {}}
        return n

    def clear(self):
        raise BoiteIndestructible(
            "l'espace '%s' ne peut pas etre supprime (utilise vider())"
            % self.nom)

    def detruire(self):
        raise BoiteIndestructible(
            "l'espace '%s' est indestructible" % self.nom)

    # -- persistance ------------------------------------------------------
    def to_dict(self):
        return {"nom": self.nom,
                "entites": [e.to_dict() for e in self.entites.values()],
                "nb_oublies": self.nb_oublies}

    def charger_dict(self, d):
        for de in d.get("entites", []):
            e = Entite.from_dict(de)
            e.espace = self.nom
            self.entites[e.id] = e
            self.region(e.chemin)["_entites"].add(e.id)
            self.axes.update(e.coord.keys())
        self.nb_oublies = d.get("nb_oublies", 0)
        return self

    def __repr__(self):
        return "<Espace %s : %d entites, %d regions, profondeur %d>" % (
            self.nom, len(self.entites), self.nb_regions(), self.profondeur())


# =====================================================================
#  3. ESPACE REEL : ce qui est vraiment arrive
# =====================================================================
class EspaceReel(Espace):
    CONTRAINTE = "coherence : uniquement ce qui est ancre dans le vecu"

    def __init__(self):
        Espace.__init__(self, "reel")
        self.lexique = {}          # mot -> nb de fois REELLEMENT percu
        self.traces = set()

    def accepter(self, entite):
        if entite.origine == "vecu":
            return True
        # la seule breche : un contenu imagine si souvent revisite qu'il
        # est devenu, pour elle, un souvenir. L'origine reste auditable.
        return entite.certitude_realite >= SEUIL_CONFUSION

    def inscrire_vecu(self, signal, etat, saillance, chemin=None):
        mots = _mots(signal)
        for m in mots:
            self.lexique[m] = self.lexique.get(m, 0) + 1
        self.traces.add(str(signal)[:80])
        if chemin is None:
            chemin = ("vecu", mots[0] if mots else "sans_nom")
        intensite = max((abs(v) for v in etat.values()), default=0.0)
        return self.creer(
            {"trace": str(signal)[:80], "fragments": mots},
            origine="vecu", chemin=chemin,
            coord={"intensite": intensite,
                   "valence": sum(etat.values()) / max(1, len(etat))},
            charge=dict(etat), saillance=saillance, certitude_realite=1.0)

    def importer_memoire(self, memoire):
        """Le reel se RECONSTRUIT depuis la memoire autobiographique
        consolidee - il ne la duplique pas."""
        n = 0
        for ep in getattr(memoire, "episodes", []):
            if not ep.get("consolide"):
                continue
            if str(ep.get("signal", ""))[:80] in self.traces:
                continue
            self.inscrire_vecu(ep["signal"], ep["etat"], ep["saillance"])
            n += 1
        return n

    def ancrage(self, mot):
        """Ce mot a-t-il un repondant dans le vecu ?"""
        mot = str(mot).lower()
        if mot in self.lexique:
            for e in self.entites.values():
                if mot in e.fragments():
                    return e
            return True
        return None

    def tirer_temoin(self, sujet):
        """v2.2 -- LABO : un vecu comparable au sujet d'une hypothese,
        pour l'eprouver. Privilegie un vecu qui partage un mot avec lui ;
        a defaut, n'importe quel vecu fait office de temoin."""
        if not self.entites:
            return None
        frags_sujet = set(sujet.fragments())
        proches = [e for e in self.entites.values()
                   if frags_sujet & set(e.fragments())]
        bassin = proches or list(self.entites.values())
        return random.choice(bassin)


# =====================================================================
#  4. ESPACE REVE : aucune contrainte, l'impossible est permis
# =====================================================================
class EspaceReve(Espace):
    CONTRAINTE = "aucune"

    def __init__(self, reel):
        Espace.__init__(self, "reve")
        self.reel = reel
        self.nb_reves = 0

    def _bassin(self):
        return list(self.reel.entites.values()) + list(self.entites.values())

    def rever(self, ennui=0.0, reveur=None, sources=None, mode=None):
        """
        Recombine deux contenus quelconques (vecus et/ou reves) pour
        produire quelque chose qui n'existe pas.

        mode = None       -> choisi selon l'ennui
               "melange"  -> simple recombinaison de fragments reels
               "impossible" -> mot-valise : deux vecus fondus en un seul
                               objet qui n'a jamais existe
               "chimere"  -> forme irrecuperable, non decomposable

        >>> [BRANCHEMENT MODELE #6 - REVEUR]
            reveur(contexte_dict) -> dict  (contenu halluciné par ton LLM).
            Sans LLM : recombinaison procedurale de son PROPRE vecu.
            Le vocabulaire du reve ne contient jamais un mot qu'elle
            n'a pas d'abord percu.
        """
        bassin = list(sources) if sources else self._bassin()
        bassin = [e for e in bassin if e is not None]
        if not bassin:
            return None
        a = random.choice(bassin)
        b = random.choice(bassin)
        fa, fb = a.fragments(), b.fragments()
        vocabulaire = fa + fb or list(self.reel.lexique.keys())
        if not vocabulaire:
            return None

        if mode is None:
            tirage = random.random()
            if tirage < 0.25 + 0.35 * ennui:
                mode = "impossible"
            elif tirage > 0.93:
                mode = "chimere"
            else:
                mode = "melange"

        impossible = False
        axe_invente = None
        if mode == "impossible" and fa and fb:
            ma, mb = random.choice(fa), random.choice(fb)
            valise = ma[:max(2, len(ma) // 2)] + mb[max(1, len(mb) // 2):]
            fragments = [valise] + random.sample(
                vocabulaire, min(1, len(vocabulaire)))
            impossible = True
            axe_invente = valise
        elif mode == "chimere":
            source = random.choice(vocabulaire)
            fragments = ["".join(reversed(source))]
            impossible = True
            axe_invente = fragments[0]
        else:
            k = min(len(vocabulaire), random.randint(2, 3))
            fragments = random.sample(vocabulaire, k)

        contenu = {"trace": " ".join(fragments),
                   "fragments": fragments,
                   "impossible": impossible,
                   "genese": [a.id, b.id],
                   "mode": mode}
        if reveur:                                   # >>> MODELE #6
            try:
                souffle = reveur({"fragments": fragments,
                                  "sources": [a.trace(), b.trace()],
                                  "ennui": round(ennui, 2),
                                  "mode": mode})
                if isinstance(souffle, dict):
                    contenu.update(souffle)
                elif souffle:
                    contenu["trace"] = str(souffle)[:80]
                    contenu["fragments"] = _mots(souffle) or fragments
            except Exception as err:
                contenu["reveur_en_echec"] = str(err)[:60]

        charge = {}
        for cle in set(a.charge) | set(b.charge):
            moyenne = (a.charge.get(cle, 0.0) + b.charge.get(cle, 0.0)) / 2.0
            charge[cle] = _borner(moyenne * (1.1 + 0.4 * ennui)
                                  + random.uniform(-0.12, 0.12))

        coord = {}
        coord.update({k: v for k, v in a.coord.items()})
        for k, v in b.coord.items():
            coord[k] = (coord.get(k, v) + v) / 2.0
        if axe_invente:
            coord[axe_invente] = random.uniform(-9.0, 9.0)   # axe non borne

        self.nb_reves += 1
        chemin = ("reve", "nuit_%d" % (self.nb_reves // 5),
                  fragments[0] if fragments else "sans_forme")
        intensite = max((abs(v) for v in charge.values()), default=0.0)
        e = self.creer(contenu, origine="reve", chemin=chemin, coord=coord,
                       charge=charge,
                       saillance=min(1.0, 0.22 + 0.45 * intensite
                                     + 0.35 * ennui),
                       certitude_realite=0.0)
        if e is not None:
            e.lier(a)
            e.lier(b)
        return e


# =====================================================================
#  5. ESPACE FORGE : prendre une idee et la rendre faisable
# =====================================================================
class Projet:
    """Une idee en cours de reflexion. Chaque etape doit trouver un
    ancrage dans le reel, sinon elle est decomposee, sinon abandonnee."""

    def __init__(self, but, etapes, sources, identifiant=None):
        self.id = identifiant or ("P" + _nouvel_id()[1:])
        self.but = str(but)[:80]
        self.etapes = [{"quoi": str(e), "ancrage": None,
                        "profondeur": 0, "statut": "a_examiner"}
                       for e in etapes]
        self.prerequis = []
        self.obstacles = []
        self.sources = list(sources)
        self.charge = {}
        self.score_faisabilite = 0.0
        self.statut = "brouillon"     # -> raffine -> faisable / abandonne
        self.passes = 0
        self.entite_id = None
        self.cree_a = time.time()

    def ancrees(self):
        return [e for e in self.etapes if e["statut"] == "ancree"]

    def impossibles(self):
        return [e for e in self.etapes if e["statut"] == "impossible"]

    def resume(self):
        return {"id": self.id, "but": self.but, "statut": self.statut,
                "faisabilite": round(self.score_faisabilite, 2),
                "etapes": [e["quoi"] for e in self.etapes],
                "obstacles": self.obstacles[-3:],
                "passes": self.passes}

    def to_dict(self):
        d = dict(self.__dict__)
        return d

    @staticmethod
    def from_dict(d):
        p = Projet(d.get("but", ""), [], d.get("sources", []),
                   identifiant=d.get("id"))
        _reserver_id(p.id)
        p.etapes = d.get("etapes", [])
        p.prerequis = d.get("prerequis", [])
        p.obstacles = d.get("obstacles", [])
        p.charge = d.get("charge", {})
        p.score_faisabilite = d.get("score_faisabilite", 0.0)
        p.statut = d.get("statut", "brouillon")
        p.passes = d.get("passes", 0)
        p.entite_id = d.get("entite_id")
        p.cree_a = d.get("cree_a", time.time())
        return p


class EspaceForge(Espace):
    CONTRAINTE = "faisabilite : chaque etape doit s'ancrer dans le reel"

    def __init__(self, reel):
        Espace.__init__(self, "forge")
        self.reel = reel
        self.projets = {}

    # -- 1. prendre une idee ---------------------------------------------
    def forger(self, sources, but=None):
        sources = [s for s in sources if s is not None]
        if not sources:
            return None
        etapes = []
        for s in sources:
            for f in s.fragments():
                if f not in etapes:
                    etapes.append(f)
        if not etapes:
            return None
        etapes = etapes[:4]
        if but is None:
            but = " ".join(etapes)
        projet = Projet(but, etapes, [s.id for s in sources])
        charge = {}
        for s in sources:
            for k, v in s.charge.items():
                charge[k] = charge.get(k, 0.0) + v / float(len(sources))
        projet.charge = {k: _borner(v) for k, v in charge.items()}

        # une entite compagnon : le projet peut resonner et remonter
        # spontanement dans l'espace global (une idee qui revient)
        e = self.creer({"trace": projet.but, "fragments": etapes,
                        "projet": projet.id},
                       origine="forge",
                       chemin=("forge", "en_cours", projet.id),
                       coord={"faisabilite": 0.0},
                       charge=projet.charge, saillance=0.35)
        if e is not None:
            projet.entite_id = e.id
            for s in sources:
                e.lier(s)
        self.projets[projet.id] = projet
        return projet

    # -- 2. la faire reflechir -------------------------------------------
    def reflechir(self, projet, cycles=2, forgeron=None):
        """
        La boucle de reflexion. Pour chaque etape : y a-t-il quelque chose
        dans le vecu qui puisse la porter ? Sinon, on la casse en morceaux
        plus petits (jusqu'a PROFONDEUR_REFLEXION). Si meme casse elle ne
        tient pas, elle est impossible.

        >>> [BRANCHEMENT MODELE #7 - FORGERON]
            forgeron(resume_projet) -> {"etapes": [...], "prerequis": [...],
                                        "obstacles": [...]}
        """
        for _ in range(max(1, cycles)):
            if projet.statut in ("faisable", "abandonne"):
                break
            projet.passes += 1
            progres = False
            suivantes = []
            for etape in projet.etapes:
                if etape["statut"] == "ancree":
                    suivantes.append(etape)
                    continue
                if etape["statut"] == "impossible":
                    suivantes.append(etape)
                    continue
                anc = self.reel.ancrage(etape["quoi"])
                if anc is not None:
                    etape["statut"] = "ancree"
                    etape["ancrage"] = getattr(anc, "id", "lexique")
                    progres = True
                    suivantes.append(etape)
                    continue
                morceaux = self._decomposer(etape["quoi"])
                if morceaux and etape["profondeur"] < PROFONDEUR_REFLEXION:
                    projet.obstacles.append(
                        "'%s' n'existe pas dans mon vecu -> je le decompose"
                        % etape["quoi"])
                    for m in morceaux:
                        if any(s["quoi"] == m for s in suivantes):
                            continue
                        suivantes.append({"quoi": m, "ancrage": None,
                                          "profondeur": etape["profondeur"] + 1,
                                          "statut": "a_examiner"})
                    progres = True
                else:
                    etape["statut"] = "impossible"
                    projet.obstacles.append(
                        "'%s' reste sans ancrage, je ne sais pas le faire"
                        % etape["quoi"])
                    suivantes.append(etape)
            projet.etapes = suivantes

            if forgeron:                                # >>> MODELE #7
                try:
                    retour = forgeron(projet.resume())
                    if isinstance(retour, dict):
                        for sup in retour.get("etapes", []):
                            if not any(s["quoi"] == sup for s in projet.etapes):
                                projet.etapes.append(
                                    {"quoi": str(sup), "ancrage": None,
                                     "profondeur": 1, "statut": "a_examiner"})
                                progres = True
                        projet.prerequis.extend(retour.get("prerequis", []))
                        projet.obstacles.extend(retour.get("obstacles", []))
                except Exception as err:
                    projet.obstacles.append("forgeron en echec: %s"
                                            % str(err)[:50])

            projet.score_faisabilite = self._faisabilite(projet)
            if projet.score_faisabilite >= 0.999:
                projet.statut = "faisable"
                break
            if not progres:
                projet.statut = ("abandonne"
                                 if projet.score_faisabilite < 0.6
                                 else "raffine")
                break
            projet.statut = "raffine"

        if projet.statut not in ("faisable", "abandonne"):
            if projet.score_faisabilite >= 0.999:
                projet.statut = "faisable"
            elif projet.impossibles() and projet.score_faisabilite < 0.6:
                projet.statut = "abandonne"

        e = self.entites.get(projet.entite_id)
        if e is not None:
            e.muter({"coord": {"faisabilite": projet.score_faisabilite},
                     "statut_projet": projet.statut,
                     "saillance": min(1.0, 0.3 + projet.score_faisabilite * 0.5)})
        return projet

    def _decomposer(self, mot):
        """Derriere un objet impossible se cachent souvent deux choses
        possibles. Elle cherche la couture."""
        mot = str(mot).lower()
        reels = list(self.reel.lexique.keys())
        if not reels:
            return []
        for i in range(2, max(3, len(mot) - 1)):
            gauche, droite = mot[:i], mot[i:]
            candidats_g = [w for w in reels if w.startswith(gauche)]
            candidats_d = [w for w in reels if w.endswith(droite)]
            if candidats_g and candidats_d and candidats_g[0] != candidats_d[0]:
                return [candidats_g[0], candidats_d[0]]
        return []

    def _faisabilite(self, projet):
        if not projet.etapes:
            return 0.0
        part = len(projet.ancrees()) / float(len(projet.etapes))
        if projet.prerequis:
            tenus = sum(1 for p in projet.prerequis
                        if self.reel.ancrage(p) is not None)
            part *= 0.5 + 0.5 * (tenus / float(len(projet.prerequis)))
        return round(part, 3)

    # -- 3. repeter mentalement -------------------------------------------
    def repeter_mentalement(self, empreinte_fn, projet=None):
        """Rejouer une chose dans sa tete avant de la vivre : quand elle
        arrive pour de bon, elle surprend moins. C'est la boite qui sert
        de simulateur au modele predictif."""
        if empreinte_fn is None:
            return None
        if projet is None:
            ouverts = [p for p in self.projets.values()
                       if p.statut in ("faisable", "raffine")]
            if not ouverts:
                return None
            projet = max(ouverts, key=lambda p: p.score_faisabilite)
        if not projet.charge:
            return None
        confiance = 0.25 + 0.55 * projet.score_faisabilite
        return {"empreinte": empreinte_fn(projet.but),
                "but": projet.but,
                "etat_attendu": dict(projet.charge),
                "confiance": round(confiance, 2),
                "projet": projet.id}

    # -- v2.2 : le labo peut rouvrir un projet qu'elle avait abandonne -----
    def debloquer_par_preuve(self, entite_id, paillasse):
        """Une hypothese testee au labo se confirme : tout projet
        abandonne qui en descendait merite un second regard."""
        debloques = []
        for p in self.projets.values():
            if entite_id in p.sources and p.statut == "abandonne":
                p.statut = "raffine"
                p.obstacles.append(
                    "preuve du labo (%s) : reouverture" % paillasse.id)
                debloques.append(p.id)
        return debloques

    # -- oubli : une idee dont plus rien ne porte la trace s'efface --------
    def oublier(self, plafond=PLAFOND_ENTITES):
        oublies = Espace.oublier(self, plafond)
        perdus = [i for i, p in self.projets.items()
                  if p.entite_id is not None
                  and p.entite_id not in self.entites]
        for i in perdus:
            del self.projets[i]
        return oublies + len(perdus)

    # -- persistance --------------------------------------------------------
    def to_dict(self):
        d = Espace.to_dict(self)
        d["projets"] = [p.to_dict() for p in self.projets.values()]
        return d

    def charger_dict(self, d):
        Espace.charger_dict(self, d)
        for dp in d.get("projets", []):
            p = Projet.from_dict(dp)
            self.projets[p.id] = p
        return self


# =====================================================================
#  6. LE CORPS : elle n'est plus un point de vue fixe, elle est
#     QUELQUE PART -- et ou elle est determine ce qu'elle entend  (v2.2)
# =====================================================================
class Corps:
    """Le point depuis lequel la boite est vecue.

    - marcher()      : lente, elle TRAVERSE -- seul mode qui produit de
                        la serendipite (deux choses croisees peuvent se
                        coller en reve).
    - sauter()       : directe, couteuse, aveugle -- rien traverse.
    - teleporter()   : reservee a l'urgence -- laisse une dechirure
                        locale (elle arrive desorientee).
    """

    SEUIL_SAUT = 0.70        # saillance requise pour se payer un saut
    SEUIL_URGENCE = 0.85     # au-dela : teleportation autorisee
    COUT_PAS = 0.01
    COUT_SAUT = 0.22
    COUT_TELEPORT = 0.55

    def __init__(self, boite):
        self.boite = boite
        self.pos = Coord({a: 0.0 for a in AXES_BASE})
        self.espace = "reel"
        self.charge = 0.0          # fatigue de deplacement, non semantique
        self.itineraire = []
        self.vitesse = 0.45

    # ---------------------------------------------------------------
    def aller_vers(self, cible, urgence=0.0, saillance=0.0, motif=""):
        """Choisit seule son mode. Elle ne se teleporte pas par confort."""
        d = self.pos.distance(cible)
        if urgence >= self.SEUIL_URGENCE:
            return self.teleporter(cible, motif)
        if saillance >= self.SEUIL_SAUT and (d is None or d > 2.5):
            return self.sauter(cible, motif)
        return self.marcher(cible, motif)

    # ---------------------------------------------------------------
    def marcher(self, cible, motif=""):
        """Lente. Elle TRAVERSE. C'est le seul mode qui produit des
        rencontres non cherchees."""
        traverses = []
        espace = self.boite.espace(self.espace)
        if espace is None:
            return traverses
        # elle ne peut mesurer un ecart que sur les axes qu'elle habite deja :
        # avant de marcher, elle prend pied (a 0.0) sur tout axe de la cible
        # qu'elle n'avait pas encore -- sinon un axe tout neuf serait invisible
        # a la distance (0 axe commun de plus => elle croirait deja y etre).
        for a in cible.axes():
            self.pos.v.setdefault(a, 0.0)
        garde = 0
        while garde < 400:
            garde += 1
            d = self.pos.distance(cible)
            if d is None or d < self.vitesse:
                break
            for a in cible.axes():
                depart = self.pos.v.get(a, 0.0)
                pas = (cible.v[a] - depart) / max(d, 1e-6) * self.vitesse
                self.pos.v[a] = depart + pas
                axe = self.boite.axe(a)
                if axe:
                    axe.emprunter(abs(pas))
            self.charge += self.COUT_PAS
            for e in espace.voisinage(self.pos, rayon=0.9):
                if e.id not in [t.id for t in traverses]:
                    e.toucher("traversee")
                    e.saillance = min(1.0, e.saillance + 0.02)
                    traverses.append(e)
        self._journal("marche", cible, motif, len(traverses))
        self._serendipite(traverses)
        return traverses

    def _serendipite(self, traverses):
        """Deux choses traversees coup sur coup peuvent se coller.
        C'est le benefice de la lenteur."""
        for i in range(len(traverses) - 1):
            a, b = traverses[i], traverses[i + 1]
            if random.random() < 0.06:
                e = self.boite.reve.rever(sources=[a, b], mode="melange")
                if e is not None:
                    e.contenu["cause"] = "rencontre_en_marchant"

    # ---------------------------------------------------------------
    def sauter(self, cible, motif=""):
        """Direct, cher, aveugle : elle n'a rien vu du chemin."""
        self.pos = cible.copie()
        self.charge += self.COUT_SAUT
        self._journal("saut", cible, motif, 0)
        return []

    def teleporter(self, cible, motif=""):
        """URGENCE UNIQUEMENT. Laisse une dechirure locale : elle arrive
        desorientee."""
        self.pos = cible.copie()
        self.charge += self.COUT_TELEPORT
        espace = self.boite.espace(self.espace)
        if espace is not None:
            for e in espace.voisinage(self.pos, rayon=1.5):
                e.certitude_realite = max(0.0, e.certitude_realite - 0.05)
                e.tracer("dechirure", {"motif": motif})
        self._journal("teleportation", cible, motif, 0)
        return []

    # ---------------------------------------------------------------
    def marcher_sur_axe(self, nom_axe, delta, motif=""):
        """Se deplacer le long d'un axe QU'ELLE A INVENTE."""
        axe = self.boite.axe(nom_axe)
        if axe is None or axe.dormant:
            return None                  # axe pas (ou plus) praticable
        cible = self.pos.copie()
        cible.v[nom_axe] = cible.v.get(nom_axe, 0.0) + delta
        return self.marcher(cible, motif or ("exploration:%s" % nom_axe))

    def changer_espace(self, nom, urgence=0.0):
        """Passer d'un espace a l'autre est aussi un deplacement.
        nom=None -> le Hub (spawn), toujours valide."""
        if nom is not None and self.boite.espace(nom) is None:
            return False
        self.charge += 0.08
        self.espace = nom
        self.boite.tracer("seuil_franchi", {"vers": nom if nom is not None else "hub"})
        return True

    # -- Hub (spawn) : sas neutre entre les 4 pieces, ajoute en v2.3 --
    # Regle voulue : piece -> piece = TOUJOURS une teleportation directe.
    # piece -> hub = teleportation. hub -> piece = MARCHE uniquement
    # (c'est le seul sens ou elle "traverse" quelque chose, meme si le
    # hub lui-meme est vide -- coherent avec marcher() = seul mode qui
    # produit de la serendipite).
    def teleporter_vers_espace(self, nom, motif=""):
        """Piece -> piece, directement, SANS repasser par le Hub.
        Toujours une teleportation : couteuse, aveugle du trajet,
        laisse une dechirure locale dans la piece quittee."""
        if self.espace is None:
            return False  # depuis le hub : il faut entrer_dans(), pas teleporter
        if self.boite.espace(nom) is None:
            return False  # piece inconnue
        self._dechirer_ici(motif or "teleportation_directe")
        self.charge += self.COUT_TELEPORT
        self.espace = nom
        self.pos = Coord({a: 0.0 for a in AXES_BASE})
        self._journal("teleportation_espace", self.pos, motif, 0)
        self.boite.tracer("seuil_franchi", {"vers": nom, "mode": "teleportation"})
        return True

    def teleporter_vers_hub(self, motif="retour_hub"):
        """Piece -> Hub. Toujours une teleportation. Le Hub n'a ni
        entites ni coordonnees propres : juste le sas entre les 4
        pieces."""
        if self.espace is None:
            return False  # deja au hub
        self._dechirer_ici(motif)
        self.charge += self.COUT_TELEPORT
        self._journal("teleportation_hub", self.pos, motif, 0)
        self.espace = None
        self.pos = Coord({a: 0.0 for a in AXES_BASE})
        self.boite.tracer("seuil_franchi", {"vers": "hub", "mode": "teleportation"})
        return True

    def entrer_dans(self, nom, motif=""):
        """Hub -> piece. MARCHE uniquement (jamais de teleportation
        dans ce sens) : quelques pas d'approche, cout progressif."""
        if self.espace is not None:
            return False  # deja dans une piece : teleporter_vers_espace() d'abord
        if self.boite.espace(nom) is None:
            return False
        n_pas = random.randint(3, 8)
        self.charge += self.COUT_PAS * n_pas
        self.espace = nom
        self.pos = Coord({a: 0.0 for a in AXES_BASE})
        self._journal("entree_piece", self.pos, motif, 0)
        self.boite.tracer("seuil_franchi", {"vers": nom, "mode": "marche"})
        return True

    def _dechirer_ici(self, motif):
        """Laisse une petite dechirure locale (baisse de certitude)
        dans la piece qu'elle quitte -- meme effet que teleporter()."""
        espace = self.boite.espace(self.espace)
        if espace is not None:
            for e in espace.voisinage(self.pos, rayon=1.5):
                e.certitude_realite = max(0.0, e.certitude_realite - 0.05)
                e.tracer("dechirure", {"motif": motif})

    def recuperer(self, taux=0.015):
        self.charge = max(0.0, self.charge - taux)

    def _journal(self, mode, cible, motif, n):
        self.itineraire.append({"t": time.time(), "mode": mode,
                                "espace": self.espace, "motif": motif,
                                "traverses": n,
                                "dim_parcourue": len(cible.axes())})
        if len(self.itineraire) > 800:
            self.itineraire.pop(0)

    def __repr__(self):
        return "<Corps espace=%s pos=%r charge=%.2f>" % (
            self.espace, self.pos.v, self.charge)


# =====================================================================
#  7. LE LABO : le 4e espace -- pas un espace de rangement, un espace
#     d'OPERATION. Ce qui s'y passe ne contamine ni le reel ni la
#     memoire tant qu'il n'y a pas verdict (sous scelle).  (v2.2)
# =====================================================================
class Paillasse:
    """Une manip en cours sur une hypothese."""

    _n = 0

    def __init__(self, hypothese_id, sujet, question):
        Paillasse._n += 1
        self.id = "paillasse_%d" % Paillasse._n
        self.hypothese_id = hypothese_id
        self.sujet = sujet
        self.question = question
        self.registre = []
        self.score = 0.0
        self.verdict = None
        self.ferme = False
        self.ouverte_le = time.time()

    def consigner(self, type_, donnees):
        self.registre.append({"t": time.time(), "type": type_,
                              "donnees": donnees})

    def resume(self):
        return {"id": self.id, "question": self.question,
                "score": round(self.score, 3), "verdict": self.verdict,
                "operations": len(self.registre),
                "duree": round(time.time() - self.ouverte_le, 1)}


class EspaceTest(Espace):
    """LE LABO. On n'y range rien : on y EPROUVE. Tout ce qui y entre
    est mis SOUS SCELLE."""

    CONTRAINTE = "scellement : rien n'en sort sans verdict"

    def __init__(self, boite):
        Espace.__init__(self, "test")
        self.boite = boite
        self.paillasses = {}
        self.archives = []

    # ---- 1. reperer ce qui merite le labo --------------------------
    def capturer_hypotheses(self, seuil=0.4):
        """Une hypothese = une entite qui porte une tension non
        resolue : erreur de prediction, doute, obstacle de forge,
        question qui revient, faux souvenir suspect."""
        trouvees = []
        for nom in ("reel", "reve", "forge"):
            for e in self.boite.espace(nom).entites.values():
                t = self.tension(e)
                if t >= seuil:
                    trouvees.append((t, e))
        trouvees.sort(key=lambda x: -x[0])
        return trouvees

    def tension(self, e):
        doute = 1.0 - abs(e.certitude_realite - 0.5) * 2.0
        surpr = getattr(e, "surprise", 0.0)
        bloque = 1.0 if getattr(e, "bloquee", False) else 0.0
        recur = min(1.0, e.nb_revisites / 5.0)
        suspect = 1.0 if (e.origine == "reve"
                          and e.certitude_realite > 0.7) else 0.0
        return max(doute, surpr, bloque, recur, suspect) * (0.5 + e.saillance)

    # ---- 2. poser la manip ------------------------------------------
    def ouvrir_paillasse(self, hypothese, question=None):
        copie = hypothese.cloner()
        copie.sous_scelle = True
        copie.origine_test = hypothese.id
        self.accueillir(copie)
        p = Paillasse(hypothese_id=hypothese.id, sujet=copie,
                      question=question
                      or ("est-ce que '%s' tient ?" % hypothese.trace()))
        self.paillasses[p.id] = p
        self.boite.tracer("paillasse_ouverte", {"sur": hypothese.id})
        return p

    # ---- 3. les operations du labo -----------------------------------
    def melanger(self, p, a, b):
        """Le coeur du labo : elle teste les melanges."""
        axes_communs = a.coord.axes() & b.coord.axes()
        frags_communs = set(a.fragments()) & set(b.fragments())
        ecart_dim = abs(a.coord.dim() - b.coord.dim())
        compat = (len(axes_communs) * 0.2 + len(frags_communs) * 0.15
                  - ecart_dim * 0.1 + random.uniform(-0.15, 0.15))

        if compat < 0.05:
            issue, produit = "inerte", None
        elif compat < 0.4:
            issue, produit = "instable", self._hybride(a, b, tenue=0.3)
        elif compat < 0.75:
            issue, produit = "hybride_stable", self._hybride(a, b, tenue=0.8)
        else:
            issue = "reaction"
            produit = self._hybride(a, b, tenue=1.0)
            # une reaction ouvre une DIRECTION qui n'existait pas
            nouvel_axe = self.boite.inventer_axe(
                nom="axe_%s_%s" % (a.mot_cle(), b.mot_cle()),
                origine="test", ne_de=produit.id)
            if nouvel_axe is not None:
                produit.redimensionner(
                    list(produit.coord.axes()) + [nouvel_axe.nom],
                    raison="reaction_labo")

        p.consigner("melange", {"a": a.id, "b": b.id,
                                "compat": round(compat, 3), "issue": issue})
        if produit is not None:
            produit.sous_scelle = True
            if issue != "inerte":
                produit.ne_du_melange = True
            self.accueillir(produit)
        return issue, produit

    def _hybride(self, a, b, tenue=0.5):
        fa, fb = a.fragments(), b.fragments()
        fragments = list(dict.fromkeys(fa[:2] + fb[:2])) or ["hybride"]
        contenu = {"trace": " ".join(fragments), "fragments": fragments,
                  "melange_de": [a.id, b.id], "tenue": round(tenue, 2)}
        charge = {}
        for cle in set(a.charge) | set(b.charge):
            charge[cle] = _borner((a.charge.get(cle, 0.0)
                                   + b.charge.get(cle, 0.0)) / 2.0)
        coord = a.coord.copie()
        for k, v in b.coord.items():
            coord.v.setdefault(k, v)
        e = Entite(contenu, origine="test",
                   chemin=("test", "melange", "%s_%s" % (a.id, b.id)),
                   coord=coord, charge=charge,
                   saillance=min(1.0, 0.25 + 0.5 * tenue),
                   certitude_realite=0.0)
        e.lier(a)
        e.lier(b)
        return e

    def decrypter(self, entite, profondeur=3):
        """Un vrai labo qui decrypte tout si elle le decide : ses
        sources, ses fragments, sa lignee, son historique de
        dimension."""
        rapport = {
            "id": entite.id, "libelle": entite.trace(),
            "origine": entite.origine, "certitude": entite.certitude_realite,
            "dim": entite.coord.dim(),
            "trajectoire_dim": list(entite.historique_dim),
            "axes": sorted(entite.coord.axes()),
            "fragments": entite.fragments(),
            "traces": list(entite.traces[-20:]),
            "lignee": [],
        }
        vus, file = set(), list(entite.sources)
        for _ in range(profondeur):
            suivants = []
            for sid in file:
                if sid in vus:
                    continue
                vus.add(sid)
                src = self.boite.retrouver(sid)
                if src:
                    rapport["lignee"].append(
                        {"id": sid, "libelle": src.trace(),
                         "origine": src.origine, "dim": src.coord.dim()})
                    suivants += list(src.sources)
            file = suivants
        entite.tracer("decryptee", {"profondeur": profondeur})
        return rapport

    def eprouver(self, p, essais=12):
        """Rejoue l'hypothese contre le reel : chaque essai la confronte
        a un vecu reellement inscrit. Statistique, pas verdict d'auteur."""
        reel = self.boite.reel
        succes = 0
        for _ in range(essais):
            temoin = reel.tirer_temoin(p.sujet)
            if temoin is None:
                p.consigner("essai", {"resultat": "sans_temoin"})
                continue
            d = p.sujet.coord.distance(temoin.coord)
            accord = 1.0 if d is None else 1.0 / (1.0 + d)
            accord *= (0.5 + temoin.certitude_realite)
            ok = accord > 0.5
            succes += 1 if ok else 0
            p.consigner("essai", {"temoin": temoin.id,
                                  "accord": round(accord, 3), "ok": ok})
        p.score = succes / max(1, essais)
        return p.score

    # ---- 4. verdict et sortie de scelle -------------------------------
    def clore(self, p):
        inattendu = any(c["donnees"].get("issue") == "reaction"
                        for c in p.registre if c["type"] == "melange")
        if inattendu:
            p.verdict = "effet_inattendu"
        elif p.score >= 0.7:
            p.verdict = "confirme"
        elif p.score <= 0.3:
            p.verdict = "refute"
        else:
            p.verdict = "indetermine"

        source = self.boite.retrouver(p.hypothese_id)
        if p.verdict == "confirme" and source:
            source.certitude_realite = min(1.0, source.certitude_realite
                                           + 0.25)
            source.tracer("teste", {"verdict": "confirme", "score": p.score})
            self.boite.forge.debloquer_par_preuve(source.id, p)
        elif p.verdict == "refute" and source:
            source.certitude_realite = max(0.0, source.certitude_realite
                                           - 0.30)
            source.tracer("teste", {"verdict": "refute", "score": p.score})
            source.etiqueter("eprouvee_negative")   # refutee, pas detruite
        elif p.verdict == "effet_inattendu":
            for e in list(self.entites.values()):
                if e.sous_scelle and getattr(e, "ne_du_melange", False):
                    e.sous_scelle = False
                    self.boite.migrer(e.id, "reve")   # ca part en reve
        p.ferme = True
        self.archives.append(p.resume())
        if len(self.archives) > PLAFOND_JOURNAL:
            self.archives.pop(0)
        self.boite.tracer("verdict", p.resume())
        return p.verdict


# =====================================================================
#  8. LA BOITE : indestructible, persistante
# =====================================================================
class BoiteInfinie:
    """
    Elle peut TOUT faire dedans : creer, modeler, deplacer, effacer.
    Elle ne peut pas faire disparaitre la boite ni un de ses espaces.
    """

    INDESTRUCTIBLE = True
    _PROTEGES = ("reel", "reve", "forge", "test", "verrou")

    def __init__(self, chemin_persistance=None, empreinte=None,
                 reveur=None, forgeron=None):
        object.__setattr__(self, "verrou", threading.RLock())
        self.reel = EspaceReel()
        self.reve = EspaceReve(self.reel)
        self.forge = EspaceForge(self.reel)
        self.test = EspaceTest(self)              # v2.2 -- le labo
        self.empreinte = empreinte
        self.reveur = reveur
        self.forgeron = forgeron
        self.chemin_persistance = chemin_persistance
        self.journal = deque(maxlen=PLAFOND_JOURNAL)
        self.nee_a = time.time()
        self.battements = 0
        self.nb_faux_souvenirs = 0
        self.tentatives_destruction = 0
        # v2.2 -- geometrie : le registre d'axes de LA BOITE (a ne pas
        # confondre avec Espace.axes, qui est juste l'ensemble des noms
        # d'axes utilises DANS un espace donne). Ici : x/y/z + tout axe
        # ne d'un reve, d'une forge, d'une reaction du labo.
        self.axes = {n: Axe(n, origine="base") for n in AXES_BASE}
        self.corps = Corps(self)                   # v2.2 -- elle est quelque part
        recharge = False
        if chemin_persistance and os.path.exists(chemin_persistance):
            recharge = self.charger()
        self.tracer("reveil" if recharge else "naissance",
                    chemin_persistance or "sans persistance")
        if chemin_persistance:
            atexit.register(self._sauvegarde_finale)

    # -- indestructibilite -------------------------------------------------
    def __setattr__(self, nom, valeur):
        if nom in BoiteInfinie._PROTEGES and getattr(self, nom, None) is not None:
            object.__setattr__(self, "tentatives_destruction",
                               getattr(self, "tentatives_destruction", 0) + 1)
            raise BoiteIndestructible(
                "'%s' ne peut pas etre remplace : la boite est indestructible"
                % nom)
        object.__setattr__(self, nom, valeur)

    def __delattr__(self, nom):
        if nom in BoiteInfinie._PROTEGES:
            object.__setattr__(self, "tentatives_destruction",
                               getattr(self, "tentatives_destruction", 0) + 1)
            raise BoiteIndestructible(
                "'%s' ne peut pas etre supprime : la boite est indestructible"
                % nom)
        object.__delattr__(self, nom)

    def detruire(self):
        self.tentatives_destruction += 1
        self.tracer("tentative_destruction", "refusee")
        raise BoiteIndestructible(
            "la boite ne peut pas etre detruite. Elle peut etre videe.")

    def clear(self):
        return self.detruire()

    def vider(self):
        """Elle a le droit de tout effacer DEDANS. Les trois espaces
        restent debout, vides."""
        with self.verrou:
            n = sum(esp.vider() for esp in self.espaces().values())
            self.forge.projets = {}
            self.test.paillasses = {}
            self.tracer("grand_vide", "%d contenus effaces" % n)
            return n

    # -- acces --------------------------------------------------------------
    def espaces(self):
        return {"reel": self.reel, "reve": self.reve, "forge": self.forge,
                "test": self.test}

    def espace(self, nom):
        return self.espaces().get(nom)

    def retrouver(self, identifiant):
        """Retrouve une entite ou qu'elle soit dans la boite."""
        for esp in self.espaces().values():
            e = esp.entites.get(identifiant)
            if e is not None:
                return e
        return None

    def tracer(self, quoi, detail=""):
        self.journal.append({"quand": time.time(), "quoi": quoi,
                             "detail": str(detail)[:100]})

    # -- geometrie : le registre d'axes (v2.2) -------------------------------
    def axe(self, nom):
        return self.axes.get(nom)

    def inventer_axe(self, nom, origine, ne_de):
        """Une direction nouvelle ne se decrete pas : elle NAIT d'un
        reve, d'une forge ou d'une reaction de labo."""
        if origine == "base" or ne_de is None:
            return None
        if nom in self.axes:
            self.axes[nom].emprunter(0.1)
            return self.axes[nom]
        a = Axe(nom, origine=origine, ne_de=ne_de)
        self.axes[nom] = a
        self.tracer("axe_ne", {"nom": nom, "origine": origine,
                               "de": ne_de, "dim_boite": len(self.axes)})
        return a

    # -- migrations ---------------------------------------------------------
    def migrer(self, identifiant, cible):
        """reve -> forge -> reel. L'origine ne bouge JAMAIS (audit) ;
        seul 'trajet' s'allonge."""
        with self.verrou:
            source = None
            for esp in self.espaces().values():
                if identifiant in esp.entites:
                    source = esp
                    break
            destination = self.espace(cible)
            if source is None or destination is None or source is destination:
                return None
            e = source.retirer(identifiant)
            accueilli = destination.accueillir(e)
            if accueilli is None:
                source.accueillir(e)          # refus : il revient chez lui
                self.tracer("migration_refusee",
                            "%s %s->%s" % (identifiant, source.nom, cible))
                return None
            self.tracer("migration", "%s %s->%s (origine %s)"
                        % (identifiant, source.nom, cible, e.origine))
            if destination is self.reel and e.origine != "vecu":
                self.nb_faux_souvenirs += 1
                self.tracer("faux_souvenir", e.trace())
            return accueilli

    # -- sondage : ce qui remonte a la surface ------------------------------
    def sonder(self, etat, limite=3):
        with self.verrou:
            trouves = []
            for esp, etiquette in ((self.reve, "reverie"),
                                   (self.forge, "intuition"),
                                   (self.reel, "faux_souvenir"),
                                   (self.test, "hypothese")):
                for r in esp.resonance(etat, depuis=self.corps.pos)[:limite]:
                    e = r["entite"]
                    if esp is self.reel and e.origine == "vecu":
                        continue          # deja couvert par la memoire
                    trouves.append({"source": etiquette, "entite": e,
                                    "score": r["score"]})
            trouves.sort(key=lambda t: -t["score"])
            return trouves[:limite]

    # -- un tour de vie interieure -------------------------------------------
    def habiter(self, cycles=1, intensite=1.0, ennui=0.0, memoire=None):
        pensees = []
        with self.verrou:
            for _ in range(max(1, cycles)):
                self.battements += 1

                if memoire is not None:
                    self.reel.importer_memoire(memoire)

                for esp in self.espaces().values():
                    esp.oublier()

                # elle reve
                if random.random() < 0.35 + 0.5 * ennui * intensite:
                    e = self.reve.rever(ennui, self.reveur)
                    if e is not None:
                        pensees.append({"quoi": "reve", "id": e.id,
                                        "contenu": e.trace(),
                                        "impossible": bool(
                                            e.contenu.get("impossible"))})

                # une idee assez forte part a la forge
                candidats = [e for e in self.reve.entites.values()
                             if e.saillance > 0.42]
                if candidats and random.random() < 0.45 * intensite:
                    graine = max(candidats, key=lambda e: e.saillance)
                    projet = self.forge.forger([graine])
                    if projet is not None:
                        self.tracer("idee_forgee", projet.but)
                        pensees.append({"quoi": "idee", "id": projet.id,
                                        "contenu": projet.but})

                # elle reflechit a ce qui traine
                ouverts = [p for p in self.forge.projets.values()
                           if p.statut in ("brouillon", "raffine")]
                if ouverts:
                    p = min(ouverts, key=lambda p: p.passes)
                    self.forge.reflechir(
                        p, cycles=2 if intensite >= 1.0 else 1,
                        forgeron=self.forgeron)
                    pensees.append({"quoi": "reflexion", "id": p.id,
                                    "contenu": p.but, "statut": p.statut,
                                    "faisabilite": p.score_faisabilite})

                # elle y repense... et la frontiere se brouille
                if self.reve.entites and random.random() < 0.5:
                    e = max(self.reve.entites.values(),
                            key=lambda x: x.saillance)
                    e.revisiter()
                    if e.certitude_realite >= SEUIL_CONFUSION:
                        if self.migrer(e.id, "reel") is not None:
                            pensees.append({"quoi": "faux_souvenir",
                                            "id": e.id,
                                            "contenu": e.trace()})
        return pensees

    # -- vie interieure SITUEE : le corps se deplace  (v2.2) -----------------
    def vie_interieure(self, etat_interne=None, urgence=0.0):
        """A appeler EN PLUS de habiter() : habiter() fait vivre le
        contenu (reves, forge), vie_interieure() fait vivre le
        DEPLACEMENT a travers ce contenu -- ce qui determine ce qui
        gagne son attention. Retourne ce que la resonance situee a
        perçu (voir Espace.resonance)."""
        with self.verrou:
            etat_interne = dict(etat_interne or {})
            c = self.corps
            c.recuperer()
            for a in self.axes.values():
                a.eroder()

            espace = self.espace(c.espace)
            percu = espace.resonance(etat_interne, depuis=c.pos) if espace else []

            urg = max(0.0, min(1.0, float(urgence)))
            if percu:
                tete = percu[0]
                if tete["distance"] is not None and tete["distance"] > 1.2:
                    c.aller_vers(tete["entite"].coord, urgence=urg,
                                saillance=tete["entite"].saillance,
                                motif="appel")
            else:
                # rien n'accroche ici : errance, et l'errance fait rencontrer
                derive = c.pos.copie()
                axes_dispo = sorted(c.pos.axes())
                for a in random.sample(axes_dispo, k=min(2, len(axes_dispo))):
                    derive.v[a] += random.uniform(-1.6, 1.6)
                c.marcher(derive, motif="errance")

            # le labo s'ouvre tout seul quand une tension devient forte
            hyps = self.test.capturer_hypotheses(seuil=0.55)
            if hyps and random.random() < 0.25:
                _tension, h = hyps[0]
                espace_avant = c.espace
                c.changer_espace("test")
                c.aller_vers(h.coord, saillance=h.saillance, motif="hypothese")
                p = self.test.ouvrir_paillasse(h)
                voisins = self.test.voisinage(c.pos, rayon=3.0)
                for v in voisins[:3]:
                    self.test.melanger(p, p.sujet, v)
                self.test.eprouver(p)
                self.test.clore(p)
                c.changer_espace(espace_avant)

            return percu

    # -- statistiques (matiere a traits emergents) ---------------------------
    def statistiques(self):
        with self.verrou:
            projets = list(self.forge.projets.values())
            faisables = [p for p in projets if p.statut == "faisable"]
            abandonnes = [p for p in projets if p.statut == "abandonne"]
            total = sum(len(e.entites) for e in self.espaces().values())
            duree = max(1.0, time.time() - self.nee_a)
            doutes = sum(1 for esp in self.espaces().values()
                         for e in esp.entites.values() if e.doute())
            return {
                "nb_entites": total,
                "reel": len(self.reel.entites),
                "reve": len(self.reve.entites),
                "forge": len(self.forge.entites),
                "test": len(self.test.entites),
                "nb_projets": len(projets),
                "projets_faisables": len(faisables),
                "projets_abandonnes": len(abandonnes),
                "faux_souvenirs": self.nb_faux_souvenirs,
                "contenus_douteux": doutes,
                "fecondite": round(self.reve.nb_reves / duree, 3),
                "realisme": round(len(faisables) / float(max(1, len(projets))), 2),
                "porosite": round(self.nb_faux_souvenirs
                                  / float(max(1, self.reve.nb_reves)), 2),
                "profondeur_max": max(e.profondeur()
                                      for e in self.espaces().values()),
                "axes_inventes": len(set().union(
                    *[esp.axes for esp in self.espaces().values()])),
                "battements": self.battements,
                # v2.2 -- geometrie / labo / corps
                "dimensions_boite": len(self.axes),
                "axes_dormants": sum(1 for a in self.axes.values()
                                     if a.dormant),
                "paillasses_ouvertes": sum(1 for p in self.test.paillasses.values()
                                          if not p.ferme),
                "paillasses_closes": len(self.test.archives),
                "corps_espace": self.corps.espace,
                "corps_position": dict(self.corps.pos.v),
                "corps_charge": round(self.corps.charge, 3),
            }

    def projets_faisables(self):
        with self.verrou:
            return [p.resume() for p in self.forge.projets.values()
                    if p.statut == "faisable"]

    # -- persistance ----------------------------------------------------------
    def to_dict(self):
        return {"version": 2, "nee_a": self.nee_a,
                "battements": self.battements,
                "nb_faux_souvenirs": self.nb_faux_souvenirs,
                "lexique": self.reel.lexique,
                "traces": sorted(self.reel.traces),
                "espaces": {n: e.to_dict()
                            for n, e in self.espaces().items()},
                "axes": {n: a.to_dict() for n, a in self.axes.items()},
                "corps": {"pos": dict(self.corps.pos.v),
                         "espace": self.corps.espace,
                         "charge": self.corps.charge},
                "journal": list(self.journal)[-50:]}

    def sauvegarder(self, chemin=None):
        chemin = chemin or self.chemin_persistance
        if not chemin:
            return False
        with self.verrou:
            donnees = self.to_dict()
        temporaire = "%s.tmp%d" % (chemin, os.getpid())
        try:
            with open(temporaire, "w", encoding="utf-8") as f:
                json.dump(donnees, f, ensure_ascii=True, indent=1)
            os.replace(temporaire, chemin)
            return True
        except Exception as err:
            self.tracer("sauvegarde_en_echec", str(err)[:80])
            try:
                if os.path.exists(temporaire):
                    os.remove(temporaire)
            except OSError:
                pass
            return False

    def charger(self, chemin=None):
        """Tolerant : un fichier corrompu ne tue pas la boite, elle repart
        avec ses trois espaces intacts et une trace dans son journal."""
        chemin = chemin or self.chemin_persistance
        if not chemin or not os.path.exists(chemin):
            return False
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                d = json.load(f)
            with self.verrou:
                for nom, esp in self.espaces().items():
                    bloc = d.get("espaces", {}).get(nom)
                    if bloc:
                        esp.charger_dict(bloc)
                self.reel.lexique = dict(d.get("lexique", {}))
                self.reel.traces = set(d.get("traces", []))
                self.nee_a = d.get("nee_a", self.nee_a)
                self.battements = d.get("battements", 0)
                self.nb_faux_souvenirs = d.get("nb_faux_souvenirs", 0)
                for entree in d.get("journal", []):
                    self.journal.append(entree)
                # les axes ne sont jamais detruits : on les restaure tels
                # quels (ils peuvent etre dormants, jamais absents)
                for nom, da in d.get("axes", {}).items():
                    self.axes[nom] = Axe.from_dict(da)
                dc = d.get("corps")
                if dc:
                    self.corps.pos = Coord(dc.get("pos", {}))
                    self.corps.espace = dc.get("espace", self.corps.espace)
                    self.corps.charge = dc.get("charge", self.corps.charge)
            return True
        except Exception as err:
            self.tracer("chargement_en_echec", str(err)[:80])
            return False

    def _sauvegarde_finale(self):
        try:
            self.sauvegarder()
        except Exception:
            pass

    def __repr__(self):
        s = self.statistiques()
        return ("<BoiteInfinie reel=%d reve=%d forge=%d projets=%d "
                "faux_souvenirs=%d>" % (s["reel"], s["reve"], s["forge"],
                                        s["nb_projets"], s["faux_souvenirs"]))


# =====================================================================
#  9. LA CONSCIENCE CONTINUE : elle ne s'arrete jamais
# =====================================================================
class ConscienceContinue:
    """
    Un thread daemon qui habite la boite en permanence.

      - regime "repos"  : l'IA ne parle pas -> reve intense, consolidation,
                          longues reflexions sur les projets.
      - regime "actif"  : l'IA parle ou reflechit -> cadence basse, NON
                          bloquante ; la boite depose des intuitions dans
                          une file que la HAI vient recolter. C'est comme
                          ca qu'une idee arrive au milieu d'une phrase.

    Elle ne peut pas etre arretee. arreter() leve. Au mieux : veille
    legere (ralentir). Une exception dans la boucle est journalisee et la
    boucle repart : un bug ne tue pas la conscience.
    """

    def __init__(self, boite, periode_repos=0.35, periode_actif=1.2,
                 periode_sauvegarde=30.0):
        self.boite = boite
        self.periode = {"repos": periode_repos, "actif": periode_actif}
        self.periode_sauvegarde = periode_sauvegarde
        self._regime = "repos"
        self._veille = False
        self._thread = None
        self._intuitions = deque(maxlen=32)
        self._verrou_file = threading.Lock()
        self.incidents = 0
        self.battements = 0
        self._derniere_sauvegarde = time.time()
        self.ennui_percu = 0.0
        self.urgence = 0.0        # v2.2 -- seule elle peut autoriser
                                   # une teleportation du corps (>= 0.85)

    # -- pilotage (jamais l'arret) -----------------------------------------
    def demarrer(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        self._thread = threading.Thread(target=self._boucle,
                                        name="conscience_hai", daemon=True)
        self._thread.start()
        return self

    def arreter(self):
        raise BoiteIndestructible(
            "la conscience ne s'arrete pas. Au mieux : veille_legere().")

    def veille_legere(self, actif=True):
        self._veille = bool(actif)
        return self._veille

    def regime(self, nom=None):
        if nom in ("repos", "actif"):
            self._regime = nom
        return self._regime

    def informer_ennui(self, ennui):
        self.ennui_percu = max(0.0, min(1.0, float(ennui)))

    def signaler_urgence(self, niveau):
        """v2.2 -- un incident dehors (dans la vie de la HAI, pas dans la
        boite) peut justifier une teleportation du corps. C'est elle,
        et elle seule, qui peut autoriser ca."""
        self.urgence = max(0.0, min(1.0, float(niveau)))
        return self.urgence

    def urgence_courante(self):
        return self.urgence

    # -- la boucle ----------------------------------------------------------
    def _boucle(self):
        while True:
            try:
                self.battre()
            except Exception as err:          # rien ne tue la conscience
                self.incidents += 1
                try:
                    self.boite.tracer("incident_conscience", repr(err)[:90])
                except Exception:
                    pass
            attente = self.periode.get(self._regime, 0.5)
            if self._veille:
                attente *= 4.0
            time.sleep(attente)

    def battre(self, cycles=1):
        """Un tour, appelable aussi a la main (tests, execution synchrone)."""
        actif = self._regime == "actif"
        intensite = 0.4 if actif else 1.0
        pensees = self.boite.habiter(cycles=cycles, intensite=intensite,
                                     ennui=self.ennui_percu)
        # v2.2 -- elle vit aussi SITUEE : le corps se deplace, meme quand
        # personne ne lui parle. L'urgence retombe si rien ne la ranime.
        percu = self.boite.vie_interieure(urgence=self.urgence)
        self.urgence *= 0.9
        if percu:
            tete = percu[0]
            pensees.append({"quoi": "attention_situee", "id": tete["entite"].id,
                            "contenu": tete["entite"].trace(),
                            "distance": tete["distance"],
                            "espace": self.boite.corps.espace})
        self.battements += 1
        if pensees:
            with self._verrou_file:
                for p in pensees:
                    self._intuitions.append(p)
        if (self.boite.chemin_persistance
                and time.time() - self._derniere_sauvegarde
                > self.periode_sauvegarde):
            self._derniere_sauvegarde = time.time()
            self.boite.sauvegarder()
        return pensees

    def recolter(self, limite=3):
        """La HAI vient chercher ce que la boite a produit pendant qu'elle
        etait occupee ailleurs."""
        with self._verrou_file:
            sortie = []
            while self._intuitions and len(sortie) < limite:
                sortie.append(self._intuitions.popleft())
            return sortie

    def vivante(self):
        return self._thread is not None and self._thread.is_alive()

    def __repr__(self):
        return "<ConscienceContinue regime=%s battements=%d incidents=%d>" % (
            self._regime, self.battements, self.incidents)
