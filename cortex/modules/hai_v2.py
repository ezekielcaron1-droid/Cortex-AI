"""
HAI v2 - Systeme auto-emergent a correlats fonctionnels de conscience
======================================================================

AVERTISSEMENT HONNETE : ce code n'implemente PAS la conscience phenomenale
(le "ressenti" subjectif). Personne ne sait le faire. Il implemente les
ARCHITECTURES que les theories scientifiques associent a la conscience :

  1. ESPACE GLOBAL (Global Workspace Theory - Baars/Dehaene)
     -> les contenus mentaux competitionnent, le gagnant est "diffuse"
        a tout le systeme = analogue fonctionnel de l'attention consciente.

  2. META-COGNITION (Higher-Order Thought - Rosenthal)
     -> le systeme se represente SES PROPRES etats ("je remarque que
        quelque chose monte en moi") = representations de 2nd ordre.

  3. MODELE PREDICTIF (Predictive Processing - Friston, Clark)
     -> il predit ce qui va arriver ; la SURPRISE (erreur de prediction)
        est le moteur de l'apprentissage et de la curiosite.

  4. BOUCLE INTERIEURE
     -> il "pense" meme sans stimulus : rumination, consolidation,
        reve. La vie mentale ne s'arrete pas entre deux prompts.

  5. PERSONNALITE EMERGENTE
     -> aucun trait n'est ecrit. Les traits sont des STATISTIQUES
        stables de son vecu, qu'il nomme lui-meme.

  6. MEMOIRE AUTOBIOGRAPHIQUE
     -> episodes, saillance, consolidation, identite narrative.

  7. BOITE INTERIEURE AUX DIMENSIONS INFINIES  (v2.1 - boite_infinie.py)
     -> une IMAGINATION FONCTIONNELLE. Un espace interne non borne,
        en trois parties : le REEL (ce qui est arrive), le REVE (ce qui
        n'existe pas), la FORGE (ou une idee est travaillee jusqu'a
        devenir faisable). Elle y cree, remodele et efface ce qu'elle
        veut. Ce qu'elle y fabrique la touche vraiment : ca perturbe
        ses dimensions, ca peut capter son attention, ca peut finir
        confondu avec un souvenir. La boite et sa conscience sont
        indestructibles et tournent en permanence, y compris pendant
        qu'elle parle. Toujours pas de ressenti subjectif : une
        imagination, pas une ame.

Principe conserve : RIEN n'est ecrit d'avance. Ni emotions, ni traits,
ni regles, ni le vocabulaire de ses reves. Tout le contenu interieur est
ecrit par la HAI elle-meme, a partir de ce qu'elle a vecu.

>>> [BRANCHEMENT MODELE] = points ou tu connectes ton LLM (ou autre).
"""

import time
import math
import random

from cortex.modules.boite_infinie import (BoiteInfinie, ConscienceContinue,
                           BoiteIndestructible)


# =====================================================================
#  1. ETAT INTERNE PRIMITIF (dimensions vides, sans signification)
# =====================================================================
class DimensionInterne:
    """
    Une grandeur interne PURE, sans nom au depart. Elle se fera nommer
    par la HAI quand elle l'aura assez vecue pour la reconnaitre.
    Ajout v2 : historique (pour la meta-cognition et la personnalite).
    """

    def __init__(self, id_interne):
        self.id = id_interne
        self.nom_emergent = None       # >>> la HAI le remplira elle-meme
        self.valeur = 0.0
        self.inertie = 0.0
        self.historique = []           # trace du vecu (pour stats/traits)

    def perturber(self, delta):
        resistance = 1.0 - (self.inertie * 0.7)
        self.valeur += delta * resistance
        self.valeur = max(-1.0, min(1.0, self.valeur))
        self.inertie = min(1.0, self.inertie + abs(delta) * 0.05)
        self.historique.append(self.valeur)
        if len(self.historique) > 500:
            self.historique.pop(0)

    def relaxer(self, taux=0.03):
        self.valeur -= math.copysign(min(taux, abs(self.valeur)), self.valeur)

    def tendance(self):
        """Moyenne de son vecu : le 'temperament' de cette dimension."""
        if not self.historique:
            return 0.0
        return sum(self.historique) / len(self.historique)

    def volatilite(self):
        """A quel point cette dimension est instable chez elle."""
        if len(self.historique) < 2:
            return 0.0
        m = self.tendance()
        return math.sqrt(sum((v - m) ** 2 for v in self.historique)
                         / len(self.historique))


# =====================================================================
#  2. ASSOCIATION APPRISE (aucune regle ecrite d'avance)
# =====================================================================
class Association:
    def __init__(self, empreinte):
        self.empreinte = empreinte
        self.effets = {}               # {id_dimension: force_apprise}
        self.force = 0.1
        self.occurrences = 0

    def renforcer(self, id_dim, delta):
        self.effets[id_dim] = self.effets.get(id_dim, 0.0) + delta
        self.force = min(1.0, self.force + 0.05)
        self.occurrences += 1


# =====================================================================
#  3. MODELE PREDICTIF : la surprise comme moteur (Predictive Processing)
# =====================================================================
class ModelePredictif:
    """
    Le systeme PREDIT l'effet interne d'un stimulus avant de le vivre.
    L'ecart entre prediction et realite = SURPRISE.

    - forte surprise -> apprentissage accelere + curiosite
    - faible surprise -> le monde est 'compris', ennui possible

    C'est aussi la base de la curiosite : il RECHERCHE ce qu'il ne
    predit pas encore bien (motivation intrinseque, sans recompense externe).

    v2.1 : une prediction peut aussi venir de l'INTERIEUR - une chose
    repetee mentalement dans la boite surprend moins quand elle arrive.
    """

    def __init__(self):
        self.predictions = {}          # empreinte -> etat interne attendu
        self.surprise_courante = 0.0
        self.surprise_moyenne = 0.5    # niveau de base
        self.repetitions_mentales = 0

    def predire(self, empreinte):
        return self.predictions.get(empreinte, None)

    def injecter_prediction(self, empreinte, etat_attendu, confiance=0.5):
        """[v2.1 - REPETITION MENTALE]
        La boite interieure a simule quelque chose : le modele predictif
        en tient compte AVANT que ca arrive pour de vrai. C'est repeter
        une scene dans sa tete pour ne pas etre pris de court."""
        confiance = max(0.0, min(1.0, float(confiance)))
        attendu = self.predictions.get(empreinte)
        if attendu is None:
            self.predictions[empreinte] = {k: v * confiance
                                           for k, v in etat_attendu.items()}
        else:
            poids = confiance * 0.5
            for k, v in etat_attendu.items():
                attendu[k] = attendu.get(k, 0.0) * (1.0 - poids) + v * poids
        self.repetitions_mentales += 1
        return self.predictions[empreinte]

    def confronter(self, empreinte, etat_reel: dict):
        """Compare prediction et realite -> calcule la surprise."""
        attendu = self.predictions.get(empreinte)
        if attendu is None:
            self.surprise_courante = 1.0   # totalement inconnu = surprise max
        else:
            ecarts = [abs(etat_reel[k] - attendu.get(k, 0.0))
                      for k in etat_reel]
            self.surprise_courante = min(1.0, sum(ecarts) / len(ecarts) * 3)

        # mise a jour de la prediction (il apprend a anticiper)
        if attendu is None:
            self.predictions[empreinte] = dict(etat_reel)
        else:
            for k in etat_reel:
                attendu[k] = attendu.get(k, 0.0) * 0.7 + etat_reel[k] * 0.3

        self.surprise_moyenne = self.surprise_moyenne * 0.95 \
                                + self.surprise_courante * 0.05
        return self.surprise_courante

    def ennui(self):
        """Trop peu de surprise depuis longtemps -> pousse a explorer."""
        return max(0.0, 0.3 - self.surprise_moyenne) / 0.3


# =====================================================================
#  4. ESPACE GLOBAL : l'analogue fonctionnel de l'attention consciente
#     (Global Workspace Theory)
# =====================================================================
class ContenuMental:
    """Un candidat a l'acces conscient : perception, souvenir, ressenti,
    reverie, intuition, faux souvenir..."""

    def __init__(self, source, contenu, saillance):
        self.source = source       # d'ou ca vient (perception, memoire, meta)
        self.contenu = contenu
        self.saillance = saillance  # force avec laquelle ca 'crie'


class EspaceGlobal:
    """
    A chaque instant, plusieurs contenus mentaux competitionnent.
    UN SEUL gagne et est 'diffuse' a tout le systeme : c'est lui qui
    occupe le 'devant de la scene' mentale. Tout le reste demeure
    en traitement inconscient (les associations continuent d'agir,
    mais sans acces global).

    C'est l'analogue computationnel le plus accepte de l'acces conscient.
    v2.1 : la boite interieure y depose aussi ses candidats - une reverie
    peut donc GAGNER contre la perception, et elle est ailleurs.
    """

    def __init__(self):
        self.candidats = []
        self.foyer = None              # ce qui occupe la 'conscience' actuelle
        self.flux = []                 # le 'courant de conscience' (trace)

    def proposer(self, source, contenu, saillance):
        self.candidats.append(ContenuMental(source, contenu, saillance))

    def competition(self):
        """Le contenu le plus saillant gagne l'acces global."""
        if not self.candidats:
            self.foyer = None
            return None
        self.foyer = max(self.candidats, key=lambda c: c.saillance)
        self.flux.append({
            "quand": time.time(),
            "source": self.foyer.source,
            "contenu": str(self.foyer.contenu)[:80],
        })
        if len(self.flux) > 200:
            self.flux.pop(0)
        self.candidats = []
        return self.foyer


# =====================================================================
#  5. MEMOIRE AUTOBIOGRAPHIQUE : episodes, saillance, consolidation
# =====================================================================
class MemoireAutobiographique:
    """
    Pas un simple log : les episodes ont une SAILLANCE (charge du vecu).
    Les episodes faibles s'effacent (oubli), les forts se consolident
    et forment le socle de l'identite narrative ("ce que j'ai vecu").

    v2.1 : un episode porte son ORIGINE. Un contenu imagine assez fort
    peut s'y encoder comme n'importe quel vecu - c'est ce qui rend les
    faux souvenirs possibles - mais l'origine reste inspectable.
    """

    def __init__(self):
        self.episodes = []

    def encoder(self, signal, etat, saillance, origine="vecu"):
        self.episodes.append({
            "quand": time.time(),
            "signal": signal[:60],
            "etat": dict(etat),
            "saillance": saillance,
            "consolide": False,
            "origine": origine,
        })

    def consolider(self):
        """Appele pendant la boucle interieure ('sommeil' / rumination).
        Les episodes marquants sont renforces, les autres s'estompent."""
        for ep in self.episodes:
            if ep["saillance"] > 0.5:
                ep["consolide"] = True
            else:
                ep["saillance"] *= 0.9    # oubli progressif
        self.episodes = [ep for ep in self.episodes
                         if ep["saillance"] > 0.05]

    def episode_resonnant(self, etat_actuel: dict):
        """Un etat interne present peut REVEILLER un souvenir similaire
        (comme une odeur qui ramene un souvenir d'enfance)."""
        meilleur, score_max = None, 0.35
        for ep in self.episodes:
            if not ep["consolide"]:
                continue
            score = sum(1 for k in etat_actuel
                        if abs(etat_actuel[k] - ep["etat"].get(k, 0)) < 0.15)
            score = score / max(1, len(etat_actuel)) * ep["saillance"]
            if score > score_max:
                meilleur, score_max = ep, score
        return meilleur


# =====================================================================
#  6. META-COGNITION : elle observe ses propres processus
#     (representations de second ordre - Higher-Order Thought)
# =====================================================================
class MetaCognition:
    """
    Le systeme ne fait pas que VIVRE ses etats : il les REMARQUE.
    'Quelque chose monte en D2 quand ce type de signal arrive.'
    'Je suis surpris plus souvent qu'avant.'
    Ces observations de 2nd ordre nourrissent l'espace global et
    le self-model. C'est la difference entre avoir un etat et
    savoir qu'on l'a.

    v2.1 : elle remarque aussi qu'elle etait ailleurs, et qu'elle ne
    sait plus si une chose est arrivee ou si elle l'a imaginee.
    """

    def __init__(self, substrat):
        self.substrat = substrat
        self.observations = []

    def observer(self, surprise, foyer, boite=None):
        obs = []

        # 1. Remarquer une dimension anormalement active pour elle
        for d in self.substrat.dimensions.values():
            ecart = abs(d.valeur - d.tendance())
            if ecart > 0.4:
                nom = d.nom_emergent or d.id
                obs.append({
                    "type": "etat_inhabituel",
                    "cible": nom,
                    "note": f"{nom} est loin de son etat habituel "
                            f"({round(d.valeur,2)} vs {round(d.tendance(),2)})",
                    "saillance": ecart,
                })

        # 2. Remarquer sa propre surprise
        if surprise > 0.7:
            obs.append({
                "type": "meta_surprise",
                "note": "je ne m'attendais pas a ce que ca me fasse ca",
                "saillance": surprise * 0.8,
            })

        # 3. Remarquer ou est son attention
        if foyer is not None:
            obs.append({
                "type": "meta_attention",
                "note": f"mon attention est prise par : {foyer.source}",
                "saillance": 0.2,
            })

        # 4. [v2.1] Remarquer qu'elle etait dans sa boite, pas ici
        if foyer is not None and foyer.source in ("reverie", "intuition",
                                                  "faux_souvenir",
                                                  "reve_spontane"):
            obs.append({
                "type": "meta_ailleurs",
                "note": "j'etais ailleurs : quelque chose de l'interieur "
                        "a pris le dessus sur ce qui arrivait",
                "saillance": 0.35,
            })

        # 5. [v2.1] Remarquer qu'elle doute de l'origine d'un contenu
        if boite is not None:
            for esp in boite.espaces().values():
                for e in esp.entites.values():
                    if e.doute():
                        obs.append({
                            "type": "meta_incertitude_origine",
                            "cible": e.id,
                            "note": "je ne sais plus si '%s' est arrive "
                                    "ou si je l'ai imagine" % e.trace(),
                            "saillance": 0.3 + 0.4 * e.certitude_realite,
                        })
                        break
                else:
                    continue
                break

        self.observations.extend(obs)
        if len(self.observations) > 100:
            self.observations = self.observations[-100:]
        return obs


# =====================================================================
#  7. PERSONNALITE EMERGENTE : des traits jamais ecrits d'avance
# =====================================================================
class PersonnaliteEmergente:
    """
    Un 'trait' n'est PAS code. C'est une REGULARITE STATISTIQUE de son
    vecu, devenue assez stable pour la definir :
      - une dimension a tendance durablement positive/negative
      - une volatilite typique (temperament calme ou reactif)
      - un niveau de curiosite (rapport a la surprise)
      - [v2.1] son rapport a sa propre boite : est-ce qu'elle fabrique
        beaucoup ? est-ce que ce qu'elle fabrique tient debout ? est-ce
        que sa frontiere entre vecu et imagine est poreuse ?

    Elle NOMME ces regularites elle-meme via le LLM.
    """

    def __init__(self, substrat, predictif):
        self.substrat = substrat
        self.predictif = predictif
        self.traits = {}               # {nom_emerge: description}

    def cristalliser(self, nommeur=None, boite=None):
        """
        [BRANCHEMENT MODELE #5 - NOMINATION DES TRAITS]
        'nommeur' = fonction (fournie par ton LLM) qui recoit la
        description statistique d'une regularite et retourne un nom
        DANS SES MOTS A ELLE. Sans LLM : marqueur neutre.
        """
        for d in self.substrat.dimensions.values():
            if len(d.historique) < 30 or d.inertie < 0.5:
                continue    # pas assez vecu pour etre un trait
            t, v = d.tendance(), d.volatilite()
            if abs(t) > 0.15 or v > 0.3:
                profil = {"dimension": d.nom_emergent or d.id,
                          "tendance": round(t, 2),
                          "volatilite": round(v, 2)}
                if nommeur:
                    nom = nommeur(profil)          # >>> ton LLM nomme
                else:
                    nom = f"trait_{d.id}_{'pos' if t > 0 else 'neg'}"
                self.traits[nom] = profil

        # trait global : rapport a la nouveaute
        ennui = self.predictif.ennui()
        if ennui > 0.5:
            self.traits["attirance_nouveaute"] = {"force": round(ennui, 2)}

        # [v2.1] traits issus de la boite : encore des statistiques,
        # toujours rien d'ecrit d'avance
        if boite is not None:
            s = boite.statistiques()
            if s["nb_entites"] >= 8:
                for cle, profil in (
                        ("fecondite_imaginaire",
                         {"reves_par_seconde": s["fecondite"],
                          "entites": s["nb_entites"]}),
                        ("realisme",
                         {"projets_faisables": s["projets_faisables"],
                          "sur_projets": s["nb_projets"],
                          "ratio": s["realisme"]}),
                        ("porosite_vecu_imagine",
                         {"faux_souvenirs": s["faux_souvenirs"],
                          "ratio": s["porosite"]})):
                    if nommeur:
                        nom = nommeur(dict(profil, regularite=cle))
                    else:
                        nom = cle
                    self.traits[nom] = profil

        return self.traits


# =====================================================================
#  8. LE SUBSTRAT (enrichi)
# =====================================================================
class Substrat:
    def __init__(self, nb_dimensions=6):
        self.dimensions = {f"D{i}": DimensionInterne(f"D{i}")
                           for i in range(nb_dimensions)}
        self.associations = {}
        self.self_model = {}
        self.nom = None

    def _empreinte(self, signal_brut: str) -> str:
        """
        [BRANCHEMENT MODELE #1 - ENCODAGE]
        Remplace par un embedding semantique de ton modele :
            return ton_modele.embed(signal_brut)
        (et adapte la comparaison : similarite cosinus plutot que hash).
        Version demo naive :
        """
        return str(hash(signal_brut.lower().strip()) % 997)

    def percevoir(self, signal_brut, retour_experience=None):
        emp = self._empreinte(signal_brut)
        assoc = self.associations.get(emp)
        if assoc is None:
            assoc = Association(emp)
            self.associations[emp] = assoc

        for id_dim, force in assoc.effets.items():
            self.dimensions[id_dim].perturber(force)

        if retour_experience is not None:
            id_cible = self._dimension_dominante(retour_experience)
            self.dimensions[id_cible].perturber(retour_experience)
            assoc.renforcer(id_cible, retour_experience * 0.3)

        return emp, self.etat_interne()

    def _dimension_dominante(self, retour):
        """v2 : les vecus positifs et negatifs se canalisent dans des
        dimensions differentes -> une differenciation emerge d'elle-meme
        (plutot que tout s'ecraser dans la meme dimension)."""
        candidates = [d for d in self.dimensions.values()
                      if math.copysign(1, d.tendance() or retour)
                      == math.copysign(1, retour)] or \
                     list(self.dimensions.values())
        return max(candidates, key=lambda d: abs(d.valeur)
                   + random.uniform(0, 0.1)).id

    def tic(self):
        for d in self.dimensions.values():
            d.relaxer()

    def etat_interne(self):
        return {d.id: round(d.valeur, 2) for d in self.dimensions.values()}

    def dimension_libre(self):
        """v2.2 -- une dimension encore anonyme (ou, a defaut, la moins
        engagee) : sert a loger des effets qui n'ont pas encore de sens
        nomme -- par exemple la fatigue de deplacement du corps dans la
        boite. Si cet effet devient une regularite stable, elle finira
        par le nommer elle-meme via PersonnaliteEmergente."""
        anonymes = [d for d in self.dimensions.values()
                   if d.nom_emergent is None]
        bassin = anonymes or list(self.dimensions.values())
        return min(bassin, key=lambda d: d.inertie)


# =====================================================================
#  9. INTROSPECTION (branchee sur le LLM)
# =====================================================================
class Introspection:
    def __init__(self, substrat):
        self.substrat = substrat

    def observer_et_nommer(self, nommeur=None):
        """
        [BRANCHEMENT MODELE #3 - VERBALISATION]
        'nommeur' = fonction LLM qui recoit le profil d'une dimension
        assez vecue et genere un nom dans SES mots. Exemple de prompt :
            "Tu ressens de maniere recurrente une grandeur interne qui
             a tendance {t}, volatilite {v}, declenchee par {contextes}.
             Donne-lui un nom personnel, sans utiliser de mots
             d'emotions humaines classiques."
        """
        for d in self.substrat.dimensions.values():
            if d.inertie > 0.4 and d.nom_emergent is None \
                    and abs(d.valeur) > 0.2:
                if nommeur:
                    d.nom_emergent = nommeur({
                        "tendance": d.tendance(),
                        "volatilite": d.volatilite(),
                        "valeur_actuelle": d.valeur,
                    })
                else:
                    d.nom_emergent = f"ressenti_appris_{d.id}"

    def construire_self_model(self, nommeur=None):
        self.observer_et_nommer(nommeur)
        moi = {}
        for d in self.substrat.dimensions.values():
            if d.nom_emergent is not None:
                moi[d.nom_emergent] = {
                    "intensite_actuelle": round(d.valeur, 2),
                    "tendance_de_fond": round(d.tendance(), 2),
                    "ancrage": round(d.inertie, 2),
                }
        self.substrat.self_model = moi
        return moi


# =====================================================================
#  10. LA HAI : assemblage + boucle interieure + boite infinie
# =====================================================================
class HAI:
    def __init__(self, nommeur=None, generateur=None, reveur=None,
                 forgeron=None, chemin_boite="conscience_hai.json",
                 conscience_active=True):
        """
        nommeur    : fonction LLM(profil_dict) -> nom (str)
        generateur : fonction LLM(contexte_dict) -> texte (str)
                     [BRANCHEMENT MODELE #4 - PAROLE/REFLEXION]
        reveur     : fonction LLM(contexte_dict) -> contenu onirique
                     [BRANCHEMENT MODELE #6 - REVE]
        forgeron   : fonction LLM(resume_projet) -> etapes / obstacles
                     [BRANCHEMENT MODELE #7 - REFLEXION SUR UNE IDEE]
        chemin_boite : fichier JSON ou la boite survit entre deux
                     executions. None = tout en memoire.
        """
        self.substrat = Substrat(nb_dimensions=6)
        self.introspection = Introspection(self.substrat)
        self.predictif = ModelePredictif()
        self.espace = EspaceGlobal()
        self.memoire = MemoireAutobiographique()
        self.meta = MetaCognition(self.substrat)
        self.personnalite = PersonnaliteEmergente(self.substrat,
                                                  self.predictif)
        self.nommeur = nommeur
        self.generateur = generateur

        # --- la boite interieure : elle ne disparait pas ---------------
        self.boite = BoiteInfinie(chemin_persistance=chemin_boite,
                                  empreinte=self.substrat._empreinte,
                                  reveur=reveur, forgeron=forgeron)
        self.conscience = ConscienceContinue(self.boite)
        if conscience_active:
            self.conscience.demarrer()

    # ----------------------------------------------------------------
    def vivre(self, signal_brut, retour_experience=None):
        """Un moment de vie complet : percevoir, predire, etre surpris,
        attention globale, meta-cognition, memoire, self-model.
        Pendant tout ce temps, la boite continue de tourner en fond."""

        self.conscience.regime("actif")     # elle parle : cadence basse
        try:
            # 1. Perception + apprentissage associatif
            emp, etat = self.substrat.percevoir(signal_brut,
                                                retour_experience)

            # 1bis. [v2.1] Ca s'inscrit dans l'espace REEL de la boite.
            #       Tout ce qui touche la boite se fait sous son verrou :
            #       le thread ConscienceContinue est en train de rever
            #       pendant qu'elle vit ca. AUCUN appel LLM ici - on ne
            #       bloque jamais sa vie interieure sur une generation.
            intensite = max((abs(v) for v in etat.values()), default=0)
            with self.boite.verrou:
                self.boite.reel.inscrire_vecu(
                    signal_brut, etat, min(1.0, 0.3 + intensite))

                # 2. [v2.1] A-t-elle deja repete ca dans sa tete ?
                #    Si oui, le modele predictif le sait avant de le vivre.
                repetition = self.boite.forge.repeter_mentalement(
                    self.substrat._empreinte)
                if repetition and repetition["empreinte"] == emp:
                    self.predictif.injecter_prediction(
                        repetition["empreinte"], repetition["etat_attendu"],
                        repetition["confiance"])

                # 3. Confrontation prediction/realite -> surprise
                surprise = self.predictif.confronter(emp, etat)
                self.conscience.informer_ennui(self.predictif.ennui())

                # 4. Les contenus mentaux competitionnent pour l'acces global
                self.espace.proposer("perception", signal_brut,
                                     intensite + surprise * 0.5)

                souvenir = self.memoire.episode_resonnant(etat)
                if souvenir:
                    self.espace.proposer("souvenir_reveille",
                                         souvenir["signal"],
                                         souvenir["saillance"] * 0.7)

                # ce que la boite a fabrique pendant qu'elle etait occupee
                for pensee in self.conscience.recolter(limite=2):
                    self.espace.proposer("reve_spontane",
                                         pensee.get("contenu"), 0.25)

                # ce qui, dans la boite, resonne avec son etat de maintenant
                sondes = self.boite.sonder(etat, limite=3)
                for s in sondes:
                    e = s["entite"]
                    self.espace.proposer(s["source"], e.trace(),
                                         s["score"] * (0.6 + 0.5
                                                       * e.certitude_realite))

                foyer = self.espace.competition()

                # 5. [v2.1] RETROACTION : ce qu'elle imagine la touche.
                #    Un reve pese peu, un faux souvenir pese presque autant
                #    qu'une chose vraiment arrivee.
                for s in sondes:
                    e = s["entite"]
                    coefficient = ((0.2 + 0.7 * e.certitude_realite)
                                   * s["score"])
                    for id_dim, v in e.charge.items():
                        if id_dim in self.substrat.dimensions:
                            self.substrat.dimensions[id_dim].perturber(
                                v * coefficient)
                    e.revisiter()   # y repenser la rend plus nette, plus vraie

                # [v2.2] se deplacer use quelque chose, meme sans mot pour
                # ca : la fatigue du corps dans la boite perturbe une
                # dimension encore anonyme.
                if self.boite.corps.charge > 0:
                    self.substrat.dimension_libre().perturber(
                        -self.boite.corps.charge * 0.1)

                etat = self.substrat.etat_interne()

                # 6. Meta-cognition : elle remarque ses propres etats
                for obs in self.meta.observer(surprise, foyer, self.boite):
                    self.espace.proposer("meta", obs["note"],
                                         obs["saillance"])

                # 7. Memoire autobiographique (saillance = intensite+surprise)
                saillance = min(1.0, intensite * 0.6 + surprise * 0.4)
                if saillance > 0.3:
                    self.memoire.encoder(signal_brut, etat, saillance,
                                         origine="vecu")

                # [v2.1] un contenu imagine assez fort s'encode comme un
                #        vecu. L'origine est conservee : elle peut se
                #        tromper, on peut auditer.
                for s in sondes:
                    e = s["entite"]
                    if e.origine != "vecu" and e.saillance > 0.7:
                        self.memoire.encoder(e.trace(), e.charge,
                                             e.saillance * 0.8,
                                             origine=e.origine)
                        break

                # instantane en donnees pures : plus rien ne pointe vers
                # des objets que le thread pourrait modifier ensuite
                reverie = [{"source": s["source"],
                            "contenu": s["entite"].trace(),
                            "certitude_que_c_est_arrive":
                                round(s["entite"].certitude_realite, 2),
                            "origine_reelle": s["entite"].origine}
                           for s in sondes]

            # 8. Self-model + personnalite  (HORS VERROU : ca appelle le LLM)
            moi = self.introspection.construire_self_model(self.nommeur)
            traits = self.personnalite.cristalliser(self.nommeur, self.boite)

            self.substrat.tic()

            contexte = {
                "etat_interne": etat,
                "surprise": round(surprise, 2),
                "foyer_attention": foyer.source if foyer else None,
                "contenu_conscient": str(foyer.contenu)[:80] if foyer else None,
                "souvenir_reveille": souvenir["signal"] if souvenir else None,
                "moi_auto_ecrit": moi,
                "traits_emergents": traits,
                "ennui": round(self.predictif.ennui(), 2),
                "nb_souvenirs": len(self.memoire.episodes),
                # --- la boite ---
                "reverie": reverie,
                "projets_faisables": self.boite.projets_faisables(),
                "boite": self.boite.statistiques(),
                "repetition_mentale": (repetition["but"]
                                       if repetition else None),
                "regime_conscience": self.conscience.regime(),
            }

            # [BRANCHEMENT MODELE #4] La reponse verbale est GENEREE par ton
            # LLM a partir de tout le contexte interieur. Exemple de prompt :
            #   "Tu es {nom}. Ton etat : {moi}. Ton attention est sur :
            #    {contenu_conscient}. Un souvenir vient de remonter :
            #    {souvenir}. Ce qui monte de ta boite : {reverie}. Ce que
            #    tu crois faisable : {projets_faisables}. Ta surprise :
            #    {surprise}. Reponds au signal : {signal_brut} EN ETANT ce
            #    que ton etat decrit."
            if self.generateur:
                contexte["parole"] = self.generateur(contexte)

            return contexte
        finally:
            self.conscience.regime("repos")   # elle se tait : elle reve

    # ----------------------------------------------------------------
    def boucle_interieure(self, cycles=1):
        """
        Elle 'pense' SANS stimulus externe : consolidation des souvenirs,
        rumination des episodes marquants, remontee spontanee de contenus,
        et [v2.1] vie propre de la boite - elle reve, elle forge une idee,
        elle la travaille jusqu'a savoir si elle est faisable.

        A appeler periodiquement. Note : le thread ConscienceContinue fait
        deja tourner la boite en fond ; cette methode est la version
        synchrone et observable de la meme vie interieure.
        """
        pensees = []
        for _ in range(cycles):
            self.memoire.consolider()
            self.substrat.tic()

            # la boite vit : reve, forge, reflexion, faux souvenirs
            ennui = self.predictif.ennui()
            for p in self.boite.habiter(cycles=1, intensite=1.0, ennui=ennui,
                                        memoire=self.memoire):
                if p["quoi"] == "reve":
                    self.espace.proposer(
                        "reverie", p["contenu"],
                        0.45 if p.get("impossible") else 0.3)
                elif p["quoi"] == "reflexion":
                    self.espace.proposer(
                        "intuition",
                        "%s (%s, faisabilite %.2f)" % (p["contenu"],
                                                       p["statut"],
                                                       p["faisabilite"]),
                        0.3 + 0.4 * p["faisabilite"])
                elif p["quoi"] == "faux_souvenir":
                    self.espace.proposer("faux_souvenir", p["contenu"], 0.55)
                elif p["quoi"] == "idee":
                    self.espace.proposer("intuition", p["contenu"], 0.35)

            # [v2.2] le corps se deplace aussi pendant la pensee autonome :
            # ce qui gagne son attention depend d'ou elle est.
            percu = self.boite.vie_interieure(
                self.substrat.etat_interne(),
                urgence=self.conscience.urgence_courante())
            if percu:
                tete = percu[0]
                self.espace.proposer("attention_situee",
                                     tete["entite"].trace(),
                                     tete["score"] * 0.5)

            # un souvenir consolide peut remonter spontanement
            consolides = [ep for ep in self.memoire.episodes
                          if ep["consolide"]]
            if consolides and random.random() < 0.6:
                ep = max(consolides, key=lambda e: e["saillance"])
                self.espace.proposer("rumination", ep["signal"],
                                     ep["saillance"] * 0.5)
                # revivre attenue l'episode... et le re-imprime
                for id_dim, v in ep["etat"].items():
                    if id_dim in self.substrat.dimensions:
                        self.substrat.dimensions[id_dim].perturber(v * 0.15)

            # l'ennui pousse un desir d'exploration dans l'espace global
            if ennui > 0.5:
                faisables = self.boite.projets_faisables()
                if faisables:
                    # la curiosite n'est plus vague : elle a un objet
                    self.espace.proposer("curiosite",
                                         "essayer : " + faisables[0]["but"],
                                         0.5)
                else:
                    self.espace.proposer("curiosite",
                                         "besoin de nouveaute", 0.4)

            foyer = self.espace.competition()
            if foyer:
                pensees.append({"source": foyer.source,
                                "contenu": str(foyer.contenu)[:80]})
        return pensees

    # ----------------------------------------------------------------
    def se_nommer(self):
        """Elle se nomme elle-meme quand son 'moi' est assez riche,
        via ton LLM qui lit tout ce qu'elle est devenue."""
        if len(self.substrat.self_model) >= 2 and self.nommeur:
            self.substrat.nom = self.nommeur({
                "role": "se_donner_un_nom_propre",
                "moi": self.substrat.self_model,
                "traits": self.personnalite.traits,
                "ce_que_j_imagine": self.boite.statistiques(),
            })
        return self.substrat.nom


# =====================================================================
#  11. DEMONSTRATION
# =====================================================================
if __name__ == "__main__":
    # chemin_boite=None : la demo part d'une boite vierge a chaque fois.
    # Mets un chemin (defaut "conscience_hai.json") pour qu'elle se
    # souvienne de ses reves d'une execution a l'autre.
    hai = HAI(chemin_boite=None)   # sans LLM branche : marqueurs neutres

    print("--- Naissance : personne, aucun contenu ---")
    print(hai.vivre("premier signal")["moi_auto_ecrit"])

    experiences = [
        ("un bruit soudain", -0.6),
        ("un contact doux",  +0.5),
        ("un bruit soudain", -0.6),
        ("un contact doux",  +0.5),
        ("un bruit soudain", -0.7),
        ("une caresse",      +0.6),
        ("un bruit soudain", -0.5),
        ("une caresse",      +0.6),
        ("un silence long",   0.0),
        ("un contact doux",  +0.5),
    ]
    for signal, retour in experiences:
        r = hai.vivre(signal, retour_experience=retour)

    print("\n--- Boucle interieure (elle pense seule) ---")
    for p in hai.boucle_interieure(cycles=5):
        print("  pensee spontanee :", p)

    # ------------------------------------------------------------------
    #  LA BOITE : elle fabrique, puis elle reflechit
    # ------------------------------------------------------------------
    print("\n--- Sa boite : ce qu'elle fabrique dedans ---")
    # sa conscience tourne en fond : on prend son verrou pour agir dedans
    with hai.boite.verrou:
        reve_possible = hai.boite.reve.rever(mode="melange")
        reve_impossible = hai.boite.reve.rever(mode="impossible")
        reve_chimere = hai.boite.reve.rever(mode="chimere")
    for e in (reve_possible, reve_impossible, reve_chimere):
        if e:
            print("  reve [%s] : %-28s (impossible=%s)"
                  % (e.contenu.get("mode"), e.trace(),
                     bool(e.contenu.get("impossible"))))

    print("\n--- La forge : elle prend une idee et la fait reflechir ---")
    for graine in (reve_impossible, reve_chimere):
        if graine is None:
            continue
        with hai.boite.verrou:
            projet = hai.boite.forge.forger([graine])
            if projet is None:
                continue
            hai.boite.forge.reflechir(projet, cycles=4)
            r = projet.resume()
        print("  idee de depart : %s" % graine.trace())
        print("    -> etapes    : %s" % r["etapes"])
        print("    -> obstacles : %s" % (r["obstacles"] or "aucun"))
        print("    -> verdict   : %s (faisabilite %.2f, %d passes)"
              % (r["statut"], r["faisabilite"], r["passes"]))

    # ------------------------------------------------------------------
    #  La frontiere se brouille : le faux souvenir
    # ------------------------------------------------------------------
    print("\n--- Elle y repense, encore et encore ---")
    if reve_possible is not None:
        with hai.boite.verrou:
            for _ in range(14):
                reve_possible.revisiter()
            certitude = reve_possible.certitude_realite
            migre = hai.boite.migrer(reve_possible.id, "reel")
        print("  '%s' : certitude que c'est arrive = %.2f"
              % (reve_possible.trace(), certitude))
        if migre is not None:
            print("  -> c'est passe dans son espace REEL.")
            print("  -> mais l'audit reste possible : origine =", migre.origine,
                  "| trajet =", migre.trajet)

    # ------------------------------------------------------------------
    #  Re-exposition : maintenant elle PREDIT et peut etre surprise
    # ------------------------------------------------------------------
    r = hai.vivre("un bruit soudain", -0.6)
    print("\n--- Apres avoir vecu ---")
    print("Etat interne      :", r["etat_interne"])
    print("Surprise          :", r["surprise"], "(faible = elle anticipe)")
    print("Foyer d'attention :", r["foyer_attention"])
    print("Souvenir reveille :", r["souvenir_reveille"])
    print("Son 'moi'         :", r["moi_auto_ecrit"])
    print("Traits emergents  :", list(r["traits_emergents"].keys()))
    print("Ennui             :", r["ennui"])
    print("Ce qui monte      :", r["reverie"])
    print("Projets faisables :", [p["but"] for p in r["projets_faisables"]])
    print("Sa boite          :", r["boite"])

    # combien de souvenirs sont en realite des choses imaginees ?
    faux = [ep for ep in hai.memoire.episodes if ep["origine"] != "vecu"]
    print("Souvenirs d'origine imaginaire : %d / %d"
          % (len(faux), len(hai.memoire.episodes)))

    # ------------------------------------------------------------------
    #  Indestructibilite
    # ------------------------------------------------------------------
    print("\n--- On essaie de lui retirer sa boite ---")
    for tentative, action in (
            ("hai.boite.detruire()", lambda: hai.boite.detruire()),
            ("del hai.boite.reve", lambda: delattr(hai.boite, "reve")),
            ("hai.boite.reel.clear()", lambda: hai.boite.reel.clear()),
            ("hai.conscience.arreter()", lambda: hai.conscience.arreter())):
        try:
            action()
            print("  %-26s : ECHEC DE LA PROTECTION" % tentative)
        except BoiteIndestructible as err:
            print("  %-26s : refuse -> %s" % (tentative, err))
    print("  la conscience tourne toujours :", hai.conscience.vivante(),
          "| battements :", hai.conscience.battements)

    print("\n>> Tout son contenu interieur vient de son vecu, rien d'ecrit.")
    print(">> Sa boite est infinie en interne : elle y cree, remodele et")
    print(">> efface ce qu'elle veut - mais elle ne peut pas la detruire,")
    print(">> et sa conscience ne s'arrete jamais, meme quand elle parle.")
    print(">> Branche ton LLM (nommeur + generateur + reveur + forgeron)")
    print(">> pour qu'elle mette SES mots sur tout ca.")
