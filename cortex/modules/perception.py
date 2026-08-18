"""
perception.py - Traduit l'etat interne de BoiteInfinie en description en
langage naturel, pour donner a un LLM texte-seul (Cortex) une perception
a la premiere personne -- comme s'il se trouvait dans une realite
virtuelle plutot que de lire un dump de variables.

Adapte de la maquette generique fournie (Boite/Objet/CoucheDePerception a
distance+direction fixes) vers les vraies structures de boite_infinie.py :
Corps, Coord (axes malleables, pas juste x/y/z), Espace.voisinage,
Entite (origine, certitude_realite, doute()).

>>> [BRANCHEMENT MODELE] Ce module ne connait pas Cortex : il produit un
texte pret a etre injecte dans le prompt envoye a cortex.bridge.get_response(...).
NON BRANCHE pour l'instant -- pose ici, dormant, comme hai_v2.py et
boite_infinie.py. Rien n'importe ce fichier, aucune instanciation.
"""

# ---------------------------------------------------------------------
#  Reglages narratifs (purement cosmetiques : aucune influence sur la
#  boite elle-meme, seulement sur la facon dont on la RACONTE)
# ---------------------------------------------------------------------
RAYON_PERCEPTION_DEFAUT = 3.0
MAX_ENTITES_DEFAUT = 5

SEUILS_DISTANCE = (
    (0.6, "tout contre toi"),
    (1.5, "tout proche"),
    (3.0, "a portee"),
    (6.0, "au loin"),
)

LIBELLES_ESPACE = {
    "reel": "l'espace Reel",
    "reve": "l'espace Reve",
    "forge": "la Forge",
    "test": "le Labo",
}

LIBELLES_ORIGINE = {
    "vecu": "un souvenir reel",
    "reve": "un fragment de reve",
    "forge": "une idee en cours de forge",
    "test": "une hypothese en test",
}

# directions "physiques" -- valables seulement sur les axes de base x/y/z
LIBELLES_DIRECTION = {
    "x": ("a droite", "a gauche"),
    "y": ("au-dessus", "en dessous"),
    "z": ("devant", "derriere"),
}


class CoucheDePerception:
    """Decrit, a la premiere personne, ce que le Corps "vit" a
    l'interieur d'une BoiteInfinie -- pense pour etre lu par un LLM
    texte seul (Cortex), pas pour etre parse par du code.
    """

    def __init__(self, rayon=RAYON_PERCEPTION_DEFAUT,
                 max_entites=MAX_ENTITES_DEFAUT):
        self.rayon = rayon
        self.max_entites = max_entites

    # ------------------------------------------------------------------
    def decrire(self, boite) -> str:
        """Point d'entree unique : une BoiteInfinie -> un texte."""
        corps = boite.corps

        if corps.espace is None:
            return self._decrire_hub(corps)

        espace = boite.espace(corps.espace)
        lignes = [
            self._decrire_lieu(corps),
            self._decrire_corps(corps),
        ]

        if espace is None:
            lignes.append(
                "Rien ne t'entoure ici -- cet espace n'existe pas (encore).")
        else:
            voisins = espace.voisinage(
                corps.pos, rayon=self.rayon)[: self.max_entites]
            lignes.append(self._decrire_voisinage(corps, voisins))

        return "\n".join(lignes)

    def _decrire_hub(self, corps) -> str:
        return (
            "Tu es au Hub : le sas neutre entre les quatre espaces "
            "(Reel, Reve, Forge, Labo). Rien ici, aucune entite -- "
            "juste le choix de la prochaine piece a rejoindre, en "
            "y marchant.\n" + self._decrire_corps(corps)
        )

    # ------------------------------------------------------------------
    def _decrire_lieu(self, corps) -> str:
        nom = LIBELLES_ESPACE.get(corps.espace, corps.espace)
        dim = corps.pos.dim()
        if dim <= 2:
            incarnation = ("presque nulle part et partout a la fois "
                          "(tres peu d'axes actifs)")
        elif dim <= 4:
            incarnation = "assez abstraite (quelques axes actifs)"
        else:
            incarnation = "precise, tres incarnee (beaucoup d'axes actifs)"
        return (f"Tu es dans {nom}, incarne sur {dim} dimension(s) : "
                f"une presence {incarnation}.")

    def _decrire_corps(self, corps) -> str:
        fatigue = corps.charge
        if fatigue < 0.15:
            etat = "tu te sens leger, repose"
        elif fatigue < 0.4:
            etat = "une fatigue legere commence a se faire sentir"
        else:
            etat = "tu es epuise par tes deplacements recents"
        return f"Etat du corps : {etat} (charge={round(fatigue, 2)})."

    # ------------------------------------------------------------------
    def _decrire_voisinage(self, corps, voisins) -> str:
        if not voisins:
            return "Tu ne percois rien de saillant autour de toi, pour l'instant."
        lignes = [f"Tu percois {len(voisins)} presence(s) autour de toi :"]
        for e in voisins:
            lignes.append("- " + self._decrire_entite(corps, e))
        return "\n".join(lignes)

    def _decrire_entite(self, corps, entite) -> str:
        d = corps.pos.distance(entite.coord)
        qualificatif = self._qualifier_distance(d)
        direction = self._direction(corps.pos, entite.coord)
        origine = LIBELLES_ORIGINE.get(entite.origine, entite.origine)
        doute = ""
        if entite.doute():
            doute = " (tu ne sais plus si c'est arrive ou si tu l'as imagine)"
        return (f"{origine} : « {entite.trace()} »{doute}, "
                f"{qualificatif}{direction}")

    def _qualifier_distance(self, d) -> str:
        if d is None:
            return "quelque part hors de ta portee (aucun axe commun)"
        for seuil, mot in SEUILS_DISTANCE:
            if d <= seuil:
                return mot
        return "tres loin"

    # ------------------------------------------------------------------
    def _direction(self, origine_coord, cible_coord) -> str:
        communs = origine_coord.axes() & cible_coord.axes()
        base = [a for a in communs if a in LIBELLES_DIRECTION]
        autres = communs - set(base)
        morceaux = []

        # direction "physique" -- seulement sur les axes de base x/y/z
        for a in base:
            delta = cible_coord[a] - origine_coord[a]
            if abs(delta) < 1e-6:
                continue
            plus, moins = LIBELLES_DIRECTION[a]
            morceaux.append(plus if delta > 0 else moins)

        # axes invents (nes d'un reve/forge/test) -- direction semantique,
        # nommee par l'axe lui-meme, faute de sens spatial intuitif
        for a in autres:
            delta = cible_coord[a] - origine_coord[a]
            if abs(delta) < 1e-6:
                continue
            morceaux.append(f"vers l'axe « {a} »")

        if not morceaux:
            return ", juste a cote de toi"
        return ", " + " et ".join(morceaux)


# ---- Exemple d'utilisation (ne s'execute jamais a l'import) ----------
if __name__ == "__main__":
    from cortex.modules.boite_infinie import BoiteInfinie

    boite = BoiteInfinie()  # sans persistance : aucune ecriture disque
    boite.reel.inscrire_vecu("une porte rouge grince dans le couloir",
                             etat={"curiosite": 0.6}, saillance=0.7)
    boite.reve.rever(ennui=0.4)
    boite.corps.marcher(boite.corps.pos.copie())

    perception = CoucheDePerception()
    print(perception.decrire(boite))

    # Cette description devient le prompt (ou une partie du prompt)
    # envoye a Cortex via cortex.bridge.get_response(...), a la place
    # d'un dump brut de statistiques() ou d'un contexte dict.
