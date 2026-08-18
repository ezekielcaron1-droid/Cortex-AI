"""
rendu_boite.py - Rendu visuel temps reel de l'etat de BoiteInfinie :
transforme les zones/entites en une VRAIE image (surfaces 3D), pas du
texte -- destinee a etre "vue" par un encodeur de vision (cf.
vision.py), pas lue comme du langage.

Vue FPS : le Corps ne se voit jamais lui-meme, seulement ce qui
l'entoure. Le sol porte la couleur de la zone (teinte, quasi
transparent) -- plus de sphere d'ambiance.

"Temps reel" ici signifie : chaque appel a capturer() lit l'etat EXACT
de la boite au moment de l'appel et regenere l'image en consequence --
pas une boucle qui tourne en continu toute seule.

>>> [BRANCHEMENT MODELE] Ce module ne connait pas Cortex : il produit
une PIL.Image, rien d'autre. NON BRANCHE pour l'instant -- pose ici,
dormant, comme le reste (hai_v2.py, boite_infinie.py, perception.py).
"""

import io
import math
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (requis pour projection="3d")
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

COULEURS_ESPACE = {
    "reel": "#2f6fe0",
    "reve": "#c04dff",
    "forge": "#ff8c1a",
    "test": "#1fd67a",
}

COULEURS_ORIGINE = {
    "vecu": "#5ec8ff",
    "reve": "#e084ff",
    "forge": "#ffb648",
    "test": "#5cf2a0",
}

LEGENDES_ESPACE = {
    "reel": "REALITE",
    "reve": "REVE",
    "forge": "FORGE",
    "test": "LABO",
}


class Crayon:
    """Outil de sculpture generique : QUELQU'UN (Cortex, plus tard --
    moi, en test) cree un point vierge puis le deforme a volonte,
    couche par couche, sans forme predefinie ni catalogue. 0 couche =
    un point/sphere neutre. Aucune limite creative sur les valeurs --
    seulement des garde-fous anti-crash (une entree invalide est
    ignoree, jamais fatale)."""

    MAX_COUCHES = 200  # garde-fou anti-emballement, pas une limite creative
    AMPLITUDE_MAX = 1.5  # au-dela, le maillage se retourne sur lui-meme

    @staticmethod
    def creer_point(entite):
        """Initialise une entite comme un point vierge, sans forme."""
        entite.contenu.setdefault("sculpture", [])

    @staticmethod
    def deformer(entite, freq_u=1.0, freq_v=1.0, amplitude=0.3, phase=0.0):
        """Ajoute UNE couche de deformation. Retourne True si acceptee,
        False si ignoree (entree invalide ou plafond atteint) -- ne
        leve jamais d'exception, pour ne jamais casser l'appelant."""
        couches = entite.contenu.setdefault("sculpture", [])
        if len(couches) >= Crayon.MAX_COUCHES:
            return False
        try:
            freq_u = float(freq_u)
            freq_v = float(freq_v)
            amplitude = max(-Crayon.AMPLITUDE_MAX, min(Crayon.AMPLITUDE_MAX, float(amplitude)))
            phase = float(phase)
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(v) for v in (freq_u, freq_v, amplitude, phase)):
            return False  # NaN/inf : ignore, ne propage jamais
        couches.append((freq_u, freq_v, amplitude, phase))
        return True

    @staticmethod
    def effacer(entite):
        entite.contenu["sculpture"] = []

    # -- Trace (dessin libre point par point, pour les formes que la
    # deformation harmonique ne peut pas produire -- angles droits,
    # traits, lettres...) ------------------------------------------------
    MAX_POINTS_PARCOURS = 500

    @staticmethod
    def commencer_parcours(entite):
        """Demarre un trace vierge (liste de points), independant de
        la sculpture harmonique -- les deux peuvent coexister."""
        entite.contenu.setdefault("parcours", [])

    @staticmethod
    def tracer_vers(entite, x, y, z):
        """Pose un point, en coordonnees LOCALES (relatives au centre
        de l'entite, pas les coordonnees globales de la boite). Un
        segment sera dessine depuis le point precedent. Jamais fatal :
        entree invalide -> ignoree."""
        points = entite.contenu.setdefault("parcours", [])
        if len(points) >= Crayon.MAX_POINTS_PARCOURS:
            return False
        try:
            x, y, z = float(x), float(y), float(z)
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(v) for v in (x, y, z)):
            return False
        points.append((x, y, z))
        return True

    @staticmethod
    def effacer_parcours(entite):
        entite.contenu["parcours"] = []

    # -- Couleur libre (n'importe laquelle du cercle chromatique, pas
    # seulement la palette liee a l'origine) ----------------------------
    @staticmethod
    def colorer(entite, couleur):
        """Definit une couleur libre pour CETTE entite -- hex ('#ff2222'),
        nom ('red', 'cyan', ...), ou tuple RGB(A) 0..1. N'importe quelle
        couleur du cercle chromatique, pas limite a la palette par
        origine. Entree invalide -> ignoree, jamais fatale."""
        if not mcolors.is_color_like(couleur):
            return False
        entite.contenu["couleur"] = couleur
        return True

    @staticmethod
    def effacer_couleur(entite):
        """Revient a la couleur par defaut (liee a l'origine)."""
        entite.contenu.pop("couleur", None)

    @staticmethod
    def colorer_segment(entite, index_segment, couleur):
        """Colore UN SEGMENT precis du parcours (entre le point
        index_segment et le suivant), independamment du reste --
        permet par exemple une moitie bleue, une moitie verte. Prime
        sur la couleur generale de l'entite pour ce segment seulement.
        Entree invalide (index negatif, couleur invalide) -> ignoree."""
        if not mcolors.is_color_like(couleur):
            return False
        try:
            index_segment = int(index_segment)
        except (TypeError, ValueError):
            return False
        if index_segment < 0:
            return False
        couleurs = entite.contenu.setdefault("couleurs_parcours", {})
        couleurs[index_segment] = couleur
        return True

    @staticmethod
    def effacer_couleurs_segments(entite):
        entite.contenu["couleurs_parcours"] = {}

    # -- Remplissage (surface pleine refermant le parcours, comme
    # l'outil "forme libre" de Paint -- au lieu d'un squelette de
    # tubes) ---------------------------------------------------------
    @staticmethod
    def remplir(entite, actif=True):
        """Bascule le remplissage : si actif et >= 3 points traces, le
        contour du parcours devient une surface pleine refermee,
        plutot que des tubes segment par segment."""
        entite.contenu["rempli"] = bool(actif)


class RenduBoite:
    """Convertit l'etat vivant d'une BoiteInfinie en une image reelle
    (surfaces 3D), depuis le point de vue du Corps (vue FPS)."""

    def __init__(self, taille_px=(512, 512), dpi=100, portee=4.0):
        self.taille_px = taille_px
        self.dpi = dpi
        self.portee = portee  # rayon (dans l'espace) affiche autour du corps

    # ------------------------------------------------------------------
    def capturer(self, boite) -> Image.Image:
        """Rend l'etat ACTUEL de la boite -> une PIL.Image en memoire.
        Vue FPS : on ne voit jamais son propre corps. Aucune ecriture
        disque. Aucun etat garde entre deux appels."""
        fig = plt.figure(
            figsize=(self.taille_px[0] / self.dpi, self.taille_px[1] / self.dpi),
            dpi=self.dpi,
        )
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#05060a")
        fig.patch.set_facecolor("#05060a")

        corps = boite.corps
        centre = self._coord_xyz(corps.pos)

        self._dessiner_sol(ax, corps.espace, centre)
        self._dessiner_entites(ax, boite, corps, centre)

        # Vue PREMIERE PERSONNE : pas une camera qui orbite au-dessus.
        # Champ de vision etroit devant soi (azim=-90 = on regarde vers
        # les entites, placees en -y), quasiment rien derriere, angle
        # bas (elev proche de l'horizontale, hauteur des yeux).
        ax.set_xlim(centre[0] - self.portee * 0.6, centre[0] + self.portee * 0.6)
        ax.set_ylim(centre[1] - self.portee * 1.8, centre[1] + 0.3)
        ax.set_zlim(centre[2] - self.portee * 0.25, centre[2] + self.portee * 0.35)
        ax.set_axis_off()
        # Perspective forte (focal_length bas) : le sol doit vraiment
        # converger vers un point de fuite, pas rester plat comme en
        # projection orthographique.
        ax.set_proj_type("persp", focal_length=0.42)
        ax.view_init(elev=6, azim=-90)

        if corps.espace is None:
            legende = "HUB"
        else:
            legende = LEGENDES_ESPACE.get(corps.espace, corps.espace.upper())
        ax.text2D(0.03, 0.95, f"TU ES DANS : {legende}", transform=ax.transAxes,
                 color="white", fontsize=13, family="sans-serif", alpha=0.85)

        image = self._figure_vers_image(fig)
        plt.close(fig)
        return image

    # ------------------------------------------------------------------
    def _coord_xyz(self, coord):
        """Projette une Coord (axes malleables) sur x/y/z. Si x/y/z
        existent reellement, on les utilise. Sinon (axes 100%
        abstraits), position stable et espacee -- _position_abstraite()."""
        axes = coord.axes()
        if axes & {"x", "y", "z"}:
            return tuple(coord[a] if a in axes else 0.0 for a in ("x", "y", "z"))
        return self._position_abstraite(coord)

    def _position_abstraite(self, coord, rayon_min=1.0, rayon_max=2.6):
        """Aucun axe spatial : position deterministe (stable pour une
        meme entite), sur une coquille [rayon_min, rayon_max]."""
        axes = sorted(coord.axes())
        cle = "|".join(f"{a}:{round(coord[a], 4)}" for a in axes) or "vide"
        h = hashlib.sha256(cle.encode("utf-8")).digest()
        bruts = [(h[i] / 255.0) * 2 - 1 for i in range(3)]
        norme = math.sqrt(sum(v * v for v in bruts)) or 1.0
        direction = [v / norme for v in bruts]
        rayon = rayon_min + (h[3] / 255.0) * (rayon_max - rayon_min)
        return tuple(d * rayon for d in direction)

    def _figure_vers_image(self, fig) -> Image.Image:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    # ------------------------------------------------------------------
    def _dessiner_sol(self, ax, nom_espace, centre):
        """Le sol : lignes de fuite PARALLELES dans le monde (meme
        largeur pres et loin) qui S'ESTOMPENT PROGRESSIVEMENT vers
        l'horizon (segments d'opacite decroissante) plutot que de
        s'arreter net -- matplotlib ne rend rien au-dela d'une certaine
        distance (limite de la librairie), donc on simule l'infini par
        un fondu au lieu de pousser les coordonnees, ce qui casse le
        rendu."""
        couleur = COULEURS_ESPACE.get(nom_espace, "#888888")
        z_sol = centre[2] - self.portee * 0.25
        y_proche = centre[1] + 0.4
        y_loin = centre[1] - self.portee * 6.0  # distance fiable (au-dela, matplotlib ne rend plus rien)
        largeur = self.portee * 4.0  # identique pres ET loin

        n_segments = 5
        alphas = [0.55, 0.4, 0.27, 0.16, 0.07]  # fondu vers le noir = "continue a l'infini"

        n_lignes = 14
        for i in range(n_lignes + 1):
            t = i / n_lignes  # 0..1
            x = centre[0] + (t - 0.5) * 2 * largeur
            for s in range(n_segments):
                t0, t1 = s / n_segments, (s + 1) / n_segments
                y0 = y_proche + t0 * (y_loin - y_proche)
                y1 = y_proche + t1 * (y_loin - y_proche)
                ax.plot([x, x], [y0, y1], [z_sol, z_sol],
                        color=couleur, alpha=alphas[s], linewidth=1.0)

        # Barreaux transversaux, de plus en plus rapproches ET de plus
        # en plus estompes vers l'horizon.
        positions_t = (0.04, 0.12, 0.25, 0.42, 0.65, 0.85)
        for j, t in enumerate(positions_t):
            y = y_proche + t * (y_loin - y_proche)
            alpha = 0.32 * (1 - t)
            ax.plot([centre[0] - largeur, centre[0] + largeur], [y, y], [z_sol, z_sol],
                    color=couleur, alpha=max(0.03, alpha), linewidth=0.9)

    def _graine_forme(self, entite):
        """Empreinte numerique stable, tiree de l'etat REEL de
        l'entite -- deux entites ne se ressemblent jamais pour les
        memes raisons que deux entites ne sont jamais identiques."""
        charge = getattr(entite, "charge", {}) or {}
        cle = "|".join([
            entite.origine,
            entite.trace(),
            str(round(entite.saillance, 4)),
            str(round(entite.certitude_realite, 4)),
            str(entite.coord.dim()),
            "|".join(f"{k}:{round(v, 4)}" for k, v in sorted(charge.items())),
        ])
        return hashlib.sha256(cle.encode("utf-8")).digest()

    def _forme_libre(self, ax, centre, rayon_base, couleur, alpha, entite,
                     resolution=16):
        """Dessine la surface EXACTEMENT telle que sculptee via Crayon
        -- aucune forme devinee. 0 couche = un point/sphere neutre.
        Chaque couche invalide est ignoree individuellement (une
        sculpture corrompue ne doit jamais faire planter le rendu)."""
        couches = entite.contenu.get("sculpture", [])

        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        U, V = np.meshgrid(u, v)

        deformation = np.ones_like(U)
        for couche in couches[: Crayon.MAX_COUCHES]:
            try:
                freq_u, freq_v, amp, phase = couche
                deformation = deformation + amp * np.sin(freq_u * U + phase) * np.sin(freq_v * V)
            except Exception:
                continue  # une couche cassee ne casse pas les autres
        deformation = np.nan_to_num(deformation, nan=1.0, posinf=3.0, neginf=0.25)
        deformation = np.clip(deformation, 0.25, 3.0)

        r = rayon_base * deformation
        x = centre[0] + r * np.cos(U) * np.sin(V)
        y = centre[1] + r * np.sin(U) * np.sin(V)
        z = centre[2] + r * np.cos(V)
        ax.plot_surface(x, y, z, color=couleur, alpha=alpha,
                        linewidth=0, shade=True)

    def _tube_segment(self, p0, p1, rayon, section="carre"):
        """Prisme reliant deux points 3D -- une VRAIE surface. Par
        defaut une section CARREE (4 faces plates, coins nets -- un L
        c'est des barres rectangulaires, pas des tubes ronds).
        section='cercle' reste disponible pour d'autres usages ronds.
        Retourne None si les deux points sont confondus."""
        p0 = np.array(p0, dtype=float)
        p1 = np.array(p1, dtype=float)
        axe = p1 - p0
        longueur = np.linalg.norm(axe)
        if longueur < 1e-6:
            return None
        axe_n = axe / longueur
        arbitraire = np.array([1.0, 0.0, 0.0]) if abs(axe_n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp1 = np.cross(axe_n, arbitraire)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(axe_n, perp1)

        if section == "carre":
            # 4 coins aux diagonales (45/135/225/315) -> les 4 FACES
            # resultantes sont bien plates, alignees haut/bas/gauche/
            # droite par rapport a perp1/perp2 -- une vraie barre.
            theta = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4, np.pi / 4])
            rayon_eff = rayon * math.sqrt(2)
        else:
            theta = np.linspace(0, 2 * np.pi, 10)
            rayon_eff = rayon

        L = np.linspace(0, longueur, 2)
        THETA, LL = np.meshgrid(theta, L)
        forme = rayon_eff * (np.cos(THETA)[..., None] * perp1 + np.sin(THETA)[..., None] * perp2)
        X = p0[0] + LL * axe_n[0] + forme[..., 0]
        Y = p0[1] + LL * axe_n[1] + forme[..., 1]
        Z = p0[2] + LL * axe_n[2] + forme[..., 2]
        return X, Y, Z

    def _dessiner_parcours(self, ax, centre, entite, couleur, alpha, epaisseur=0.12):
        """Trace le crayon : soit une SURFACE PLEINE refermant le
        contour (si Crayon.remplir() actif -- comme l'outil forme libre
        de Paint), soit des tubes segment par segment (par defaut).
        Coordonnees locales -> deplacees au centre de l'entite. Points
        casses ignores, jamais fatal."""
        points = entite.contenu.get("parcours", [])

        if entite.contenu.get("rempli") and len(points) >= 3:
            try:
                pts = [(centre[0] + p[0], centre[1] + p[1], centre[2] + p[2]) for p in points]
            except Exception:
                pts = None
            if pts:
                try:
                    poly = Poly3DCollection([pts], facecolor=couleur, alpha=alpha, edgecolor="none")
                    ax.add_collection3d(poly)
                    return
                except Exception:
                    pass  # contour invalide (points non exploitables) -> repli sur les tubes

        couleurs_segments = entite.contenu.get("couleurs_parcours", {})
        for i in range(len(points) - 1):
            try:
                p0 = (centre[0] + points[i][0], centre[1] + points[i][1], centre[2] + points[i][2])
                p1 = (centre[0] + points[i + 1][0], centre[1] + points[i + 1][1], centre[2] + points[i + 1][2])
            except Exception:
                continue
            segment = self._tube_segment(p0, p1, epaisseur)
            if segment is None:
                continue
            X, Y, Z = segment
            couleur_segment = couleurs_segments.get(i, couleur)
            ax.plot_surface(X, Y, Z, color=couleur_segment, alpha=alpha, linewidth=0, shade=True)

    # ------------------------------------------------------------------
    def _dessiner_entites(self, ax, boite, corps, centre):
        """Chaque entite percue -> soit un TRACE au crayon (si elle en
        a un -- priorite, car explicite), soit une sculpture harmonique,
        soit un simple point. Posee sur le sol, ORGANISEE DU PLUS RECENT
        (le plus proche) AU MOINS RECENT (le plus loin)."""
        espace = boite.espace(corps.espace)
        if espace is None:
            return
        voisins = espace.voisinage(corps.pos, rayon=self.portee)
        voisins = sorted(voisins, key=lambda e: e.cree_a, reverse=True)

        hauteur_sol = centre[2] - self.portee * 0.25
        n = max(1, len(voisins))
        for i, entite in enumerate(voisins):
            h = self._graine_forme(entite)
            decalage_x = ((h[10] / 255.0) * 2 - 1) * (self.portee * 0.35)
            distance = 0.9 + i * (self.portee * 1.3) / n
            couleur = entite.contenu.get("couleur") or COULEURS_ORIGINE.get(entite.origine, "#e0e0e0")
            alpha = 0.55 + 0.4 * max(0.0, min(1.0, entite.certitude_realite))

            points = entite.contenu.get("parcours")
            if points:
                # Le point le plus bas du trace doit toucher le sol --
                # pas une formule pensee pour une sphere symetrique.
                try:
                    z_min_local = min(p[2] for p in points)
                except Exception:
                    z_min_local = 0.0
                pos = (centre[0] + decalage_x, centre[1] - distance,
                      hauteur_sol - z_min_local)
                self._dessiner_parcours(ax, pos, entite, couleur, alpha)
            else:
                rayon = 0.2 + 0.45 * max(0.0, min(1.0, entite.saillance))
                pos = (centre[0] + decalage_x, centre[1] - distance,
                      hauteur_sol + rayon)
                self._forme_libre(ax, pos, rayon, couleur, alpha, entite)


# ---- Exemple d'utilisation (ne s'execute jamais a l'import) ----------
if __name__ == "__main__":
    from cortex.modules.boite_infinie import BoiteInfinie

    boite = BoiteInfinie()
    boite.reel.inscrire_vecu("une porte rouge grince dans le couloir",
                             etat={"curiosite": 0.6}, saillance=0.7)
    boite.reve.rever(ennui=0.4)

    rendu = RenduBoite()
    image = rendu.capturer(boite)
    image.save("apercu_boite.png")
    print("Image generee :", image.size, image.mode)
