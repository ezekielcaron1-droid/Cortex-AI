"""
rendu_hub.py - Vue "plan d'ensemble" : les 4 pieces (Realite/Reve/
Forge/Labo) cote a cote, cloisons transparentes teintees, legendees --
la vue qu'on a depuis le Hub (spawn), PAS depuis l'interieur d'une
zone. Les colonnes font TOUJOURS la meme taille, quel que soit leur
contenu (contrairement a l'interieur d'une zone, qui lui est "infini"
-- cf. rendu_boite.py pour la vue immersive interieure).

>>> [BRANCHEMENT MODELE] Produit une PIL.Image, rien d'autre. NON
BRANCHE pour l'instant -- dormant, comme le reste.
"""

import hashlib
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from cortex.modules.rendu_boite import COULEURS_ESPACE, COULEURS_ORIGINE, Crayon

ORDRE_ZONES = ["reel", "reve", "forge", "test"]
LEGENDES = {"reel": "REALITE", "reve": "REVE", "forge": "FORGE", "test": "LABO"}


class RenduHub:
    """Vue d'ensemble des 4 pieces, toujours a taille egale, telle
    qu'on la voit depuis le Hub (spawn) -- pas depuis l'interieur."""

    def __init__(self, taille_px=(1024, 576), dpi=100, max_objets=6):
        self.taille_px = taille_px
        self.dpi = dpi
        self.max_objets = max_objets

    # ------------------------------------------------------------------
    def capturer(self, boite) -> Image.Image:
        fig, ax = plt.subplots(
            figsize=(self.taille_px[0] / self.dpi, self.taille_px[1] / self.dpi),
            dpi=self.dpi)
        fig.patch.set_facecolor("#050608")
        ax.set_facecolor("#050608")

        n = len(ORDRE_ZONES)
        largeur = 1.0 / n

        for i, nom in enumerate(ORDRE_ZONES):
            x0 = i * largeur
            couleur = COULEURS_ESPACE[nom]
            ax.add_patch(plt.Rectangle(
                (x0, 0), largeur, 1, transform=ax.transAxes,
                color=couleur, alpha=0.07, zorder=0))
            ax.text(x0 + largeur / 2, 0.94, LEGENDES[nom],
                    transform=ax.transAxes, color="white", fontsize=17,
                    ha="center", va="top", family="sans-serif", zorder=4)
            self._dessiner_zone(ax, boite, nom, x0, largeur)
            if i < n - 1:
                self._cloison(ax, x0 + largeur, couleur,
                             COULEURS_ESPACE[ORDRE_ZONES[i + 1]])

        self._marquer_corps(ax, boite)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        image = self._figure_vers_image(fig)
        plt.close(fig)
        return image

    # ------------------------------------------------------------------
    def _cloison(self, ax, x, couleur_gauche, couleur_droite):
        """Cloison transparente : deux traits fins, chacun teinte de la
        couleur de la piece qu'il borde -- jamais un mur plein/opaque."""
        ax.axvline(x, ymin=0, ymax=1, color=couleur_gauche, alpha=0.35,
                  linewidth=2, zorder=3)
        ax.axvline(x + 0.002, ymin=0, ymax=1, color=couleur_droite, alpha=0.35,
                  linewidth=2, zorder=3)

    def _marquer_corps(self, ax, boite):
        """Cadre blanc autour de la piece ou se trouve reellement le
        Corps en ce moment -- "on sait dans quelle partie on est"."""
        nom = boite.corps.espace
        if nom not in ORDRE_ZONES:
            return
        i = ORDRE_ZONES.index(nom)
        largeur = 1.0 / len(ORDRE_ZONES)
        ax.add_patch(plt.Rectangle(
            (i * largeur, 0.01), largeur, 0.98, transform=ax.transAxes,
            fill=False, edgecolor="white", linewidth=1.5, alpha=0.55,
            zorder=5))

    def _figure_vers_image(self, fig) -> Image.Image:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    # ------------------------------------------------------------------
    def _dessiner_zone(self, ax, boite, nom, x0, largeur):
        espace = boite.espace(nom)
        if espace is None or not espace.entites:
            return
        entites = sorted(espace.entites.values(),
                         key=lambda e: e.cree_a, reverse=True)[: self.max_objets]
        for entite in entites:
            self._forme_2d(ax, entite, x0, largeur)

    def _forme_2d(self, ax, entite, x0, largeur):
        """Version plate (2D) : soit le TRACE au crayon projete (x,z),
        soit la sculpture harmonique, soit un rond neutre -- meme
        priorite que la vue immersive. La POSITION reste deterministe/
        hachee (pas de la forme, juste un emplacement stable)."""
        charge = getattr(entite, "charge", {}) or {}
        cle = "|".join([
            entite.origine, entite.trace(),
            str(round(entite.saillance, 4)),
            str(round(entite.certitude_realite, 4)),
            "|".join(f"{k}:{round(v, 4)}" for k, v in sorted(charge.items())),
        ])
        h = hashlib.sha256(cle.encode("utf-8")).digest()

        cx = x0 + largeur * (0.22 + 0.56 * (h[0] / 255.0))
        cy = 0.15 + 0.6 * (h[1] / 255.0)
        rayon_base = 0.025 + 0.05 * max(0.0, min(1.0, entite.saillance))
        aspect = self.taille_px[0] / self.taille_px[1]
        couleur = entite.contenu.get("couleur") or COULEURS_ORIGINE.get(entite.origine, "#e0e0e0")
        alpha = 0.55 + 0.35 * max(0.0, min(1.0, entite.certitude_realite))

        points = entite.contenu.get("parcours")
        if points:
            couleurs_segments = entite.contenu.get("couleurs_parcours", {})
            rempli = bool(entite.contenu.get("rempli"))
            self._parcours_2d(ax, points, cx, cy, rayon_base, aspect, couleur, alpha, couleurs_segments, rempli)
            return

        couches = entite.contenu.get("sculpture", [])
        theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        r = np.ones_like(theta)
        for couche in couches[: Crayon.MAX_COUCHES]:
            try:
                freq_u, _freq_v, amp, phase = couche
                r = r + amp * np.sin(freq_u * theta + phase)
            except Exception:
                continue  # une couche cassee ne casse pas les autres
        r = np.nan_to_num(r, nan=1.0, posinf=3.0, neginf=0.25)
        r = np.clip(r, 0.25, 3.0) * rayon_base

        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta) * aspect
        ax.fill(xs, ys, color=couleur, alpha=alpha,
               transform=ax.transAxes, zorder=2, linewidth=0)

    def _parcours_2d(self, ax, points, cx, cy, rayon_base, aspect, couleur, alpha, couleurs_segments=None, rempli=False):
        """Le trace au crayon, aplati en 2D (axes locaux x,z -> ecran),
        mis a l'echelle pour tenir dans la case. Si rempli et >= 3
        points : une surface pleine refermee (comme Paint). Sinon,
        chaque segment garde sa propre couleur si definie."""
        try:
            xs_locaux = [p[0] for p in points]
            zs_locaux = [p[2] for p in points]
        except Exception:
            return
        etendue = max(0.3, max(abs(v) for v in xs_locaux + zs_locaux))
        echelle = (rayon_base * 2.2) / etendue
        xs = [cx + x * echelle for x in xs_locaux]
        ys = [cy + z * echelle * aspect for z in zs_locaux]

        if rempli and len(xs) >= 3:
            ax.fill(xs, ys, color=couleur, alpha=alpha,
                   transform=ax.transAxes, zorder=2, linewidth=0)
            return

        couleurs_segments = couleurs_segments or {}
        for i in range(len(xs) - 1):
            c = couleurs_segments.get(i, couleur)
            ax.plot(xs[i:i + 2], ys[i:i + 2], color=c, alpha=alpha, linewidth=3.5,
                   solid_capstyle="round", transform=ax.transAxes, zorder=2)


# ---- Exemple d'utilisation (ne s'execute jamais a l'import) ----------
if __name__ == "__main__":
    from cortex.modules.boite_infinie import BoiteInfinie

    boite = BoiteInfinie()
    boite.reel.inscrire_vecu("une porte rouge grince", etat={"curiosite": 0.7}, saillance=0.7)
    boite.reel.inscrire_vecu("un chat noir dort", etat={"calme": 0.5}, saillance=0.5)
    boite.reve.rever(ennui=0.5)
    boite.corps.changer_espace("reve")

    RenduHub().capturer(boite).save("apercu_hub.png")
    print("Hub genere")
