"""
vision.py - Les "yeux" de Cortex : encode une image (produite par
rendu_boite.py) en un vecteur, puis le projette dans l'espace interne
de Cortex (d_embed=512).

L'ENCODEUR (SigLIP-base, google/siglip-base-patch16-224) est
pre-entraine, fige, deja fonctionnel -- aucun entrainement necessaire
dessus, aucune donnee a lui fournir pour qu'il "sache regarder".

Le PROJECTEUR (ProjecteurVision), lui, est une simple couche lineaire
NON ENTRAINEE (poids aleatoires a l'init) : tant qu'il n'a pas ete
entraine sur des exemples reels (image -> ce que Cortex doit "en
comprendre"), il transmettrait du bruit a Cortex, pas une vraie
perception. C'est exactement le principe des modeles vision-langage
type LLaVA : encodeur de vision fige + petit projecteur a entrainer
(bien plus leger qu'entrainer une vision from scratch).

>>> [BRANCHEMENT MODELE] Rien n'est charge tant qu'aucune fonction de
ce fichier n'est appelee (import paresseux de transformers/SigLIP,
meme principe que cortex/bridge.py). NON BRANCHE pour l'instant --
pose ici, dormant.
"""

import torch
import torch.nn as nn

NOM_MODELE_VISION = "google/siglip-base-patch16-224"
DIM_VISION = 768   # sortie de SigLIP-base (pooler_output)
DIM_CORTEX = 512   # cortex.config.CortexConfig().d_embed

_processeur = None
_encodeur = None


def _charger_encodeur(device="cpu"):
    """Charge SigLIP une seule fois, paresseusement. Fige : jamais
    entraine, jamais mis a jour par un optimiseur."""
    global _processeur, _encodeur
    if _encodeur is None:
        from transformers import SiglipVisionModel, SiglipImageProcessor
        _processeur = SiglipImageProcessor.from_pretrained(NOM_MODELE_VISION)
        _encodeur = SiglipVisionModel.from_pretrained(NOM_MODELE_VISION)
        _encodeur.eval()
        for p in _encodeur.parameters():
            p.requires_grad = False
        _encodeur.to(device)
    return _processeur, _encodeur


class ProjecteurVision(nn.Module):
    """Pont entre l'espace visuel (SigLIP, 768d) et l'espace interne de
    Cortex (512d). NON ENTRAINE : poids aleatoires a l'initialisation.
    A entrainer avant tout usage reel -- sinon c'est du bruit injecte
    dans Cortex, pas une perception."""

    def __init__(self, dim_in=DIM_VISION, dim_out=DIM_CORTEX):
        super().__init__()
        self.projection = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, embedding_vision):
        return self.norm(self.projection(embedding_vision))


@torch.no_grad()
def voir(image, device="cpu"):
    """image (PIL.Image) -> embedding SigLIP brut, shape (1, 768).
    Non projete -- fonction pure, testable isolement, sans effet de
    bord sur Cortex."""
    processeur, encodeur = _charger_encodeur(device)
    entrees = processeur(images=image, return_tensors="pt")
    entrees = {k: v.to(device) for k, v in entrees.items()}
    sortie = encodeur(**entrees)
    return sortie.pooler_output  # (1, 768)


# ---- Exemple d'utilisation (ne s'execute jamais a l'import) ----------
if __name__ == "__main__":
    from cortex.modules.boite_infinie import BoiteInfinie
    from cortex.modules.rendu_boite import RenduBoite

    boite = BoiteInfinie()
    boite.reel.inscrire_vecu("une porte rouge grince", etat={"curiosite": 0.6}, saillance=0.6)
    image = RenduBoite().capturer(boite)

    embedding = voir(image)
    projecteur = ProjecteurVision()
    vecteur_cortex = projecteur(embedding)

    print("SigLIP  :", tuple(embedding.shape))
    print("Projete :", tuple(vecteur_cortex.shape), "(non entraine)")
