"""
CORTEX — Gating fractal et activation dynamique.

Section 4.4 du document :
    A_{n,i} = sigmoid(W_g · X_{n,i} + b_g) · tanh(Z_{n,i})

    « Les sous-modèles peuvent être mis en sommeil ou activés dynamiquement :
    si la contribution d'un modèle de niveau N est jugée faible, il reçoit
    un gain G_{n,i} proche de 0. »

    « Cette modulation permet de gérer la charge de calcul et d'éviter
    un déclenchement inutile de tous les 3 125 composants. »

ÉQUILIBRAGE DE CHARGE (anti-effondrement du routage) :
    Diagnostic (voir diagnostic_gating.py) : sans garde-fou, le routeur
    apprend a desactiver quasiment tous les enfants en permanence (cercle
    vicieux bien connu des architectures MoE - "gate collapse"), ce qui
    plafonne la capacite reellement utilisee au seul tronc commun.

    activer_capture_equilibrage() / perte_equilibrage() permettent a
    train.py de calculer une perte auxiliaire qui encourage le taux
    d'activation moyen (sur tout l'arbre parcouru pendant un forward) a
    rester proche d'une cible raisonnable, au lieu de s'effondrer a zero.
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig

# ── Capture des gains pour la perte d'equilibrage (voir train.py) ──────
_capture_active = False
_gains_captures: list = []


def activer_capture_equilibrage() -> None:
    """A appeler juste avant un forward() d'entrainement dont on veut
    calculer la perte d'equilibrage de charge."""
    global _capture_active, _gains_captures
    _capture_active = True
    _gains_captures = []


def desactiver_capture_equilibrage() -> None:
    """A appeler juste apres avoir recupere perte_equilibrage() - la
    capture reste sinon inactive par defaut (generation normale, pas de
    surcout)."""
    global _capture_active
    _capture_active = False


def perte_equilibrage(cible: float = 0.2, poids: float = 0.09, poids_uniformite: float = 0.2):
    """Perte auxiliaire en DEUX parties :

    1. Ecart a la cible : penalise l'ecart entre le taux d'activation
       MOYEN reellement observe et une cible raisonnable.

    2. Uniformite PAR NOEUD (voir diagnostic_gating.py) : sans ce terme,
       l'optimiseur peut satisfaire la moyenne cible en activant UN SEUL
       enfant a fond par noeud et en laissant les autres eteints.
       IMPORTANT : la variance est calculee INDIVIDUELLEMENT pour chaque
       noeud visite (dim=1, pas une moyenne globale sur tout l'arbre) -
       un premier essai qui moyennait d'abord sur l'arbre entier pouvait
       masquer un effondrement present a CHAQUE noeud si des noeuds
       differents favorisaient des indices differents (l'agregat semblait
       equilibre en apparence, alors qu'aucun noeud individuel ne l'etait
       reellement). Ce terme penalise donc chaque noeud individuellement,
       peu importe ce que font les autres.

       poids_uniformite releve de 0.03 a 0.2 : le premier poids etait trop
       faible pour surmonter la saturation de la sigmoide une fois les
       gains figes proches de 0/1 (gradient local quasi nul a ces
       extremes) - constate en pratique, le motif "un seul enfant actif"
       persistait identique apres des milliers de pas malgre le terme.
       Voir aussi active_mask ci-dessous (garantie architecturale directe,
       independante du poids de cette perte).

    Retourne None si aucun FractalGate n'a ete traverse (capture vide).
    """
    if not _gains_captures:
        return None
    tous = torch.cat([g.reshape(-1, g.shape[-1]) for g in _gains_captures], dim=0)  # (N, n_models)

    moyenne_globale = tous.mean()
    perte_cible = (moyenne_globale - cible) ** 2

    variance_par_noeud = tous.var(dim=1)  # (N,) - variance des gains DE CHAQUE noeud individuellement
    perte_uniformite = variance_par_noeud.mean()

    return poids * perte_cible + poids_uniformite * perte_uniformite


class FractalGate(nn.Module):
    """Mécanisme de gating fractal.

    Contrôle l'activation de chaque sous-modèle dans la hiérarchie :
        - A_{n,i} : activation du noeud courant
        - G_{n,i} : gain par enfant (contrôle l'activation/sommeil)
        - C0      : coefficient d'activation central (appris)
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        d = config.d_embed
        self.sleep_threshold = config.sleep_threshold
        self.n_children = config.n_models
        self.max_active_children = config.max_active_children
        self.gate_ratio_secondaire = config.gate_ratio_secondaire

        # ── Projection de gating : W_g, b_g ────────────────────────────
        self.gate_proj = nn.Linear(d, d)

        # ── Gains par enfant : G_{n,i} ∈ [0, 1] pour chaque enfant ────
        self.gain_net = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, config.n_models),
            nn.Sigmoid(),
        )

        # ── Coefficient d'activation central C0 (appris) ───────────────
        self.c0 = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        Args:
            x: (B, T, D) — entrée du noeud courant
            z: (B, T, D) — sortie du transformer décodeur

        Returns:
            activation: (B, T, D) — A_{n,i} (activation du noeud)
            gains:      (B, M)    — G_{n,i} (gain par enfant)
            active_mask: (M,)     — masque booléen des enfants actifs
        """
        # ── Activation : A = sigmoid(Wg · X + bg) · tanh(Z) · C0 ──────
        activation = torch.sigmoid(self.gate_proj(x)) * torch.tanh(z)
        activation = activation * self.c0

        # ── Gains par enfant : décision basée sur le pooling séquentiel ─
        z_pooled = z.mean(dim=1)             # (B, D)
        gains = self.gain_net(z_pooled)       # (B, M)

        # Capture (si active) pour la perte d'equilibrage de charge -
        # garde le tenseur DANS le graphe d'autograd (pas de .detach()),
        # sinon la perte auxiliaire ne produirait aucun gradient reel.
        if _capture_active:
            _gains_captures.append(gains)

        # ── Masque d'activation : plafond dur + seuil renforce ─────────
        # HISTORIQUE : la version precedente (seuil_mask = avg_gains >
        # sleep_threshold, OR-e avec un force_mask top-1) n'avait AUCUNE
        # limite superieure - n'importe quel nombre d'enfants au-dessus du
        # seuil s'activait. A froid (poids aleatoires), les scores softmax
        # sont quasi uniformes (~1/n_models, ce qui colle exactement a
        # CIBLE_EQUILIBRAGE=0.2) : un exces meme leger (2-3 actifs au lieu
        # de 1) se multiplie en cascade sur les 5 niveaux de l'arbre.
        # Confirme empiriquement par diagnostic_activation_oom.py : le
        # pas 8 d'un cold-start a fait passer les parametres actifs de 77M
        # (tronc seul) a 1 095 044 244 (quasi tout l'arbre), crash CUDA
        # OOM immediat.
        #
        # NOUVELLE APPROCHE : plafond DUR a max_active_children (3 par
        # defaut) - torch.topk borne deja le nombre de CANDIDATS a ce
        # plafond avant meme d'appliquer un seuil, donc depasser ce nombre
        # est structurellement impossible ici (pas une question de
        # probabilite). Le rang 1 (meilleur score) est toujours actif -
        # equivalent a l'ancien k_min=1. Les rangs 2 et 3 ne s'activent que
        # si leur score depasse gate_secondaire_seuil (plus strict que
        # sleep_threshold) - rend "2 ou 3 actifs" rare en pratique, le cas
        # courant restant 1 seul actif.
        #
        # Pire cas theorique avec cap=3 sur 5 niveaux : 3+9+27+81+243 =
        # 363 noeuds actifs max (~102M valeurs) - loin en-dessous du
        # budget de 6 Go de VRAM (marge x2 environ meme dans ce pire cas
        # absolu, jamais observe en pratique).
        avg_gains = gains.mean(dim=0)         # (M,)

        k_max = min(self.max_active_children, self.n_children)
        topk_vals, topk_idx = torch.topk(avg_gains, k_max)

        active_mask = torch.zeros(self.n_children, dtype=torch.bool, device=avg_gains.device)
        active_mask[topk_idx[0]] = True  # rang 1 : toujours actif

        # Seuil ADAPTATIF : une fraction du meilleur gain de CE noeud, pas
        # une valeur absolue - s'ajuste automatiquement si l'echelle des
        # gains derive au fil de l'entrainement (observe en pratique : le
        # ratio 2e/1er gain reste stable ~0.5-0.6 meme quand les valeurs
        # absolues chutent d'un facteur 5-10). Evite d'avoir a recalibrer
        # gate_ratio_secondaire manuellement a chaque nouvelle tranche
        # d'entrainement.
        meilleur_gain = topk_vals[0].clamp(min=1e-8)  # evite division par ~0
        seuil_relatif = meilleur_gain * self.gate_ratio_secondaire

        for rang in range(1, k_max):
            if topk_vals[rang] > seuil_relatif:
                active_mask[topk_idx[rang]] = True

        return activation, gains, active_mask
