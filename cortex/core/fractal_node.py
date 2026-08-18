"""
CORTEX — Noeud fractal récursif, avec offloading CPU <-> GPU guidé par le
routage (Mixture of Experts).

Cœur de l'architecture fractale (sections 4.1 à 4.5).

Chaque FractalNode contient :
    1. Un TransformerDecoderBlock      — traitement principal
    2. Un MiniBrainVerifier            — vérification et correction
    3. Un FractalGate                  — contrôle d'activation des enfants
    4. 5 FractalNode enfants           — récursion (instanciation paresseuse)
    5. Un HiddenStateManager           — gestion des états cachés H_{n,i}
    6. Un ShortTermMemory              — mémoire locale M_{n,i}

La récursion s'arrête au niveau N (n_levels).

STRATÉGIE MÉMOIRE (Mixture of Experts, 2 niveaux) :
    Une fois nn.ModuleList utilisé pour enregistrer correctement les
    enfants (bugfix), le modèle complet dépasse le milliard de paramètres
    dès la première profondeur (5^N). Impossible de garder tout le monde
    en VRAM en permanence sur 6 Go.

    Les enfants sont donc créés sur CPU par défaut (là où vit la grande
    majorité du modèle). Seuls les enfants réellement ACTIFS pour ce
    passage précis (déterminés par FractalGate, comme un routeur MoE)
    sont déplacés sur GPU le temps du calcul, puis renvoyés sur CPU
    juste après. Les enfants endormis ne quittent jamais le CPU.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from cortex.config import CortexConfig
from cortex.core.transformer import TransformerDecoderBlock
from cortex.modules.mini_brain import MiniBrainVerifier
from cortex.modules.gating import FractalGate
from cortex.modules.memory import ShortTermMemory, HiddenStateManager


class FractalNode(nn.Module):
    """Noeud fractal récursif — brique de base de la hiérarchie à 5 niveaux."""

    def __init__(self, config: CortexConfig, level: int):
        super().__init__()
        self.config = config
        self.level = level
        self.max_level = config.n_levels
        self.n_children = config.n_models
        d = config.d_embed

        # ── 1. Transformer décodeur ────────────────────────────────────
        self.transformer = TransformerDecoderBlock(config)

        # ── 2. Mini-cerveau de vérification ────────────────────────────
        self.mini_brain = MiniBrainVerifier(config)

        # ── 3. Intégration du contexte fractal ─────────────────────────
        # input_{n,i} = f(concat(R_{n-1, parent}, C_n))
        self.context_proj = nn.Linear(d * 2, d)

        # ── 4. Auto-gating (activation A du noeud courant) ─────────────
        self.self_gate_proj = nn.Linear(d, d)

        # ── 5. État caché H_{n,i} ─────────────────────────────────────
        self.hidden_state_mgr = HiddenStateManager(config)

        # ── 6. Mémoire court terme M_{n,i} ─────────────────────────────
        self.memory = ShortTermMemory(capacity=config.memory_capacity)

        # ── 7. Gating enfants + agrégation (seulement si non-feuille) ──
        self._children_initialized = False
        # nn.ModuleList : PyTorch enregistre bien ces sous-modules
        # (gradients, state_dict) — contrairement à une liste Python nue.
        self._children = nn.ModuleList()

        if level < self.max_level:
            self.child_gate = FractalGate(config)
            self.child_aggregator = nn.Linear(d * config.n_models, d)

        # ── 8. Projection résiduelle → parent ──────────────────────────
        self.residual_proj = nn.Linear(d, d)

    def decharger_enfants_gpu(self) -> int:
        """Balayage recursif : renvoie sur CPU tout enfant reste sur GPU
        apres un pas d'entrainement (voir _forward_child_offloaded).

        A appeler explicitement apres optimizer.step(), une fois que le
        backward() du pas est termine et que le graphe d'autograd n'a
        plus besoin des poids en place. Retourne le nombre d'enfants
        dechargeurs (pour diagnostic/logging).
        """
        n_decharges = 0
        for child in self._children:
            try:
                sur_gpu = next(child.parameters()).is_cuda
            except StopIteration:
                sur_gpu = False
            if sur_gpu:
                child.to("cpu")
                n_decharges += 1
            n_decharges += child.decharger_enfants_gpu()
        return n_decharges

    def _init_children(self) -> None:
        """Instanciation paresseuse des enfants.

        Créés sur CPU par défaut — seuls les enfants actifs seront
        déplacés vers le GPU au moment du calcul (voir forward()).
        """
        if not self._children_initialized and self.level < self.max_level:
            for _ in range(self.n_children):
                child = FractalNode(self.config, self.level + 1)  # reste sur CPU
                self._children.append(child)
            self._children_initialized = True

    def _forward_child_offloaded(
        self,
        child: "FractalNode",
        device: torch.device,
        z_corrected: torch.Tensor,
        max_depth: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Exécute un enfant actif : le déplace sur GPU, calcule, le renvoie
        sur CPU juste après (Mixture of Experts — seuls les enfants
        sélectionnés par le routeur occupent de la VRAM, et seulement le
        temps de leur propre calcul)."""
        use_gpu = device.type == "cuda"

        if use_gpu:
            child.to(device)

        child_out, child_res, child_h, child_info = child(
            z_corrected,
            fractal_context=z_corrected,
            hidden_state=None,
            max_depth=max_depth,
            mask=mask,
        )

        # En generation (torch.no_grad()), on decharge tout de suite pour
        # limiter la VRAM. En entrainement (gradients actifs), l'enfant DOIT
        # rester sur GPU jusqu'a la fin du backward() de ce pas - sinon le
        # graphe d'autograd se retrouve avec des poids qui ont change de
        # device sous ses pieds (RuntimeError). Le dechargement se fera
        # explicitement apres optimizer.step() via decharger_enfants_gpu().
        if use_gpu and not torch.is_grad_enabled():
            child.to("cpu")
            torch.cuda.empty_cache()

        return child_out, child_res, child_h, child_info

    def forward(
        self,
        x: torch.Tensor,
        fractal_context: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        max_depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass récursif à travers ce noeud fractal.

        Args:
            x:              (B, T, D) — entrée
            fractal_context: (B, T, D) — contexte fractal (résiduel parent ou G_global)
            hidden_state:   (B, T, D) optionnel — état caché précédent
            max_depth:      (B,) optionnel — profondeur budgétée (score global du prompt)
            mask:           (T, T) optionnel — masque d'attention

        Returns:
            output:       (B, T, D) — sortie traitée
            residual:     (B, T, D) — R_{n,i} pour le parent
            hidden_state: (B, T, D) — H_{n,i} mis à jour
            info:         dict      — métriques (cohérence, confiance, enfants actifs)
        """
        B, T, D = x.shape
        device = x.device

        # ── Initialisation de l'état caché ─────────────────────────────
        if hidden_state is None:
            hidden_state = self.hidden_state_mgr.init_state(B, T, device)

        # ── 1. Fusion entrée + contexte fractal ────────────────────────
        combined = torch.cat([x, fractal_context], dim=-1)   # (B, T, 2D)
        x_in = self.context_proj(combined)                    # (B, T, D)

        # ── 2. Transformer décodeur ────────────────────────────────────
        z = self.transformer(x_in, mask)                      # (B, T, D)

        # ── 3. Mini-cerveau : vérification et correction ───────────────
        z_corrected, h_check, q, v = self.mini_brain(z)

        # ── 4. Auto-activation du noeud courant ────────────────────────
        # A_{n,i} = sigmoid(W_g · X + b_g) · tanh(Z)
        activation = torch.sigmoid(self.self_gate_proj(x_in)) * torch.tanh(z_corrected)

        # ── 5. Traitement récursif des enfants (Mixture of Experts) ────
        children_residual = torch.zeros(B, T, D, device=device)
        n_active_children = 0
        children_info = []

        if self.level < self.max_level:
            # Vérifier si la profondeur le permet
            should_recurse = True
            if max_depth is not None:
                should_recurse = (max_depth > self.level).any().item()

            if should_recurse:
                # Instanciation paresseuse des enfants (sur CPU)
                self._init_children()

                # Calculer les gains par enfant (routeur MoE)
                _, gains, active_mask = self.child_gate(x_in, z_corrected)

                # ── Garde-fou anti-explosion (defense en profondeur) ────
                # Le plafond dur vit normalement dans FractalGate (topk +
                # seuil secondaire, voir gating.py) - ce garde-fou ne
                # DEVRAIT jamais se declencher. Il protege contre un bug
                # futur qui contournerait FractalGate (nouveau chemin de
                # code, regression). Si le nombre d'enfants actifs
                # depasse le plafond configure, troncature IMMEDIATE aux
                # meilleurs gains - le(s) enfant(s) en trop ne sont
                # JAMAIS deplaces vers le GPU (rejet direct vers CPU,
                # avant le moindre cout memoire). Correction synchrone,
                # dans la meme passe forward - pas de pause multi-thread
                # necessaire ici (pas de calcul concurrent a coordonner
                # a ce point precis du code).
                max_actifs = self.config.max_active_children
                n_actifs_bruts = int(active_mask.sum().item())
                if n_actifs_bruts > max_actifs:
                    print(
                        f"[GARDE-FOU GATING] Noeud niveau {self.level} : "
                        f"{n_actifs_bruts} enfants actifs detectes (max "
                        f"autorise {max_actifs}) - troncature aux "
                        f"{max_actifs} meilleurs gains, le(s) surplus "
                        f"reste(nt) sur CPU, jamais charges."
                    )
                    ordre = torch.argsort(gains.mean(dim=0), descending=True)
                    active_mask = torch.zeros_like(active_mask)
                    active_mask[ordre[:max_actifs]] = True

                # Forward récursif SEULEMENT sur les enfants actifs
                child_outputs = []
                for i in range(self.n_children):
                    if active_mask[i].item():
                        child_out, child_res, child_h, child_info = \
                            self._forward_child_offloaded(
                                self._children[i], device,
                                z_corrected, max_depth, mask,
                            )
                        # Pondérer par le gain G_{n,i}
                        gain_i = gains[:, i].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                        child_outputs.append(child_out * gain_i)
                        n_active_children += 1
                        children_info.append(child_info)
                    else:
                        # Enfant en sommeil : reste sur CPU, contribution nulle
                        child_outputs.append(torch.zeros(B, T, D, device=device))

                # Agrégation des sorties enfants
                all_children = torch.cat(child_outputs, dim=-1)     # (B, T, D*M)
                children_residual = self.child_aggregator(all_children)  # (B, T, D)

        # ── 6. Mise à jour de l'état caché ─────────────────────────────
        # H^{t+1} = LayerNorm(H^t + A + R)
        hidden_state = self.hidden_state_mgr.update(
            hidden_state, activation, children_residual
        )

        # Stockage en mémoire court terme
        self.memory.store(hidden_state)

        # ── 7. Sortie et résiduel pour le parent ───────────────────────
        output = z_corrected + children_residual
        residual = self.residual_proj(output)

        # ── 8. Métriques ───────────────────────────────────────────────
        info = {
            "level": self.level,
            "coherence": h_check.mean().item(),
            "confidence": q.mean().item(),
            "active_children": n_active_children,
            "children_info": children_info,
        }

        return output, residual, hidden_state, info
