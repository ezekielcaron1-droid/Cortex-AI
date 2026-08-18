"""
CORTEX v2 — Configuration globale du modèle fractal.

Variables internes clés (section 4.2 du document) :
    N = 5   : nombre de niveaux fractals
    M = 5   : nombre de modèles par niveau
    T = 3125 : nombre total de modèles imbriqués (5^5)

Pipeline :
    Prompt → T(in) → [CRVG: CO → RF → I → CA ⟲ → E] → T(out) → Sortie
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CortexConfig:
    """Hyperparamètres du modèle CORTEX."""

    # ── Structure fractale ──────────────────────────────────────────────
    n_levels: int = 5           # N : nombre de niveaux fractals
    n_models: int = 5           # M : modèles par niveau
    # T (total) = n_models ** n_levels = 3 125

    # ── Dimensions du transformer ───────────────────────────────────────
    d_embed: int = 512          # D_embed : dimension de l'embedding
    n_heads: int = 8            # H : nombre de têtes d'attention
    d_hidden: int = 2048        # D_hidden : taille feed-forward
    n_decoder_layers: int = 1   # Couches transformer par noeud fractal

    # ── Vocabulaire ─────────────────────────────────────────────────────
    vocab_size: int = 32_005  # aligne exactement sur CamemBERT (32005 tokens reels)
    max_seq_len: int = 512

    # ── Multi-dimensionnel 3D ───────────────────────────────────────────
    # Nombre de plans dimensionnels parallèles pour les imbrications 3D.
    # Chaque plan traite une "vue" de la donnée (d_plane = d_embed // n_planes).
    # Plus de plans = plus de capacité dans un espace compact.
    n_planes: int = 4

    # ── Routing intelligent (Cx) ────────────────────────────────────────
    complexity_thresholds: List[float] = field(
        default_factory=lambda: [0.15, 0.35, 0.55, 0.75]
    )

    # ── Gating / activation dynamique ───────────────────────────────────
    sleep_threshold: float = 0.1

    # ── Garde-fous d'activation MoE (anti-explosion combinatoire) ───────
    # Sans plafond dur, un gate a peine entraine (scores quasi uniformes,
    # cf. CIBLE_EQUILIBRAGE=0.2) peut activer plus d'enfants que prevu a
    # chaque noeud. Sur un arbre a 5 niveaux, un leger depassement se
    # multiplie en cascade (effet combinatoire) - confirme empiriquement
    # par diagnostic_activation_oom.py : plus d'1 milliard de valeurs
    # actives simultanement, crash CUDA OOM. Voir gating.py pour le detail
    # du calcul (pire cas cap=3 sur 5 niveaux : 363 noeuds actifs max,
    # ~102M valeurs, marge x2 sur 6 Go de VRAM).
    max_active_children: int = 2      # plafond dur d'enfants actifs par noeud
    # NOTE (voir diagnostic_activation_oom.py) : cap=3 a ete teste et a
    # quand meme fini par crasher (OOM) apres 31 pas - le cout reel par
    # noeud (~4,4M valeurs actives en moyenne) est bien plus eleve que
    # l'estimation initiale (~280K), rendant le pire cas theorique a
    # cap=3 (363 noeuds x 4,4M = ~1,6 milliard de valeurs) plus risque
    # que prevu. Redescendu a 2 (pire cas 62 noeuds x 4,4M = ~272M
    # valeurs) par prudence -- a tenu ~850 pas de plus (jusqu'au pas
    # 46850) avant de re-crasher (OOM), l'arbre continuant de grossir
    # au fil de l'entrainement. Redescendu a 1 (pire cas = un seul
    # chemin racine-feuille, 5 noeuds x 4,4M = ~22M valeurs) : marge
    # bien plus large, au prix d'un routage moins riche par pas -- a
    # tenu sans probleme sur les 3500 derniers pas (46500->50000).
    #
    # REMONTE A 2 pour la reprise sur RTX 5060 (8 Go, marge un peu
    # plus large qu'a 6 Go) -- COMBINE avec gate_ratio_secondaire
    # releve (0.55 -> 0.78, voir plus bas) pour rendre le 2e enfant
    # actif nettement plus rare qu'avant, sans reproduire le regime
    # ou il s'activait facilement. Attention : le PIRE CAS exact (62
    # noeuds actifs) reste geometriquement possible et identique a
    # celui qui a crashe deux fois sur 6 Go -- seule sa PROBABILITE
    # baisse, pas sa gravite si jamais il se reproduit. A surveiller
    # avec nvidia-smi sur un vrai volume de pas, pas juste quelques
    # dizaines, avant de considerer ce reglage valide sur la duree.
    gate_ratio_secondaire: float = 0.78  # ADAPTATIF : fraction du meilleur
    # gain du noeud (pas une valeur absolue). Remplace l'ancien seuil fixe
    # (gate_secondaire_seuil), qui devenait obsolete au fil de l'entrainement :
    # les gains absolus s'ecrasent naturellement avec la specialisation du
    # modele (observe : ~0.04-0.11 a 20k pas, ~0.005-0.02 a 40k pas, soit
    # une chute x5-10), alors que le RATIO entre le meilleur et le 2e
    # meilleur gain reste stable (~0.5-0.6 dans les deux cas mesures).
    # Avec ce seuil relatif, plus besoin de recalibrer manuellement a
    # chaque nouvelle tranche d'entrainement.
    #
    # RELEVE de 0.55 a 0.78 en meme temps que max_active_children
    # remonte a 2 (voir plus haut). ATTENTION SENS : seuil_relatif =
    # meilleur_gain * gate_ratio_secondaire, et le rang 2 s'active si
    # son score DEPASSE ce seuil (gating.py) -- donc un ratio PLUS
    # HAUT rend le seuil plus dur a atteindre, donc le 2e enfant plus
    # RARE (pas l'inverse -- premiere version de ce commentaire s'etait
    # trompee de sens, corrigee ici). A 0.78, le 2e enfant doit avoir
    # un score a moins de 22% du meilleur pour s'activer, contre 45%
    # de marge avant (0.55) -- nettement plus strict.

    # ── Mémoire ─────────────────────────────────────────────────────────
    memory_capacity: int = 64

    # ── Boucle de rétroaction fractale ──────────────────────────────────
    n_feedback_iterations: int = 3

    # ── Section CA — Comparaison (feedback loop) ────────────────────────
    max_feedback_loops: int = 3       # Max retours CA avant passage forcé
    comparison_threshold: float = 0.7  # Seuil de similarité CA (en dessous → retour)

    # ── Section E — Évaluation ──────────────────────────────────────────
    evaluation_threshold: float = 0.8  # Seuil de score E acceptable

    # ── Modules cerveau ─────────────────────────────────────────────────
    brain_activation_threshold: float = 0.5

    # ── Traducteur ──────────────────────────────────────────────────────
    n_languages: int = 32         # Nombre de langues détectables
    n_translators: int = 3        # Nombre de traducteurs parallèles

    # ── Régularisation ──────────────────────────────────────────────────
    dropout: float = 0.1

    # ── Dispatch hybride CPU / GPU ──────────────────────────────────────
    hybrid_cpu_threshold: float = 0.3
    hybrid_cpu_max_depth: int = 2
    hybrid_enabled: bool = True

    # ── Propriétés calculées ────────────────────────────────────────────
    @property
    def total_models(self) -> int:
        """Nombre total de modèles imbriqués (T = M^N)."""
        return self.n_models ** self.n_levels

    @property
    def head_dim(self) -> int:
        """Dimension par tête d'attention."""
        return self.d_embed // self.n_heads

    @property
    def d_plane(self) -> int:
        """Dimension par plan 3D."""
        return self.d_embed // self.n_planes

    def __post_init__(self):
        assert self.d_embed % self.n_heads == 0, (
            f"d_embed ({self.d_embed}) doit être divisible par n_heads ({self.n_heads})"
        )
        assert self.d_embed % self.n_planes == 0, (
            f"d_embed ({self.d_embed}) doit être divisible par n_planes ({self.n_planes})"
        )
