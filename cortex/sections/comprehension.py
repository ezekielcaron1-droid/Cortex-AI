"""
cortex/sections/comprehension.py
Section CO : La Compréhension

Comprend et analyse les données traduites. Utilise un traitement 3D
multi-plans (MultiPlaneProcessor) pour enrichir la représentation,
puis extrait l'intention et le contexte sémantique.

Pipeline interne :
    traduit (B, T, D)
        → MultiPlaneProcessor (traitement 3D sur 4 plans parallèles)
        → TransformerDecoderBlock (analyse profonde)
        → IntentExtractor  (vecteur d'intention condensé)
        → ContextBuilder   (contexte sémantique enrichi)
        → dict {'meaning', 'intent', 'context'}
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig
from cortex.core.transformer import TransformerDecoderBlock


class MultiPlaneProcessor(nn.Module):
    """Traitement multi-dimensionnel 3D.

    Divise l'espace d'embedding en `n_planes` plans parallèles,
    traite chaque plan indépendamment, puis fusionne via une
    attention croisée inter-plans.

    Cela permet de capturer des « vues » complémentaires de
    l'information dans un espace compact.
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.n_planes = config.n_planes       # 4
        self.d_plane = config.d_plane          # 128 (= 512 // 4)
        d = config.d_embed
        dropout = config.dropout

        # --- Processeur indépendant par plan (FFN avec résiduel) ---
        self.plane_processors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.d_plane),
                nn.Linear(self.d_plane, self.d_plane * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_plane * 2, self.d_plane),
            )
            for _ in range(self.n_planes)
        ])

        # --- Attention croisée inter-plans (mélange les 4 plans) ---
        self.cross_plane_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, D)
        Returns:
            out : (B, T, D) — représentation enrichie multi-plans
        """
        # 1. Diviser en plans : chaque chunk a la forme (B, T, d_plane)
        plans = list(torch.chunk(x, self.n_planes, dim=-1))

        # 2. Traiter chaque plan + connexion résiduelle
        for i, processeur in enumerate(self.plane_processors):
            plans[i] = plans[i] + processeur(plans[i])

        # 3. Recombiner les plans → (B, T, D)
        combined = torch.cat(plans, dim=-1)

        # 4. Attention croisée inter-plans (Query = Key = Value = combined)
        attn_out, _ = self.cross_plane_attn(combined, combined, combined)

        # 5. Résiduel + normalisation
        out = self.out_norm(combined + attn_out)
        return out


class Comprehension(nn.Module):
    """Section CO — Compréhension et analyse sémantique.

    Étapes :
        1. Traitement 3D multi-plans (MultiPlaneProcessor)
        2. Analyse profonde (TransformerDecoderBlock)
        3. Extraction de l'intention (MLP sur le vecteur poolé)
        4. Construction du contexte sémantique (MLP + LayerNorm)
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        d = config.d_embed

        # 1. Traitement 3D multi-plans
        self.multi_plane = MultiPlaneProcessor(config)

        # 2. Analyse profonde (transformer décodeur pré-existant)
        self.analyzer = TransformerDecoderBlock(config)

        # 3. Extracteur d'intention (MLP réducteur puis expanseur)
        #    Réduit vers un goulot d_embed//4 puis reprojette à d_embed
        #    afin de forcer une compression sémantique de l'intention.
        self.intent_extractor = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, d),
        )

        # 4. Constructeur de contexte (MLP + normalisation)
        self.context_builder = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, translated: torch.Tensor) -> dict:
        """
        Args:
            translated : (B, T, D) — sortie du TraducteurEntree
        Returns:
            dict avec :
                'meaning' : (B, T, D) — représentation sémantique analysée
                'intent'  : (B, D)    — vecteur d'intention condensé
                'context' : (B, T, D) — contexte sémantique enrichi
        """
        # 1. Traitement 3D multi-plans
        x = self.multi_plane(translated)          # (B, T, D)

        # 2. Analyse profonde avec le transformer
        meaning = self.analyzer(x)                # (B, T, D)

        # 3. Extraction de l'intention
        #    Pooling temporel (moyenne sur T) → vecteur global (B, D)
        pooled = meaning.mean(dim=1)              # (B, D)
        intent = self.intent_extractor(pooled)    # (B, D)

        # 4. Construction du contexte sémantique
        context = self.context_builder(meaning)   # (B, T, D)

        return {
            'meaning': meaning,
            'intent': intent,
            'context': context,
        }
