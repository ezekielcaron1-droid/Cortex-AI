"""
cortex/sections/traducteur.py
Section T : Le Traducteur

Porte d'entrée et de sortie du modèle. Simplifie le langage humain pour
le rendre digeste pour le "cerveau" interne, tout en retenant la langue d'origine.
"""

import torch
import torch.nn as nn


class TraducteurEntree(nn.Module):
    """
    Reçoit les identifiants de tokens (B, T) et produit :
      - Une représentation simplifiée (B, T, D) pour le cerveau.
      - Les probabilités de langue (B, num_langues).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_seq_len: int = 1024,
        num_langues: int = 3,
        n_heads: int = 8,
        ff_mult: int = 4,
        n_traducteurs: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_langues = num_langues
        self.n_traducteurs = n_traducteurs

        # --- Étape 1 : Embedding + Position ---
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # --- Étape 2 : Détecteur de langue (MLP 2 couches -> Softmax) ---
        self.detecteur_langue = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_langues),
        )

        # --- Étape 3 : Traducteurs parallèles (x3 blocs Transformer) ---
        self.traducteurs = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * ff_mult,
                batch_first=True,
            )
            for _ in range(n_traducteurs)
        ])

        # --- Étape 4 : Fusion (concat 3*D -> projection D + LayerNorm) ---
        self.projection_fusion = nn.Linear(d_model * n_traducteurs, d_model)
        self.norm_fusion = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor):
        """
        Args:
            token_ids : (B, T) entiers
        Returns:
            traduit        : (B, T, D)
            probas_langue  : (B, num_langues)
        """
        B, T = token_ids.shape

        # --- Étape 1 : Embedding & Position ---
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)  # (1, T)
        embeddings = self.token_embedding(token_ids) + self.position_embedding(positions)
        # embeddings : (B, T, D)

        # --- Étape 2 : Détection de la langue ---
        vecteur_moyen = embeddings.mean(dim=1)                 # (B, D)
        logits_langue = self.detecteur_langue(vecteur_moyen)   # (B, num_langues)
        probas_langue = torch.softmax(logits_langue, dim=-1)   # (B, num_langues)

        # --- Étape 3 : Traducteurs parallèles ---
        sorties = [bloc(embeddings) for bloc in self.traducteurs]  # liste de (B, T, D)

        # --- Étape 4 : Fusion ---
        concat = torch.cat(sorties, dim=-1)          # (B, T, 3*D)
        projete = self.projection_fusion(concat)     # (B, T, D)
        traduit = self.norm_fusion(projete)          # (B, T, D)

        return traduit, probas_langue


class TraducteurSortie(nn.Module):
    """
    Chemin inverse : prend la représentation traitée par le cerveau (B, T, D)
    et les probabilités de langue (B, num_langues) pour reconstruire des logits
    de vocabulaire (B, T, vocab_size).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_langues: int = 3,
        n_heads: int = 8,
        ff_mult: int = 4,
        n_decoder_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        # Adaptation de la langue : (B, num_langues) -> (B, D)
        self.adaptation_langue = nn.Linear(num_langues, d_model)

        # Décodeur Transformer pour reconstruire la structure grammaticale
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            batch_first=True,
        )
        self.decodeur = nn.TransformerEncoder(decoder_layer, num_layers=n_decoder_layers)

        # Projection finale vers le vocabulaire
        self.projection_vocab = nn.Linear(d_model, vocab_size)

    def forward(self, repr_cerveau: torch.Tensor, probas_langue: torch.Tensor):
        """
        Args:
            repr_cerveau  : (B, T, D)
            probas_langue : (B, num_langues)
        Returns:
            logits : (B, T, vocab_size)
        """
        B, T, D = repr_cerveau.shape

        # --- Adaptation de la langue ---
        contexte_langue = self.adaptation_langue(probas_langue)   # (B, D)
        contexte_langue = contexte_langue.unsqueeze(1).expand(B, T, D)  # (B, T, D)

        # --- Fusion & Décodage ---
        repr_finale = repr_cerveau + contexte_langue   # (B, T, D)
        decode = self.decodeur(repr_finale)            # (B, T, D)

        # --- Projection vers le vocabulaire ---
        logits = self.projection_vocab(decode)         # (B, T, vocab_size)

        return logits
