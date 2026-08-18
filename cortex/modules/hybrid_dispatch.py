"""
CORTEX — Dispatcher hybride CPU ↔ GPU.

Sépare intelligemment le traitement entre CPU et GPU en fonction
de la complexité des données :

    ┌─────────────────────────────────────────────────┐
    │              HYBRID DISPATCHER                  │
    │                                                 │
    │  Entrée (batch de séquences)                    │
    │         │                                       │
    │         ├─ Cx faible ──→ CPU (chemin court)     │
    │         │                  Niveaux 0-1 max      │
    │         │                  Rapide, économique    │
    │         │                                       │
    │         └─ Cx élevé ──→ GPU (chemin profond)    │
    │                           Niveaux 0-5           │
    │                           Puissant, complet     │
    │         │                                       │
    │         ├─────── Fusion des résultats ──────────│
    │         ▼                                       │
    │  Sortie (batch unifié)                          │
    └─────────────────────────────────────────────────┘

Avantage pour l'entraînement :
    - Les tokens simples / données légères ne gaspillent pas
      de cycles GPU → le GPU est libre pour les calculs lourds.
    - Le CPU traite en parallèle les données simples.
    - Réduction significative de la charge GPU totale.
"""

import torch
import torch.nn as nn

from cortex.config import CortexConfig


class HybridDispatcher(nn.Module):
    """Dispatch hybride CPU ↔ GPU basé sur la complexité.

    Analyse chaque séquence du batch et décide si elle doit
    être traitée sur CPU (données simples) ou GPU (données complexes).

    Seuils configurables :
        - cpu_threshold : Cx moyen en dessous duquel → CPU
        - gpu_device    : device GPU cible
    """

    def __init__(self, config: CortexConfig):
        super().__init__()
        self.config = config

        # Seuil de complexité pour le dispatch CPU/GPU
        # En dessous → CPU, au dessus → GPU
        self.cpu_threshold = config.hybrid_cpu_threshold

        # Profondeur max autorisée sur CPU (chemin court)
        self.cpu_max_depth = config.hybrid_cpu_max_depth

    def analyze_batch(
        self, cx_scores: torch.Tensor
    ) -> dict:
        """Analyse le batch et sépare en groupes CPU / GPU.

        Args:
            cx_scores: (B, T) — scores de complexité par token

        Returns:
            dict contenant :
                "cpu_indices": indices des séquences pour CPU
                "gpu_indices": indices des séquences pour GPU
                "avg_cx":      (B,) complexité moyenne par séquence
        """
        # Complexité moyenne par séquence
        avg_cx = cx_scores.mean(dim=-1)  # (B,)

        # Séparer les indices
        cpu_mask = avg_cx < self.cpu_threshold
        gpu_mask = ~cpu_mask

        cpu_indices = torch.where(cpu_mask)[0]
        gpu_indices = torch.where(gpu_mask)[0]

        return {
            "cpu_indices": cpu_indices,
            "gpu_indices": gpu_indices,
            "avg_cx": avg_cx,
            "cpu_count": cpu_indices.numel(),
            "gpu_count": gpu_indices.numel(),
        }

    def split_batch(
        self, tensor: torch.Tensor, dispatch_info: dict
    ) -> tuple:
        """Sépare un tenseur en portions CPU et GPU.

        Args:
            tensor: (B, ...) — tenseur à séparer
            dispatch_info: résultat de analyze_batch

        Returns:
            (cpu_tensor, gpu_tensor) — portions CPU et GPU
                cpu_tensor est déplacé sur CPU
                gpu_tensor reste sur le device original
        """
        cpu_idx = dispatch_info["cpu_indices"]
        gpu_idx = dispatch_info["gpu_indices"]

        cpu_tensor = None
        gpu_tensor = None

        if cpu_idx.numel() > 0:
            cpu_tensor = tensor[cpu_idx].to("cpu")

        if gpu_idx.numel() > 0:
            gpu_tensor = tensor[gpu_idx]  # reste sur GPU

        return cpu_tensor, gpu_tensor

    def merge_results(
        self,
        cpu_result: torch.Tensor | None,
        gpu_result: torch.Tensor | None,
        dispatch_info: dict,
        target_device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Fusionne les résultats CPU et GPU en un seul tenseur.

        Args:
            cpu_result: résultat du traitement CPU (ou None)
            gpu_result: résultat du traitement GPU (ou None)
            dispatch_info: résultat de analyze_batch
            target_device: device cible pour le résultat fusionné
            batch_size: taille originale du batch

        Returns:
            (B, ...) — tenseur unifié sur target_device
        """
        cpu_idx = dispatch_info["cpu_indices"]
        gpu_idx = dispatch_info["gpu_indices"]

        # Déterminer la forme de sortie
        if gpu_result is not None:
            shape = (batch_size,) + gpu_result.shape[1:]
            dtype = gpu_result.dtype
        elif cpu_result is not None:
            shape = (batch_size,) + cpu_result.shape[1:]
            dtype = cpu_result.dtype
        else:
            raise ValueError("Les deux résultats sont None")

        # Créer le tenseur fusionné
        merged = torch.zeros(shape, dtype=dtype, device=target_device)

        if cpu_result is not None and cpu_idx.numel() > 0:
            merged[cpu_idx] = cpu_result.to(target_device)

        if gpu_result is not None and gpu_idx.numel() > 0:
            merged[gpu_idx] = gpu_result.to(target_device)

        return merged

    def get_dispatch_summary(self, dispatch_info: dict) -> str:
        """Retourne un résumé textuel du dispatch."""
        total = dispatch_info["cpu_count"] + dispatch_info["gpu_count"]
        cpu_pct = (dispatch_info["cpu_count"] / max(total, 1)) * 100
        gpu_pct = (dispatch_info["gpu_count"] / max(total, 1)) * 100
        avg_cx = dispatch_info["avg_cx"]

        return (
            f"Dispatch: {dispatch_info['cpu_count']} séquences → CPU ({cpu_pct:.0f}%), "
            f"{dispatch_info['gpu_count']} séquences → GPU ({gpu_pct:.0f}%) | "
            f"Cx moyen: {avg_cx.mean().item():.3f}"
        )
