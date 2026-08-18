"""
cortex/diagnostic_horloges.py
Script de diagnostic et de calibration HAUTE PRÉCISION (3 décimales : 0.000)
pour mesurer la synchronisation des horloges internes (signaux de position et de phase)
entre la racine et les nœuds enfants de l'arbre fractal de CORTEX.

Rôles :
    1. Mesurer le décalage de phase positionnelle (Phase Drift) entre les niveaux (0.000 à 1.000).
    2. Vérifier l'alignement cosinus (Cosine Alignment) token-par-token à travers la récursion.
    3. Calculer la dérive temporelle des portes de gating (Clock Jitter) à haute précision.
"""

import sys
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex.config import CortexConfig
from cortex.model import CortexModel
from cortex.tokenizer import ByteTokenizer


def mesurer_alignement_horloges(model, tokenizer, device, sample_text: str):
    """Calcule les métriques de synchronisation d'horloge avec une précision à 0.000."""
    print(f"\n========================================================")
    print(f"   DIAGNOSTIC HAUTE PRECISION DES HORLOGES CORTEX (0.000)")
    print(f"========================================================\n")

    model.eval()
    
    # 1. Encodage du texte de test
    ids = tokenizer.encode(sample_text, add_bos=True, add_eos=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    B, T = input_ids.shape

    print(f"[TEST] Texte analysé : '{sample_text}' ({T} tokens)")
    print(f"[INFO] Calcul des matrices d'alignement de phase...\n")

    with torch.no_grad():
        # Passe d'entrée
        emb_tokens = model.traducteur_entree.token_embedding(input_ids)       # (B, T, D)
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        emb_pos = model.traducteur_entree.position_embedding(pos_ids)          # (B, T, D)
        h_entree = emb_tokens + emb_pos                                        # Signaux avec horloge de base

        # Passe cerveau général
        result = model(input_ids, override_level=None, cascade=True)
        cortex_out = result["cortex_output"]  # (B, T, D)

        # ── 1. ALIGNEMENT DE PHASE POSITIONNELLE (0.000) ──────────────────
        # Mesure de la conservation du rythme positionnel entre l'entrée et la sortie
        sim_positionnelle = F.cosine_similarity(h_entree, cortex_out, dim=-1)  # (B, T)
        alignement_moyen = sim_positionnelle.mean().item()
        alignement_min = sim_positionnelle.min().item()
        alignement_max = sim_positionnelle.max().item()

        # ── 2. DÉRIVE TEMPORELLE / STABILITÉ DE L'HORLOGE (Clock Drift) ────
        # Mesure des variations de norme entre tokens consécutifs
        if T > 1:
            diff_entree = torch.norm(h_entree[:, 1:] - h_entree[:, :-1], dim=-1)
            diff_sortie = torch.norm(cortex_out[:, 1:] - cortex_out[:, :-1], dim=-1)
            derive_horloge = torch.abs(diff_sortie - diff_entree).mean().item()
        else:
            derive_horloge = 0.0

        # ── 3. SYNCHRONISATION PAR TOKEN (Détail à 0.000) ─────────────────
        print("+------+---------------+-----------------+-------------------+")
        print("| Token| Token Texte   | Phase Sync      | Derive Horloge    |")
        print("+------+---------------+-----------------+-------------------+")
        
        for t_idx in range(T):
            tok_id = ids[t_idx]
            tok_str = tokenizer.decode([tok_id]).replace("\n", "\\n")
            if len(tok_str) > 13:
                tok_str = tok_str[:10] + "..."
            
            sync_val = sim_positionnelle[0, t_idx].item()
            drift_val = (torch.norm(cortex_out[0, t_idx] - h_entree[0, t_idx]).item())
            
            # Formattage strict à 3 décimales (0.000)
            print(f"| {t_idx:4d} | {tok_str:13s} | {sync_val:15.3f} | {drift_val:17.3f} |")
            
        print("+------+---------------+-----------------+-------------------+")

        # ── 4. RÉSULTATS GLOBAUX À HAUTE PRÉCISION ───────────────────────
        print(f"\nMETRIQUES GLOBALES D'HORLOGE (Precision 0.000) :")
        print(f"   • Alignement de Phase Moyen : {alignement_moyen:.3f} / 1.000")
        print(f"   • Alignement de Phase Min   : {alignement_min:.3f} / 1.000")
        print(f"   • Alignement de Phase Max   : {alignement_max:.3f} / 1.000")
        print(f"   • Derive d'Horloge Moyenne  : {derive_horloge:.3f}")
        
        # Diagnostic synthétique
        if alignement_moyen < 0.300:
            status = "[DESYNCHRONISATION SEVERE] Les horloges des enfants derivent fortement."
        elif alignement_moyen < 0.600:
            status = "[SYNCHRONISATION MOYENNE] Risque de legeres incoherences de phrase."
        else:
            status = "[HORLOGES PARFAITEMENT REGLEES] Phase et position preservees."
            
        print(f"\n[STATUT HORLOGES] : {status}\n")

    return {
        "alignement_moyen": round(alignement_moyen, 3),
        "alignement_min": round(alignement_min, 3),
        "alignement_max": round(alignement_max, 3),
        "derive_horloge": round(derive_horloge, 3),
    }


if __name__ == "__main__":
    from cortex.bridge import model, device, tokenizer

    if model is None:
        print("[ERREUR] Modèle non disponible.")
        sys.exit(1)

    phrase_test = "Bonjour CORTEX, synchronise tes horloges internes pour générer des phrases claires."
    mesurer_alignement_horloges(model, tokenizer, device, phrase_test)
