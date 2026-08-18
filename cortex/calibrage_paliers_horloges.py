"""
cortex/calibrage_paliers_horloges.py
Calibration progressive de CORTEX par PALIERS STRICTS DE 5 MILLIONS DE PARAMETRES.

PASSE 1 — Parametres statiques de base (77M)
    Decoupe les 77M en paliers de ~5M. Pour chaque palier k :
        1. Gele tous les paliers precedents.
        2. Calibre le palier courant (5 micro-pas AdamW).
        3. Valide la coherence cumulative (Loss + Phase Sync a 0.000).

PASSE 2 — Enfants fractals (Niveaux 1 a N)
    Force l'instanciation paresseuse de chaque niveau fractal
    (_init_children), deplace les enfants sur GPU, et applique
    la meme logique de paliers de 5M niveau par niveau.
    Apres chaque niveau, decharge les enfants (decharger_enfants_gpu)
    pour ne pas exploser la VRAM.

PASSE 3 — Validation de coherence inter-niveaux
    Verifie que l'horloge du niveau le plus profond reste synchronisee
    avec la racine (Niveau 0) apres traversee complete.
"""

import sys
import os
import gc
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_BLOCK_SIZE = 5_000_000  # 5 Millions de parametres par palier


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def decouper_en_paliers(param_list):
    """Decoupe une liste de (name, param) en blocs de ~5M parametres."""
    paliers = []
    current_block = []
    current_count = 0

    for name, param in param_list:
        if not param.requires_grad:
            continue
        numel = param.numel()
        current_block.append((name, param))
        current_count += numel

        if current_count >= TARGET_BLOCK_SIZE:
            paliers.append((current_block, current_count))
            current_block = []
            current_count = 0

    if current_block:
        paliers.append((current_block, current_count))

    return paliers


def mesurer_phase_sync(model, tokenizer, device, input_ids):
    """Retourne l'alignement de phase moyen entre entree et sortie cortex (0.000)."""
    T = input_ids.shape[1]
    with torch.no_grad():
        emb_tokens = model.traducteur_entree.token_embedding(input_ids)
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        emb_pos = model.traducteur_entree.position_embedding(pos_ids)
        h_entree = emb_tokens + emb_pos
        result = model(input_ids, override_level=None, cascade=True)
        cortex_out = result["cortex_output"]
        sync = F.cosine_similarity(h_entree, cortex_out, dim=-1).mean().item()
    return round(sync, 3)


def calibrer_bloc(block_params, all_params, model, tokenizer, device,
                   input_ids, palier_num, total_paliers, count, nb_micro_pas=5,
                   target_sync=0.050, max_iterations=10):
    """Gele tout, degage ce bloc, calibre de manière adaptative jusqu'à atteindre le target_sync.
    
    Args:
        target_sync: Seuil cible de Phase Sync (défaut: 0.050)
        max_iterations: Nombre max de cycles de recalibrage (défaut: 10)
    """
    # 1. Geler tous les parametres
    for p in all_params:
        p.requires_grad = False

    # 2. Degeler uniquement ce bloc
    for _, param in block_params:
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for _, p in block_params], lr=1e-4
    )

    loss_val = 0.0
    T = input_ids.shape[1]
    
    # Boucle de recalibrage adaptatif
    iteration = 0
    sync_val = -999.0  # Valeur initiale très basse
    
    print(f"   [CIBLE] Phase Sync cible: {target_sync:+.3f}")
    
    while iteration < max_iterations:
        iteration += 1
        print(f"   [ITERATION {iteration}/{max_iterations}]")
        
        # Faire nb_micro_pas micro-pas
        for step in range(nb_micro_pas):
            optimizer.zero_grad()
            res = model(input_ids, override_level=None, cascade=True)
            logits = res["logits"]
            cortex_out = res["cortex_output"]

            # Perte token suivant
            shift_logits = logits[:, :-1, :tokenizer.vocab_size].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, tokenizer.vocab_size),
                shift_labels.view(-1)
            )

            # Alignement de phase positionnel (horloge)
            emb_t = model.traducteur_entree.token_embedding(input_ids)
            pos_ids = torch.arange(T, device=device).unsqueeze(0)
            emb_p = model.traducteur_entree.position_embedding(pos_ids)
            h_entree = emb_t + emb_p
            sync_loss = 1.0 - F.cosine_similarity(h_entree, cortex_out, dim=-1).mean()
            total_loss = loss + 0.1 * sync_loss

            total_loss.backward()
            optimizer.step()
            loss_val = loss.item()

        # Decharger les enfants apres backward
        if hasattr(model, "cerveau"):
            if hasattr(model.cerveau, "reflexion") and hasattr(model.cerveau.reflexion, "fractal_root"):
                model.cerveau.reflexion.fractal_root.decharger_enfants_gpu()

        # Mesurer le Phase Sync après cette itération
        sync_val = mesurer_phase_sync(model, tokenizer, device, input_ids)
        
        # Vérifier si on a atteint la cible
        if sync_val >= target_sync:
            status = "CIBLE ATTEINTE ✓"
            print(f"   --> Phase Sync: {sync_val:+.3f} | Loss: {loss_val:.3f} | {status}")
            break
        else:
            remaining = target_sync - sync_val
            status = f"ENCORE {remaining:+.3f} points"
            print(f"   --> Phase Sync: {sync_val:+.3f} | Loss: {loss_val:.3f} | {status}")
            
            # Si c'est la dernière itération, on marque comme non atteint
            if iteration == max_iterations:
                status = "MAX ITERATIONS - CIBLE NON ATTEINTE"
                print(f"   [WARNING] Cible {target_sync:+.3f} non atteinte après {max_iterations} itérations")

    # Statut final
    if sync_val >= target_sync:
        final_status = "OK (CIBLE ATTEINTE)"
    elif sync_val > -0.100:
        final_status = "ACCEPTABLE (CIBLE NON ATTEINTE)"
    else:
        final_status = "DEVIATION"

    return sync_val, loss_val, final_status


def calibrer_liste_params(param_list, label, model, tokenizer, device, input_ids, 
                          nb_micro_pas=5, target_sync=0.050, max_iterations=10):
    """Applique la calibration par paliers de 5M sur une liste de parametres.
    
    Args:
        target_sync: Seuil cible de Phase Sync pour chaque palier
        max_iterations: Nombre max de cycles de recalibrage par palier
    """
    paliers = decouper_en_paliers(param_list)
    all_params = [p for _, p in param_list]
    results = []
    n = len(paliers)
    for k, (block, count) in enumerate(paliers, start=1):
        print(f"[{label} - PALIER {k:02d}/{n:02d}] ~{count / 1e6:.2f}M params...")
        sync_val, loss_val, status = calibrer_bloc(
            block, all_params, model, tokenizer, device, input_ids, k, n, count, 
            nb_micro_pas, target_sync, max_iterations
        )
        results.append({
            "palier": k, "label": label,
            "params_m": round(count / 1e6, 2),
            "sync": sync_val, "loss": round(loss_val, 3),
            "status": status
        })
    return results


def afficher_tableau(results):
    """Affiche le tableau de synthese en ASCII (compatible cp1252)."""
    print("+--------+-------------------------+------------+------------+-----------+--------------+")
    print("| Palier | Composant               | Taille (M) | Phase Sync | Loss      | Statut       |")
    print("+--------+-------------------------+------------+------------+-----------+--------------+")
    for r in results:
        label = r["label"][:23]
        print(f"| {r['palier']:6d} | {label:23s} | {r['params_m']:10.2f}M | {r['sync']:10.3f} | {r['loss']:9.3f} | {r['status']:12s} |")
    print("+--------+-------------------------+------------+------------+-----------+--------------+")


def sauvegarder_modele(model, checkpoint_path=None):
    """Sauvegarde le modèle calibré dans le fichier checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "checkpoint.pt"
        )
    
    # S'assurer que le répertoire data existe
    data_dir = os.path.dirname(checkpoint_path)
    os.makedirs(data_dir, exist_ok=True)
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[SAUVEGARDE] Modèle calibré sauvegardé dans {checkpoint_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def calibrer_complet(model, tokenizer, device, sample_texts, nb_micro_pas=5,
                     target_sync=0.050, max_iterations=10):
    """Calibration complete : base statique + enfants fractals + validation.
    
    Args:
        sample_texts: Liste de phrases variées pour un calibrage robuste
        nb_micro_pas: Nombre de micro-pas par palier (5=rapide, 25=précis)
        target_sync: Seuil cible de Phase Sync pour chaque palier (0.050=défaut)
        max_iterations: Nombre max de cycles de recalibrage par palier (10=défaut)
    """

    all_results = []

    # Itérer sur chaque phrase de test
    for phrase_idx, sample_text in enumerate(sample_texts, start=1):
        print(f"\n{'='*60}")
        print(f"   PHRASE {phrase_idx}/{len(sample_texts)} : {sample_text[:60]}...")
        print(f"{'='*60}")
        
        ids = tokenizer.encode(sample_text, add_bos=True, add_eos=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        # ═══════════════════════════════════════════════════════════════════
        # PASSE 1 — Parametres statiques de base (77M)
        # ═══════════════════════════════════════════════════════════════════
        print("\n========================================================")
        print("   PASSE 1 : CALIBRATION BASE STATIQUE (77M PARAMS)")
        print("========================================================")

        base_params = list(model.named_parameters())
        r1 = calibrer_liste_params(base_params, "BASE", model, tokenizer, device, input_ids, 
                                   nb_micro_pas, target_sync, max_iterations)
        all_results.extend(r1)

        sync_apres_passe1 = mesurer_phase_sync(model, tokenizer, device, input_ids)
        print(f"\n[PASSE 1 TERMINEE] Phase Sync globale : {sync_apres_passe1:+.3f}\n")

    # ═══════════════════════════════════════════════════════════════════
    # PASSE 2 — Enfants fractals niveau par niveau
    # ═══════════════════════════════════════════════════════════════════
    print("========================================================")
    print("   PASSE 2 : CALIBRATION ENFANTS FRACTALS (NIVEAUX 1-N)")
    print("========================================================")

    # Recuperer la racine fractale
    fractal_root = None
    try:
        fractal_root = model.cerveau.reflexion.fractal_root
    except AttributeError:
        print("[AVERTISSEMENT] Racine fractale non accessible. Passe 2 ignoree.")

    if fractal_root is not None:
        niveaux_calibres = 0
        node_queue = [(fractal_root, 0)]  # (noeud, niveau)

        while node_queue:
            node, niveau = node_queue.pop(0)

            # Forcer l'instanciation paresseuse des enfants de ce noeud
            if not node._children_initialized:
                node._init_children()

            if len(node._children) == 0:
                continue

            # Collecter les parametres de TOUS les enfants directs de ce noeud
            niveau_params = []
            for i, child in enumerate(node._children):
                # Charger l'enfant sur GPU pour la calibration
                child.to(device)
                for name, param in child.named_parameters():
                    niveau_params.append((f"N{niveau+1}_enfant{i}.{name}", param))

            if niveau_params:
                label = f"FRACTAL N{niveau+1}"
                print(f"\n[NIVEAU {niveau+1}] {len(node._children)} enfants | "
                      f"{sum(p.numel() for _, p in niveau_params) / 1e6:.2f}M params total")

                r2 = calibrer_liste_params(
                    niveau_params, label, model, tokenizer, device, input_ids, 
                    nb_micro_pas, target_sync, max_iterations
                )
                all_results.extend(r2)
                niveaux_calibres += 1

            # Decharger les enfants de ce niveau apres calibration
            for child in node._children:
                child.to("cpu")
            node.decharger_enfants_gpu()
            torch.cuda.empty_cache()
            gc.collect()

            # Ajouter les enfants du niveau suivant dans la queue
            for child in node._children:
                if child.level < child.max_level:
                    node_queue.append((child, niveau + 1))

        sync_apres_passe2 = mesurer_phase_sync(model, tokenizer, device, input_ids)
        print(f"\n[PASSE 2 TERMINEE] {niveaux_calibres} niveaux calibres | "
              f"Phase Sync globale : {sync_apres_passe2:+.3f}\n")

    # ═══════════════════════════════════════════════════════════════════
    # PASSE 3 — Validation de coherence inter-niveaux
    # ═══════════════════════════════════════════════════════════════════
    print("========================================================")
    print("   PASSE 3 : VALIDATION COHERENCE INTER-NIVEAUX")
    print("========================================================")

    sync_final = mesurer_phase_sync(model, tokenizer, device, input_ids)
    sync_passe1 = sync_apres_passe1
    sync_degradation = round(abs(sync_final - sync_passe1), 3)

    print(f"\n   Phase Sync apres Passe 1 (Base)    : {sync_passe1:+.3f}")
    print(f"   Phase Sync finale (Base + Fractals) : {sync_final:+.3f}")
    print(f"   Degradation inter-niveaux           : {sync_degradation:.3f}")

    if sync_degradation < 0.050:
        verdict = "COHERENCE INTER-NIVEAUX PRESERVEE"
    elif sync_degradation < 0.150:
        verdict = "LEGERE DERIVE INTER-NIVEAUX - Acceptable"
    else:
        verdict = "DERIVE SEVERE INTER-NIVEAUX - Repasser Passe 1"

    print(f"\n[VERDICT PASSE 3] : {verdict}\n")

    # ═══════════════════════════════════════════════════════════════════
    # TABLEAU FINAL DE SYNTHESE
    # ═══════════════════════════════════════════════════════════════════
    print("========================================================")
    print("   TABLEAU DE SYNTHESE COMPLET (BASE + FRACTALS)")
    print("========================================================")
    afficher_tableau(all_results)

    total_params_calibres = sum(
        r["params_m"] for r in all_results
    )
    print(f"\n[BILAN FINAL]")
    print(f"   Total parametres calibres : {total_params_calibres:.2f}M")
    print(f"   Phase Sync finale         : {sync_final:+.3f} / 1.000")
    print(f"   Coherence inter-niveaux   : {verdict}")

    # ═══════════════════════════════════════════════════════════════════
    # SAUVEGARDE DU MODELE CALIBRE
    # ═══════════════════════════════════════════════════════════════════
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "checkpoint.pt"
    )
    sauvegarder_modele(model, checkpoint_path)


if __name__ == "__main__":
    from cortex.bridge import model, device, tokenizer

    if model is None:
        print("[ERREUR] Modele non disponible.")
        sys.exit(1)

    # Configuration du calibrage (ajustée pour contrainte 10h max)
    nb_micro_pas = 25  # 25x plus précis que 5, mais 2x moins que 50 (compromis temps/qualité)
    target_sync = 0.050  # Seuil cible de Phase Sync par palier (5% d'alignement)
    max_iterations = 10  # Max de cycles de recalibrage par palier (évite boucles infinies)
    
    # Phrases de test variées pour un calibrage robuste (4 phrases)
    phrases_test = [
        "CORTEX aligne ses horloges paliers par paliers de cinq millions de parametres.",
        "Le cerveau fractal traite l'information en cascade à travers plusieurs niveaux.",
        "La coherence inter-niveaux assure une propagation stable des representations.",
        "L'apprentissage progressif par paliers garantit une convergence stable."
    ]
    
    print(f"[CONFIGURATION]")
    print(f"   Micro-pas par palier: {nb_micro_pas}")
    print(f"   Phase Sync cible: {target_sync:+.3f}")
    print(f"   Max itérations par palier: {max_iterations}")
    print(f"   Phrases de test: {len(phrases_test)}")
    print(f"[ESTIMATION TEMPS] Variable selon convergence (max ~{nb_micro_pas * max_iterations * len(phrases_test)}x version actuelle)")
    
    calibrer_complet(model, tokenizer, device, phrases_test, nb_micro_pas, target_sync, max_iterations)
