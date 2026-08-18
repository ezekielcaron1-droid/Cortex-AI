"""
cortex/bridge.py
Module de liaison entre le serveur (Flask) et le modèle CORTEX.

Batching dynamique + apprentissage continu en tâche de fond.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from cortex.tokenizer import ByteTokenizer

model = None
device = None
model_load_error = None
memoire = None
tokenizer = None

try:
    from cortex.config import CortexConfig
    from cortex.model import CortexModel

    config = CortexConfig()
    # Initialiser le tokenizer avec le vocab_size du modèle pour compatibilité
    tokenizer = ByteTokenizer(max_seq_len=config.max_seq_len, vocab_size=config.vocab_size)
    print(f"[INFO] Tokenizer chargé — vocab_size = {tokenizer.vocab_size} (compatible avec le modèle)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "checkpoint.pt"
    )
    
    if os.path.exists(checkpoint_path):
        # CRITIQUE : Reconstruction de l'arbre fractal AVANT le chargement du checkpoint
        # Sinon, seuls les poids du tronc sont charges (~77M parametres), les branches
        # profondes restent aleatoires a chaque demarrage (>1 milliard perdus).
        #
        # CORRECTIF (le premier essai appelait une methode
        # reconstruire_arbre_complet() qui n'a jamais existe nulle part
        # dans le code -- AttributeError silencieusement avale par le
        # try/except global de ce fichier, laissant tourner un modele
        # aux poids 100% ALEATOIRES sans avertissement visible cote
        # chat, pire que le bug d'origine). On reutilise ici la
        # fonction reellement testee de train.py, qui a besoin du
        # state_dict CHARGE pour savoir quels noeuds reconstruire --
        # d'ou l'ordre : torch.load() d'abord, reconstruction ensuite,
        # load_state_dict() en dernier.
        from train import reconstruire_arbre_depuis_checkpoint, placer_hors_arbre_sur_device

        print(f"[INFO] Reconstruction de l'arbre fractal avant chargement du checkpoint...")
        model = CortexModel(config)
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
        n_noeuds = reconstruire_arbre_depuis_checkpoint(model, checkpoint_dict)
        resultat = model.load_state_dict(checkpoint_dict, strict=False)
        del checkpoint_dict
        placer_hors_arbre_sur_device(model, device)

        print(f"[INFO] Checkpoint entraîné chargé depuis {checkpoint_path}")
        print(f"[INFO] Arbre fractal reconstruit : {n_noeuds} noeud(s) reinities, "
              f"{len(resultat.unexpected_keys)} cle(s) ignoree(s) "
              f"(0 = reconstruction complete).")
        if resultat.missing_keys:
            print(
                f"[INFO] {len(resultat.missing_keys)} poids absents du checkpoint "
                "(probablement les enfants fractals, jamais sauvegardés avant "
                "le correctif nn.ModuleList) — initialisés aléatoirement."
            )
    else:
        print("[INFO] Aucun checkpoint trouvé — poids aléatoires (pas encore entraîné).")
        model = CortexModel(config).to(device)

    model.eval()
    n_params = model.count_parameters()
    print(f"[INFO] Modèle CORTEX chargé sur {device} — {n_params:,} paramètres entraînables")

    from cortex.modules.gestionnaire_memoire import GestionnaireMemoire
    memoire = GestionnaireMemoire()
    print("[INFO] Gestionnaire de mémoire (Index + Vérificateur) chargé")
except Exception as e:
    # Fallback si le modèle ne se charge pas
    tokenizer = ByteTokenizer(max_seq_len=1024)
    print(f"[INFO] Tokenizer chargé (fallback) — vocab_size = {tokenizer.vocab_size}")
    print(f"[ERREUR] Le modèle CORTEX a échoué à se charger : {e}")
    model_load_error = str(e)

    from cortex.modules.rafraichissement_schemas import demarrer as demarrer_rafraichissement_schemas
    demarrer_rafraichissement_schemas()
except ImportError as e:
    model_load_error = str(e)
    print(f"[INFO] Modèle CORTEX pas encore complet ({model_load_error}).")
    print("[INFO] Le bridge tourne en mode placeholder (tokenizer seul).")
except Exception as e:
    model_load_error = str(e)
    print(f"[ERREUR] Le modèle CORTEX a échoué à se charger : {model_load_error}")
    print("[INFO] Le bridge tourne en mode placeholder (tokenizer seul).")


def is_model_ready() -> bool:
    return model is not None


# ══════════════════════════════════════════════════════════════════════
#  Coordination de pause — pour permettre un entrainement de rappel
#  (voir apprentissage_force.py) SANS copier le modele et SANS jamais
#  faire tourner un forward() et un backward() en meme temps dessus.
# ══════════════════════════════════════════════════════════════════════
pret_pour_generation = threading.Event()
pret_pour_generation.set()  # actif par defaut : generation normale autorisee

_generation_compteur_lock = threading.Lock()
_generations_en_cours = 0

# ── Verrou d'exclusion mutuelle sur le modele partage ───────────────────
# BUG CORRIGE : le mode "fast" (cortex/fast.py) tourne dans son propre
# thread et appelait le modele directement, EN PARALLELE du mode
# standard/thinking (_batch_worker ci-dessous) - les deux deplacent des
# enfants fractals entre CPU et GPU sur le MEME modele partage, sans
# aucune coordination. Resultat observe en pratique : "Expected all
# tensors to be on the same device, cuda:0 and cpu" quand les deux
# tournaient en meme temps (l'un renvoyait un enfant sur CPU pile au
# moment ou l'autre s'attendait a le trouver sur GPU).
#
# CE VERROU doit etre acquis par TOUT code qui appelle model(...) -
# _run_batched_generation ci-dessous ET cortex/fast.py - pour garantir
# qu'un seul forward pass tourne a la fois sur le modele, quel que soit
# le mode. Un verrou standard (pas RLock) suffit : jamais d'appel
# imbrique a model(...) depuis le meme thread.
verrou_modele = threading.Lock()


def _generation_debut():
    global _generations_en_cours
    with _generation_compteur_lock:
        _generations_en_cours += 1


def _generation_fin():
    global _generations_en_cours
    with _generation_compteur_lock:
        _generations_en_cours -= 1


def attendre_pause_sure(timeout: float = 120.0) -> bool:
    """A appeler depuis l'exterieur (apprentissage_force.py) avant un pas
    de gradient de rappel : coupe le signal pour empecher toute NOUVELLE
    generation de demarrer, puis attend que celles deja en cours (s'il y
    en a) se terminent proprement. Ne rend la main que quand plus aucun
    calcul n'est en train de tourner sur le modele.

    Retourne True si la pause a bien ete etablie, False si le timeout a
    ete atteint (dans ce cas, appeler reprendre_generation() quand meme
    pour ne pas bloquer les workers indefiniment)."""
    pret_pour_generation.clear()
    t0 = time.time()
    while True:
        with _generation_compteur_lock:
            if _generations_en_cours == 0:
                return True
        if time.time() - t0 > timeout:
            return False
        time.sleep(0.05)


def reprendre_generation():
    """Reactive le signal - a appeler juste apres le pas de gradient de
    rappel, que la pause ait reussi ou echoue (timeout)."""
    pret_pour_generation.set()


BATCH_WINDOW_SECONDS = 0.15


def echantillonner_logits(logits_dernier_pas, deja_generes=None, temperature: float = 0.8, top_k: int = 40, penalite_repetition: float = 1.3, max_repetition_consecutive: int = 3):
    """
    Echantillonnage temperature + top-k + penalite de repetition, au lieu
    du pur argmax (greedy).

    Observe en pratique : le greedy pur (toujours choisir le token le
    plus probable) cause des boucles de repetition frequentes sur un
    modele encore peu entraine - "opérations opérations", "ma ma ma ma
    ma...". Un peu de hasard pondere (temperature+top-k) aide, MAIS
    quand le modele devient tres confiant sur un token (proche de 100%
    de probabilite - observe en pratique : "au au au au au..." des
    dizaines de fois), meme un tirage aleatoire retombe presque toujours
    dessus. La penalite de repetition (technique standard, ex. CTRL/HF
    generate()) cible precisement ce cas : elle penalise explicitement
    les tokens DEJA utilises recemment, en reduisant leur logit avant
    le tirage, quelle que soit leur confiance de depart.

    CORRECTIF 1 (observe sur un vrai test, mode standard/override_level=3
    apres 100k pas) : la premiere version penalisait chaque token vu au
    moins une fois d'un facteur FIXE (1.3), une seule fois, peu importe
    combien de fois il revenait. La penalite est maintenant EXPONENTIELLE
    au nombre d'occurrences deja vues (plafonnee a 10 occurrences par
    securite numerique).

    CORRECTIF 2 (le premier ne suffisait toujours PAS dans certains cas -
    verifie par test isole avec un ecart de logit extreme, echec total
    de la penalite seule) : AUCUNE penalite multiplicative, aussi forte
    soit-elle, ne peut renverser un ecart de logit arbitrairement grand
    face a un noeud fractal profond extremement (mal) confiant sur un
    seul token. Un garde-fou DUR est ajoute en complement : si les
    max_repetition_consecutive derniers tokens consecutifs sont tous
    identiques, ce token est totalement EXCLU du tirage suivant
    (logit -> -inf), quelle que soit sa confiance. Ca garantit qu'aucune
    boucle ne peut jamais depasser ce nombre de repetitions consecutives.

    logits_dernier_pas : (B, vocab) - deja restreint aux classes reelles
    deja_generes : tensor (B, T) ou liste de listes - les tokens deja
                   produits dans cette generation (prompt inclus), pour
                   calculer la penalite. None = pas de penalite.
    temperature : >1 = plus aleatoire, <1 = plus proche du greedy,
                  0 = greedy pur (equivalent a argmax, ignore la penalite
                  ET le garde-fou dur - usage deconseille en pratique)
    top_k : ne considere que les k tokens les plus probables a chaque
            pas (evite de tirer un token totalement absurde par hasard)
    penalite_repetition : >1 rend les tokens deja vus moins probables,
                  de facon exponentielle avec le nombre d'occurrences
                  (1.0 = pas de penalite ; 1.3 est une valeur standard
                  raisonnable par occurrence, pas trop agressive)
    max_repetition_consecutive : au-dela de ce nombre de repetitions
                  IDENTIQUES D'AFFILEE, le token est exclu de force au
                  tirage suivant (0 ou None = garde-fou desactive)
    """
    if temperature <= 0:
        return logits_dernier_pas.argmax(dim=-1, keepdim=True)

    logits_scaled = logits_dernier_pas.clone() / temperature

    if deja_generes is not None:
        for b in range(logits_scaled.size(0)):
            ids_deja_vus = deja_generes[b]
            if torch.is_tensor(ids_deja_vus):
                ids_deja_vus = ids_deja_vus.tolist()

            if penalite_repetition and penalite_repetition != 1.0:
                comptes = {}
                for tid in ids_deja_vus:
                    comptes[tid] = comptes.get(tid, 0) + 1
                for tid, n in comptes.items():
                    if 0 <= tid < logits_scaled.size(-1):
                        facteur = penalite_repetition ** min(n, 10)
                        val = logits_scaled[b, tid]
                        # CTRL-style : si le logit est positif, on divise (le
                        # rend moins probable) ; s'il est negatif, on multiplie
                        # (le rend encore moins probable) - diviser un negatif
                        # l'augmenterait par erreur.
                        logits_scaled[b, tid] = val / facteur if val > 0 else val * facteur

            if max_repetition_consecutive and len(ids_deja_vus) >= max_repetition_consecutive:
                derniers = ids_deja_vus[-max_repetition_consecutive:]
                if len(set(derniers)) == 1:
                    logits_scaled[b, derniers[0]] = float("-inf")

    if top_k is not None and 0 < top_k < logits_scaled.size(-1):
        valeurs_seuil = torch.topk(logits_scaled, top_k, dim=-1).values[:, -1, None]
        logits_scaled = torch.where(
            logits_scaled < valeurs_seuil,
            torch.full_like(logits_scaled, float("-inf")),
            logits_scaled,
        )


    probs = torch.softmax(logits_scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@dataclass
class _PendingRequest:
    user_message: str
    max_new_tokens: int
    override_level: Optional[int]
    cascade: bool
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None


_queue_lock = threading.Lock()
_pending = []
_new_request_signal = threading.Event()


def _run_batched_generation(group, override_level, cascade):
    texts = [req.user_message for req in group]
    input_ids, _attention_mask = tokenizer.encode_batch(texts)
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    max_new = max(req.max_new_tokens for req in group)
    finished = [False] * len(group)
    end_pos = [None] * len(group)

    generated = input_ids
    last_output = None
    with torch.no_grad():
        for step in range(max_new):
            # Verrou pris UNIQUEMENT pour ce pas (un seul forward), pas
            # pour toute la generation - sinon le mode fast (qui partage
            # ce meme verrou) devrait attendre la fin complete d'une
            # reponse standard/thinking entiere (potentiellement plusieurs
            # MINUTES) avant meme de pouvoir commencer, ce qui va a
            # l'encontre du but meme du mode fast. Avec le verrou pris
            # pas par pas, l'attente maximale pour fast est la duree d'UN
            # forward pass (quelques secondes), pas une reponse entiere.
            with verrou_modele:
                last_output = model(generated, override_level=override_level, cascade=cascade)
            logits = last_output["logits"]
            # BUG CORRIGE : logits a une forme (B, config.vocab_size=32000),
            # mais le tokenizer ne sait decoder que 260 classes reelles
            # (tokenizer.vocab_size). Sans cette restriction, argmax pouvait
            # choisir une des ~31 740 classes "fantomes" (jamais reliees a un
            # octet reel) - le tokenizer les ignore SILENCIEUSEMENT au
            # decodage, ce qui pouvait rendre une reponse entiere vide
            # ("reponse brute"/ids bruts affiches), meme si le modele avait
            # bien "genere" quelque chose a chaque etape.
            next_ids = echantillonner_logits(logits[:, -1, :tokenizer.vocab_size], deja_generes=generated)
            generated = torch.cat([generated, next_ids], dim=1)

            for i in range(len(group)):
                if finished[i]:
                    continue
                if next_ids[i, 0].item() == tokenizer.EOS_ID:
                    finished[i] = True
                    end_pos[i] = step + 1
                elif (step + 1) >= group[i].max_new_tokens:
                    finished[i] = True
                    end_pos[i] = step + 1

            if all(finished) or generated.shape[1] >= tokenizer.max_seq_len:
                break

    cortex_output_pooled = None
    if last_output is not None:
        cortex_output_pooled = last_output["cortex_output"].mean(dim=1)

    for i, req in enumerate(group):
        limit = end_pos[i] if end_pos[i] is not None else generated.shape[1] - prompt_len
        new_ids = generated[i, prompt_len: prompt_len + limit]
        raw_ids = new_ids.tolist()
        response = tokenizer.decode(new_ids)
        if not response.strip():
            response = f"(aucun caractere affichable - ids bruts generes : {raw_ids})"
        req.result = response
        req.event.set()

        if memoire is not None and cortex_output_pooled is not None:
            memoire.apprendre_async(response, cortex_output_pooled[i])


def _process_batch(batch):
    if model is None:
        for req in batch:
            ids, _ = tokenizer.encode_batch([req.user_message])
            req.result = (
                f"Cortex a bien reçu : {req.user_message} "
                f"(tokenizer OK, {ids.shape[1]} tokens — modèle en attente : "
                f"{model_load_error or 'model.py absent'})"
            )
            req.event.set()
        return

    groups = {}
    for req in batch:
        key = (req.override_level, req.cascade)
        groups.setdefault(key, []).append(req)

    for (override_level, cascade), group in groups.items():
        _generation_debut()
        try:
            _run_batched_generation(group, override_level, cascade)
        except Exception as e:
            # Une erreur ici (ex. CUDA out of memory) ne doit JAMAIS bloquer
            # les requetes en attente indefiniment - on leur donne un message
            # d'erreur clair au lieu de les laisser attendre sans fin.
            print(f"[ERREUR] Echec de generation pour un groupe de requetes : {e}")
            for req in group:
                if req.result is None:
                    req.result = f"(erreur pendant la generation : {e})"
                    req.event.set()
        finally:
            _generation_fin()
            # Libere la memoire GPU fragmentee entre chaque groupe, pour
            # eviter l'accumulation qui menait au crash out-of-memory.
            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()


def _batch_worker():
    while True:
        _new_request_signal.wait()
        time.sleep(BATCH_WINDOW_SECONDS)
        # Si une pause de rappel linguistique est en cours (voir
        # attendre_pause_sure ci-dessus), on attend ici sans consommer la
        # file - les requetes en attente restent en securite jusqu'a la
        # reprise, elles ne sont ni perdues ni traitees en double.
        pret_pour_generation.wait()
        with _queue_lock:
            batch = _pending.copy()
            _pending.clear()
            _new_request_signal.clear()
        if batch:
            try:
                _process_batch(batch)
            except Exception as e:
                # Filet de securite ultime : meme si _process_batch plante
                # d'une maniere totalement imprevue, le worker NE DOIT JAMAIS
                # mourir - sinon plus aucune requete future ne recoit de
                # reponse pour le reste de la session (c'est ce qui s'est
                # passe lors du crash out-of-memory).
                print(f"[ERREUR CRITIQUE] _process_batch a echoue : {e}")
                for req in batch:
                    if req.result is None:
                        req.result = f"(erreur critique du serveur : {e})"
                        req.event.set()


_worker_thread = threading.Thread(target=_batch_worker, daemon=True)
_worker_thread.start()


# Gestion du serveur fast
_fast_server_thread = None
_fast_server_started = False

def _ensure_fast_server():
    """S'assure que le serveur fast est démarré."""
    global _fast_server_thread, _fast_server_started
    
    if _fast_server_started:
        return True
    
    try:
        from cortex.fast import _start_socket_server
        _fast_server_thread = threading.Thread(target=_start_socket_server, daemon=True)
        _fast_server_thread.start()
        _fast_server_started = True
        print("[INFO] Serveur fast démarré automatiquement")
        return True
    except Exception as e:
        print(f"[ERREUR] Impossible de démarrer le serveur fast : {e}")
        return False


def get_response(
    user_message: str,
    history=None,
    max_new_tokens: int = 100,
    override_level: int | None = None,
    cascade: bool = True,
    timeout: float = 600.0,
    mode: str = "standard",
) -> str:
    # Mode fast : utiliser le serveur fast
    if mode == "fast":
        try:
            if _ensure_fast_server():
                from cortex.fast import _send_to_fast_server
                # 15 tokens pour des réponses plus complètes (~5-8 mots)
                return _send_to_fast_server(user_message, max_tokens=15)
            else:
                return "Impossible de démarrer le serveur fast. Utilisez le mode standard."
        except ImportError:
            # Fallback si le module fast n'existe pas
            return "Module fast non disponible. Utilisez le mode standard."
    
    # Convertir le mode en override_level approprié
    if override_level is None and mode is not None:
        if mode == "fast":
            override_level = 1  # 1 niveau fractal max (cascade=False pour strict)
        elif mode == "standard":
            override_level = 3  # 3 niveaux fractal max
        elif mode == "thinking":
            override_level = None  # Tous les 5 niveaux (traitement automatique par complexité)
        else:
            override_level = 3  # Défaut: standard
    
    req = _PendingRequest(
        user_message=user_message,
        max_new_tokens=max_new_tokens,
        override_level=override_level,
        cascade=cascade,
    )
    with _queue_lock:
        _pending.append(req)
        _new_request_signal.set()

    if not req.event.wait(timeout=timeout):
        return "(délai dépassé, la génération n'a pas abouti à temps)"
    return req.result
