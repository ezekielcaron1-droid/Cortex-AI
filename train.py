"""
train.py
Pre-entrainement reel de CORTEX par descente de gradient (loss.backward()
+ optimizer.step()) - la seule chose qui manquait encore pour que le
modele apprenne vraiment le francais, au lieu de generer des octets
aleatoires.

Corpus : cortex/data/corpus_entrainement.txt (texte brut, un long flux
continu, encode une seule fois puis decoupe en fenetres aleatoires).

entrainer_quelques_pas() est reutilisable depuis l'exterieur (voir
apprentissage_force.py) pour faire quelques pas de gradient de rappel
sur le modele GPU deja charge, pendant une pause coordonnee avec
cortex/bridge.py - sans dupliquer le modele.

PERTE D'EQUILIBRAGE DE CHARGE : diagnostic_gating.py a revele que le
routeur fractal (FractalGate) s'est effondre - quasiment aucun enfant ne
s'active jamais, plafonnant la capacite reellement utilisee au seul
tronc commun (~77M parametres sur >1 milliard). Une perte auxiliaire
(voir cortex/modules/gating.py) est maintenant ajoutee a chaque pas pour
contrer cet effondrement.

Usage : python train.py
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F

from cortex.config import CortexConfig
from cortex.model import CortexModel
from cortex.modules import gating
from cortex.tokenizer import ByteTokenizer

CORPUS_PATH = os.path.join("cortex", "data", "corpus_entrainement.txt")
CHECKPOINT_PATH = os.path.join("cortex", "data", "checkpoint.pt")
OPTIMIZER_PATH = os.path.join("cortex", "data", "optimizer_state.pt")
STEP_STATE_PATH = os.path.join("cortex", "data", "train_step.json")

BATCH_SIZE = 1
LEARNING_RATE = 3e-4
N_STEPS = 80000
LOG_EVERY = 50
SAVE_EVERY = 500

# Accumulation de gradient : le vrai "batch effectif" devient
# BATCH_SIZE * ACCUMULATION_STEPS = 4, sans jamais augmenter la VRAM par
# micro-pas (chaque micro-pas individuel reste aussi leger qu'avant, un
# seul exemple a la fois). Seule la MISE A JOUR de l'optimiseur devient
# moins frequente mais bien moins bruitee - voir diagnostic_loss.py et
# la discussion sur le plateau de la loss autour de ~3.1-3.2.
ACCUMULATION_STEPS = 8

# Perte d'equilibrage de charge (anti-effondrement du routage fractal) -
# voir cortex/modules/gating.py pour le detail. CIBLE = taux d'activation
# moyen vise (20% des enfants actifs en moyenne, pas 0% ni 50%).
CIBLE_EQUILIBRAGE = 0.2
POIDS_EQUILIBRAGE = 0.09

# Learning rate variable : montee progressive (warmup) puis decroissance
# cosinus jusqu'a 10% du taux de pointe - aide a stabiliser la convergence
# fine, plutot qu'un taux constant du debut a la fin (voir discussion sur
# le plateau de loss).
WARMUP_STEPS = 500
LR_MIN_RATIO = 0.1


def taux_apprentissage(step: int, lr_max: float, total_steps: int,
                        warmup_steps: int = WARMUP_STEPS, lr_min_ratio: float = LR_MIN_RATIO) -> float:
    """Calcule le learning rate pour un pas ABSOLU donne (compatible avec
    la reprise entre sessions - le schedule est cale sur le pas absolu,
    pas sur le nombre de pas de CET appel)."""
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    progres = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    facteur = lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progres))
    return lr_max * facteur


def get_batch(ids: torch.Tensor, seq_len: int, batch_size: int, device):
    """Echantillonne aleatoirement batch_size fenetres de seq_len+1 tokens
    dans le corpus encode, et les decoupe en (entree, cible decalee de 1)."""
    max_start = ids.shape[0] - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([ids[s: s + seq_len] for s in starts])
    y = torch.stack([ids[s + 1: s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


def reconstruire_arbre_depuis_checkpoint(model, state_dict) -> int:
    """Reconstruit la STRUCTURE de l'arbre fractal (quels enfants existaient)
    a partir des noms de cles du checkpoint, AVANT load_state_dict.

    BUG CRITIQUE CORRIGE ICI : les enfants fractals sont crees paresseusement
    (_init_children, voir fractal_node.py) - un CortexModel() fraichement
    instancie n'a QUE le tronc (~77M parametres), aucun enfant. Charger un
    checkpoint de ~900M parametres avec strict=False dans ce modele frais
    ignore SILENCIEUSEMENT les ~5600 cles sans sous-module correspondant
    (les enfants profonds) - confirme empiriquement : le modele reste a
    77M parametres apres chargement, quel que soit le contenu reel du
    fichier. Consequence : chaque redemarrage de train.py perdait la quasi
    totalite de l'arbre explore (~820M parametres a chaque fois), qui
    devait se re-explorer de zero avec des poids aleatoires neufs.

    Principe : chaque cle du checkpoint contenant "._children.N." indique
    qu'un noeud a cette profondeur avait ses 5 enfants instancies (rappel :
    _init_children() les cree TOUJOURS tous les 5 en un coup, jamais un
    seul). On extrait tous les chemins concernes, on les trie du moins
    profond au plus profond (necessaire pour pouvoir naviguer jusqu'aux
    chemins profonds une fois leurs parents instancies), et on appelle
    _init_children() sur chacun.

    Retourne le nombre de noeuds dont les enfants ont ete (re)crees.
    """
    chemins = set()
    for cle in state_dict.keys():
        parties = cle.split(".")
        for i, p in enumerate(parties):
            if p == "_children" and i + 1 < len(parties) and parties[i + 1].isdigit():
                chemins.add(".".join(parties[:i]))

    # Du moins profond au plus profond : un chemin profond n'est
    # navigable qu'une fois ses parents deja reconstruits.
    chemins_tries = sorted(chemins, key=lambda c: c.count("_children"))

    n_inities = 0
    for chemin in chemins_tries:
        try:
            noeud = model.get_submodule(chemin)
        except AttributeError:
            continue  # chemin incoherent (checkpoint corrompu/etranger) - ignore, jamais fatal
        if hasattr(noeud, "_init_children") and not noeud._children_initialized:
            noeud._init_children()
            n_inities += 1
    return n_inities


def placer_hors_arbre_sur_device(model, device) -> None:
    """Remplace un model.to(device) global (dangereux ici) : ne deplace
    sur device QUE ce qui est hors de tout _children fractal (tronc,
    embeddings, tete de sortie...). Les enfants fractals restent sur CPU,
    comme le veut l'architecture (voir fractal_node.py) - seuls les
    enfants actifs sont deplaces au coup par coup pendant forward().

    Un model.to(device) global chargerait d'un coup les ~900M de
    parametres de l'arbre entier sur une carte a 6 Go - OOM quasi certain,
    et de toute facon contraire au principe d'offloading MoE de Cortex.
    """
    for nom, param in model.named_parameters():
        if "_children" not in nom:
            param.data = param.data.to(device)
    for nom, buf in model.named_buffers():
        if "_children" not in nom:
            buf.data = buf.data.to(device)


def synchroniser_parametres_optimiseur(model, optimizer) -> int:
    """Ajoute a l'optimiseur tout parametre du modele qu'il ne connait pas
    encore.

    BUG CRITIQUE CORRIGE ICI : torch.optim.AdamW(model.parameters(), ...)
    capture une liste FIGEE au moment de sa creation. Or les enfants
    fractals sont crees paresseusement (_init_children) au fil des
    forward - tout enfant jamais visite avant la creation de l'optimiseur
    n'est JAMAIS mis a jour par optimizer.step(), meme s'il recoit bien
    un gradient correct via backward(). Verifie et confirme par
    test_bug_optimiseur.py : l'optimiseur restait fige a 77M parametres
    alors que le modele grandissait a plus d'1 milliard.

    A appeler avant CHAQUE optimizer.step() (cout negligeable : simple
    comparaison d'identites Python, pas une operation sur les tenseurs).
    Retourne le nombre de nouveaux parametres ajoutes (pour diagnostic).
    """
    deja_suivis = {id(p) for g in optimizer.param_groups for p in g['params']}
    nouveaux = [p for p in model.parameters() if id(p) not in deja_suivis]
    if nouveaux:
        optimizer.add_param_group({'params': nouveaux})
        # Pre-cree l'etat AdamW DIRECTEMENT sur CPU pour ces nouveaux
        # parametres. Sans ca, optimizer.step() decouvre ces parametres
        # au moment meme de la mise a jour et cree leur etat (exp_avg /
        # exp_avg_sq) sur LEUR DEVICE ACTUEL - potentiellement GPU, en
        # rafale, si plusieurs branches jamais vues sont visitees d'un
        # coup (frequent en tout debut d'entrainement, modele fraichement
        # reinitialise). C'est ce qui causait un crash CUDA
        # out-of-memory des les tout premiers pas apres le changement de
        # tokenizer (repart de zero = beaucoup de branches nouvelles a la
        # fois). En pre-placant l'etat sur CPU des le depart,
        # synchroniser_etat_optimiseur() (appele juste apres) se charge
        # de ne monter sur GPU QUE l'etat des parametres reellement
        # actifs ce pas-ci - jamais de rafale.
        for p in nouveaux:
            optimizer.state[p] = {
                'step': torch.tensor(0.0),
                'exp_avg': torch.zeros_like(p, device='cpu'),
                'exp_avg_sq': torch.zeros_like(p, device='cpu'),
            }
    return len(nouveaux)


def synchroniser_etat_optimiseur(model, optimizer) -> None:
    """Aligne le device de l'etat interne de l'optimiseur (momentum et
    variance d'AdamW, un tenseur par parametre) sur le device ACTUEL de
    chaque parametre.

    Necessaire a cause de l'offloading MoE (voir fractal_node.py) : un
    enfant fractal change de device (GPU <-> CPU) d'un pas a l'autre,
    mais son poids et l'etat de l'optimiseur associe NE bougent PAS
    ensemble automatiquement - PyTorch ne les lie pas. Sans cette
    synchronisation explicite :
      - Un enfant reactive sur GPU avec un etat reste sur CPU fait
        planter optimizer.step() (RuntimeError, tenseurs sur des devices
        differents).
      - Un enfant renvoye sur CPU dont l'etat reste sur GPU s'accumule
        indefiniment en VRAM fantome - c'est exactement ce qui a cause
        le crash CUDA out-of-memory apres ~4900 pas.

    A appeler juste avant optimizer.step() (ramene sur GPU l'etat des
    enfants actifs ce pas-ci) ET juste apres le dechargement des enfants
    inactifs (renvoie sur CPU l'etat de ceux qu'on vient de decharger).
    """
    for p in model.parameters():
        state = optimizer.state.get(p)
        if not state:
            continue
        for k, v in state.items():
            if torch.is_tensor(v) and v.device != p.device:
                state[k] = v.to(p.device)


def entrainer_quelques_pas(
    model, optimizer, ids, config, device,
    n_pas: int,
    seq_len: int = 64,
    batch_size: int = 1,
    log_every: int | None = None,
    save_every: int | None = None,
    sauvegarde_callback=None,
    step_offset: int = 0,
    total_pour_log: int | None = None,
    accumulation_steps: int = 1,
    cible_equilibrage: float = CIBLE_EQUILIBRAGE,
    poids_equilibrage: float = POIDS_EQUILIBRAGE,
) -> float:
    """Execute n_pas MICRO-PAS (forward + backward) sur des fenetres
    aleatoires du corpus deja encode (ids).

    Avec accumulation_steps > 1 : les gradients de plusieurs micro-pas
    consecutifs sont accumules (moyennes) AVANT qu'optimizer.step() ne
    soit reellement appele - donne un signal de gradient bien moins
    bruite qu'un batch_size=1 pur (utile pour un modele MoE tres eparse
    ou chaque micro-pas n'active qu'une petite fraction aleatoire des
    poids), sans jamais augmenter la VRAM par micro-pas. La derniere
    serie incomplete d'un appel est toujours appliquee (pas de gradient
    perdu en fin de fonction).

    Chaque pas ajoute aussi une perte d'equilibrage de charge (voir
    cortex/modules/gating.py) pour contrer l'effondrement du routage
    fractal observe (diagnostic_gating.py) - sans elle, le routeur
    apprend a desactiver quasiment tous les enfants en permanence.

    Reutilisable a la fois par main() ci-dessous (pre-entrainement dedie,
    des milliers de pas) et par apprentissage_force.py (quelques pas de
    rappel sur le modele GPU deja actif, sans copie - voir
    _faire_un_rappel_pause dans apprentissage_force.py).

    Ne charge ni ne sauvegarde de checkpoint lui-meme (sauf via
    sauvegarde_callback, optionnel, appele tous les save_every pas) - la
    responsabilite du chargement/sauvegarde final reste a l'appelant.

    Retourne la loss (language modeling PURE, sans la perte d'equilibrage)
    moyenne sur les n_pas effectues - comparable aux mesures precedentes.
    """
    model.train()
    seq_len = min(seq_len, config.max_seq_len - 1)
    total_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad()

    for i in range(1, n_pas + 1):
        step = step_offset + i
        x, y = get_batch(ids, seq_len, batch_size, device)

        gating.activer_capture_equilibrage()
        output = model(x)
        logits = output["logits"]  # (B, T, vocab_size)

        loss_lm = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            y.reshape(-1),
        )
        perte_aux = gating.perte_equilibrage(cible=cible_equilibrage, poids=poids_equilibrage)
        gating.desactiver_capture_equilibrage()

        loss_totale = loss_lm if perte_aux is None else loss_lm + perte_aux

        # Divise avant le backward pour que l'accumulation fasse une
        # MOYENNE des gradients (pas une somme) - garde le pas d'apprentissage
        # effectif comparable, quel que soit accumulation_steps.
        (loss_totale / accumulation_steps).backward()

        # Mise a jour reelle de l'optimiseur seulement tous les
        # accumulation_steps micro-pas - ou en tout dernier recours a la
        # fin de cet appel, pour ne jamais perdre un gradient accumule.
        mise_a_jour = (i % accumulation_steps == 0) or (i == n_pas)
        n_nouveaux = 0
        lr_actuel = None
        if mise_a_jour:
            n_nouveaux = synchroniser_parametres_optimiseur(model, optimizer)
            synchroniser_etat_optimiseur(model, optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if total_pour_log:
                lr_actuel = taux_apprentissage(step, LEARNING_RATE, total_pour_log)
                for g in optimizer.param_groups:
                    g['lr'] = lr_actuel
            optimizer.step()
            optimizer.zero_grad()

        # Le backward de CE micro-pas est termine : les enfants fractals
        # restes sur GPU pendant ce micro-pas peuvent etre renvoyes sur
        # CPU en toute securite (leur .grad accumule les suit sur CPU,
        # intact, en attendant la prochaine mise a jour reelle).
        model.cerveau.reflexion.fractal_root.decharger_enfants_gpu()
        synchroniser_etat_optimiseur(model, optimizer)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        total_loss += loss_lm.item()

        if log_every and step % log_every == 0:
            elapsed = time.time() - t0
            suffixe = f"/{total_pour_log}" if total_pour_log else ""
            aux_str = f", aux = {perte_aux.item():.4f}" if perte_aux is not None else ""
            nouv_str = f", +{n_nouveaux} params optim." if n_nouveaux else ""
            lr_str = f", lr = {lr_actuel:.2e}" if lr_actuel is not None else ""
            print(f"[PAS {step}{suffixe}] loss = {loss_lm.item():.4f}{aux_str}{nouv_str}{lr_str} - {elapsed:.1f}s ecoulees")

        if save_every and sauvegarde_callback and step % save_every == 0:
            sauvegarde_callback(step)

    model.eval()
    return total_loss / n_pas if n_pas > 0 else 0.0


def main():
    print("=== PRE-ENTRAINEMENT CORTEX ===\n")

    if not os.path.exists(CORPUS_PATH):
        print(f"[ERREUR] Corpus introuvable : {CORPUS_PATH}")
        return

    config = CortexConfig()
    tokenizer = ByteTokenizer(max_seq_len=config.max_seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Chargement du corpus depuis {CORPUS_PATH}...")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        texte = f.read()
    print(f"[INFO] Corpus charge : {len(texte):,} caracteres")

    # Encodage en UN SEUL flux continu (pas de BOS/EOS par fenetre, pas de
    # troncature - c'est tout le corpus qui devient un long tenseur d'ids)
    ids_liste = tokenizer.encode(texte, add_bos=False, add_eos=False, truncate=False)
    ids = torch.tensor(ids_liste, dtype=torch.long)
    print(f"[INFO] Corpus encode : {ids.shape[0]:,} tokens")

    # 1. Créer le modèle
    model = CortexModel(config)

    # 2. Charger le checkpoint : reconstruire d'abord la STRUCTURE de
    #    l'arbre fractal (quels enfants existaient), sinon load_state_dict
    #    ignore silencieusement tout ce qui n'a pas encore de sous-module
    #    correspondant dans un modele fraichement instancie (voir
    #    reconstruire_arbre_depuis_checkpoint ci-dessus pour le detail du
    #    bug corrige ici - jusque-la, ~820M parametres perdus a chaque
    #    redemarrage).
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        n_noeuds_reconstruits = reconstruire_arbre_depuis_checkpoint(model, checkpoint_dict)
        resultat = model.load_state_dict(checkpoint_dict, strict=False)
        print(f"[INFO] Reprise depuis un checkpoint existant (strict=False).")
        print(f"[INFO] Arbre fractal reconstruit : {n_noeuds_reconstruits} noeud(s) "
              f"reinities, {len(resultat.unexpected_keys)} cle(s) encore ignoree(s) "
              f"(0 = reconstruction complete).")
        del checkpoint_dict
    else:
        print("[INFO] Depart avec des poids aleatoires (premier entrainement).")

    # 3. Placer sur device UNIQUEMENT ce qui est hors de l'arbre fractal
    #    (tronc, embeddings, tete de sortie) - PAS un model.to(device)
    #    global, qui deplacerait tout l'arbre (potentiellement ~900M
    #    parametres) sur une carte a 6 Go d'un coup. Les enfants fractals
    #    restent sur CPU, deplaces au coup par coup pendant forward()
    #    (voir fractal_node.py, principe d'offloading MoE).
    placer_hors_arbre_sur_device(model, device)

    n_params = model.count_parameters()
    print(f"[INFO] Modele CORTEX - {n_params:,} parametres entrainables au total\n")

    # 4. Créer l'optimiseur APRÈS que l'arbre soit reconstruit (sinon il
    #    ne connaitrait que le tronc, meme bug que celui corrige par
    #    synchroniser_parametres_optimiseur - ici on l'evite des le depart).
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Reprise de l'etat de l'optimiseur (momentum/variance d'AdamW) - sans
    # ca, chaque redemarrage manuel (Ctrl+C puis relance) reperd tout cet
    # historique, ce qu'on a observe plusieurs fois dans les logs (valeurs
    # qui sautent juste apres un redemarrage). Protege par try/except : si
    # la structure du modele a change depuis la derniere sauvegarde (plus
    # ou moins d'enfants fractals instancies), on repart proprement sur un
    # etat neuf plutot que de planter.
    if os.path.exists(OPTIMIZER_PATH):
        try:
            optimizer.load_state_dict(torch.load(OPTIMIZER_PATH, map_location=device))
            print("[INFO] Etat de l'optimiseur (AdamW) restaure.")
        except Exception as e:
            print(f"[INFO] Etat de l'optimiseur incompatible ({e}) - repart a neuf.")

    # Reprise du compteur de pas (pas juste des poids) : si un run precedent
    # s'est arrete en cours de route (ex. crash), on reprend exactement la
    # ou on s'est arrete au lieu de repartir a 1 et de refaire N_STEPS en
    # plus du total deja fait.
    step_deja_fait = 0
    if os.path.exists(STEP_STATE_PATH):
        with open(STEP_STATE_PATH, encoding="utf-8") as f:
            step_deja_fait = json.load(f).get("dernier_pas", 0)
        print(f"[INFO] Reprise du compteur de pas : {step_deja_fait} deja effectues.")

    n_pas_restants = max(0, N_STEPS - step_deja_fait)
    if n_pas_restants == 0:
        print(f"[INFO] Les {N_STEPS} pas prevus sont deja termines. Rien a faire.")
        return

    seq_len = min(64, config.max_seq_len - 1)  # fenetres courtes pour demarrer vite
    batch_effectif = BATCH_SIZE * ACCUMULATION_STEPS
    print(f"[INFO] Demarrage - {n_pas_restants} pas restants sur {N_STEPS}, "
          f"batch={BATCH_SIZE} x accumulation={ACCUMULATION_STEPS} "
          f"(effectif={batch_effectif}), seq_len={seq_len}")
    print(f"[INFO] Perte d'equilibrage active : cible={CIBLE_EQUILIBRAGE}, poids={POIDS_EQUILIBRAGE}\n")

    def _sauver(step):
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
        with open(STEP_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"dernier_pas": step}, f)
        print(f"[CHECKPOINT] Sauvegarde a l'etape {step}")

    entrainer_quelques_pas(
        model, optimizer, ids, config, device,
        n_pas=n_pas_restants,
        seq_len=seq_len,
        batch_size=BATCH_SIZE,
        log_every=LOG_EVERY,
        save_every=SAVE_EVERY,
        sauvegarde_callback=_sauver,
        step_offset=step_deja_fait,
        total_pour_log=N_STEPS,
        accumulation_steps=ACCUMULATION_STEPS,
    )

    _sauver(N_STEPS)
    print(f"\n[TERMINE] Entrainement fini, checkpoint final sauvegarde dans {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
