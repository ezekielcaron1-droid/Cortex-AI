import os
import sys
import json
import subprocess
from datetime import datetime

# ==========================================
# AUTO-INSTALLATION DES DÉPENDANCES
# ==========================================
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import torch  # requis par cortex.bridge
except ImportError:
    print("[INFO] Dépendances manquantes détectées.")
    print("[INFO] Installation automatique de Flask, Flask-CORS et PyTorch en cours...")
    print("[INFO] (PyTorch peut prendre plusieurs minutes à télécharger la première fois)")
    os.system(f"{sys.executable} -m pip install flask flask-cors torch")
    print("[INFO] Installation terminée. Redémarrage du script...")
    os.execv(sys.executable, ['python'] + sys.argv)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Toute la logique IA (tokenizer, modèle, génération, fallback, routing
# intelligent) vit dans cortex/bridge.py. Le serveur n'a besoin que de
# get_response().
from cortex.bridge import get_response

# ==========================================
# INITIALISATION DU SERVEUR
# ==========================================
# NOTE IMPORTANTE : ce serveur ne gère PLUS les comptes/conversations ni
# ne sert chat_23.html/login.html - c'est desormais le role de la
# facade (deployee sur Render). Ce serveur PC n'expose que :
#   - /api/chat          : le calcul reel (modele Cortex)
#   - /api/disponibilite : le statut horaire (lu par surveillance_acces_cortex.py)
# La facade les appelle via l'URL Tailscale Funnel de ce PC.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fichier de statut ecrit par surveillance_acces_cortex.py (tourne en
# parallele sur ce meme PC).
STATUT_ACCES_PATH = os.path.join(SCRIPT_DIR, "acces_cortex_statut.json")

app = Flask(__name__)
CORS(app)  # la facade (origine differente) doit pouvoir appeler ce serveur


@app.route('/api/disponibilite', methods=['GET'])
def disponibilite():
    """
    Lu par la facade (Render) via l'URL Tailscale Funnel de ce PC, pour
    savoir si le modele est disponible et selon quel creneau horaire.
    """
    if not os.path.isfile(STATUT_ACCES_PATH):
        return jsonify({
            "disponible_maintenant": False,
            "autorise": False,
            "debut": None,
            "fin": None,
            "raison": "aucun_statut_recu",
        })

    with open(STATUT_ACCES_PATH, encoding="utf-8") as f:
        statut = json.load(f)

    disponible_maintenant = False
    if statut.get("autorise") and statut.get("debut") and statut.get("fin"):
        maintenant = datetime.now().strftime("%H:%M")
        debut = statut["debut"]
        fin = statut["fin"]
        if debut <= fin:
            # Creneau normal, dans la meme journee (ex: 09:00 -> 17:00)
            disponible_maintenant = debut <= maintenant <= fin
        else:
            # Creneau qui traverse minuit (ex: 23:40 -> 02:00) - BUG
            # CORRIGE : une comparaison de texte simple ("23:51" <= "02:00")
            # est FAUSSE dans ce cas (le caractere '2' > '0'), donc
            # disponible_maintenant restait toujours a False des que le
            # creneau chevauchait minuit, meme en plein dans la fenetre
            # autorisee. Ici, on est dans le creneau si l'heure actuelle
            # est *apres* le debut OU *avant* la fin (logique different
            # d'un creneau normal, a cause du chevauchement).
            disponible_maintenant = maintenant >= debut or maintenant <= fin

    return jsonify({
        "disponible_maintenant": disponible_maintenant,
        "autorise": statut.get("autorise", False),
        "debut": statut.get("debut"),
        "fin": statut.get("fin"),
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Appelé par la façade (Render) - c'est ici que le calcul réel se fait."""
    data = request.json or {}
    user_message = data.get("message", "")
    history = data.get("history", [])
    mode = data.get("mode", "standard")

    if not user_message:
        return jsonify({"error": "Message vide"}), 400

    print(f"\n[HUMAIN] {user_message} [Mode: {mode}]")
    cortex_response = get_response(user_message, history, mode=mode)
    print(f"[CORTEX] {cortex_response}")

    return jsonify({
        "response": cortex_response,
        "status": "success"
    })


def activer_tailscale_funnel(port=5000, tentatives=3, delai_secondes=5):
    """
    Active Tailscale Funnel automatiquement au demarrage - sans ca, la
    facade (Render) ne peut pas joindre ce serveur, silencieusement,
    apres chaque redemarrage du PC (Funnel n'est pas permanent par
    defaut). "tailscale funnel --bg" est idempotent : le relancer alors
    qu'il est deja actif ne casse rien.

    VERIFIE ensuite que Funnel est vraiment actif publiquement (pas juste
    "tailnet only") et reessaie si besoin - observe en pratique : la
    commande peut "reussir" (code 0) juste apres le demarrage du PC/de
    Tailscale, sans que l'activation publique se propage tout de suite
    (service Tailscale pas encore totalement connecte a ce moment-la).
    """
    import time

    for tentative in range(1, tentatives + 1):
        try:
            resultat = subprocess.run(
                ["tailscale", "funnel", "--bg", str(port)],
                capture_output=True, text=True, timeout=15
            )
            if resultat.returncode != 0:
                print(f"[TAILSCALE] Tentative {tentative}/{tentatives} - echec :")
                print(f"  {resultat.stderr.strip()}")
            else:
                statut = subprocess.run(
                    ["tailscale", "funnel", "status"],
                    capture_output=True, text=True, timeout=15
                )
                sortie = statut.stdout
                if "Funnel on" in sortie and "tailnet only" not in sortie:
                    print(f"[TAILSCALE] Funnel actif publiquement sur le port {port} (verifie).")
                    return
                print(f"[TAILSCALE] Tentative {tentative}/{tentatives} - "
                      f"commande acceptee mais pas encore actif publiquement "
                      f"(sortie : {sortie.strip()!r}).")
        except FileNotFoundError:
            print("[TAILSCALE] ATTENTION - commande 'tailscale' introuvable (pas dans le PATH).")
            print("  Lance manuellement : tailscale funnel --bg 5000")
            return
        except subprocess.TimeoutExpired:
            print(f"[TAILSCALE] Tentative {tentative}/{tentatives} - timeout.")
        except Exception as e:
            print(f"[TAILSCALE] Tentative {tentative}/{tentatives} - erreur inattendue : {e}")

        if tentative < tentatives:
            print(f"[TAILSCALE] Nouvel essai dans {delai_secondes}s...")
            time.sleep(delai_secondes)

    print(f"[TAILSCALE] ATTENTION - Funnel pas confirme actif apres {tentatives} tentatives.")
    print("  La facade (Render) risque de ne pas pouvoir joindre ce serveur.")
    print("  Verifie/relance manuellement : tailscale funnel --bg 5000")


if __name__ == '__main__':
    print("\n========================================================")
    print("  SERVEUR CORTEX (PC) DÉMARRÉ")
    print("  Ce serveur n'expose que /api/chat et /api/disponibilite.")
    print("  Les comptes, conversations et l'interface web sont gérés")
    print("  par la façade (Render) - voir facade_app.py.")
    print("  Exposé publiquement via Tailscale Funnel pour que la")
    print("  façade puisse l'appeler.")
    print("========================================================\n")

    activer_tailscale_funnel(port=5000)

    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
