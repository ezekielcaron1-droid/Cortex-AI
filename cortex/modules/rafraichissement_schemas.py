"""
cortex/modules/rafraichissement_schemas.py
Régénère schemas.pt périodiquement en tâche de fond.
"""

import os
import subprocess
import sys
import threading
import time

INTERVALLE_SECONDES = 4 * 3600


def _boucle_rafraichissement():
    while True:
        time.sleep(INTERVALLE_SECONDES)
        try:
            print("[Schemas] Regeneration en cours (analyse_schemas.py)...")
            racine = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            subprocess.run([sys.executable, "analyse_schemas.py"], cwd=racine, timeout=1800)
            from cortex.bridge import model
            if model is not None:
                ok = model.cerveau.invention.recharger_schemas()
                if ok:
                    print("[Schemas] Nouveaux centroides recharges dans le modele actif.")
                else:
                    print("[Schemas] Aucun schemas.pt trouve apres regeneration.")
        except Exception as e:
            print(f"[Schemas] Echec du rafraichissement : {e}")


def demarrer():
    thread = threading.Thread(target=_boucle_rafraichissement, daemon=True)
    thread.start()
    print(f"[Schemas] Rafraichissement automatique active (toutes les {INTERVALLE_SECONDES/3600:.0f}h)")
