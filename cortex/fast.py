"""
cortex/fast.py
Serveur rapide pour les réponses en mode fast.

Utilise le VRAI modèle CORTEX avec des paramètres optimisés (override_level=1).
"""

import sys
import os
import socket
import json
import threading
import time
from typing import Optional

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Références vers le vrai modèle CORTEX
_model = None
_device = None
_tokenizer = None
_verrou_modele = None  # partage avec bridge.py - voir _init_fast()
_echantillonner = None  # fonction d'echantillonnage partagee - voir _init_fast()
_server_running = False


def _init_fast():
    """Initialise les références vers le vrai modèle CORTEX."""
    global _model, _device, _tokenizer
    
    if _model is not None:
        return True
    
    try:
        from cortex.bridge import model, device, tokenizer, verrou_modele, echantillonner_logits
        _model = model
        _device = device
        _tokenizer = tokenizer
        global _verrou_modele
        _verrou_modele = verrou_modele
        global _echantillonner
        _echantillonner = echantillonner_logits
        
        if _model is not None:
            print("[FAST] Serveur initialisé avec le vrai modèle CORTEX")
            return True
        else:
            print("[FAST] Modèle CORTEX non disponible")
            return False
    except Exception as e:
        print(f"[FAST] Erreur d'initialisation : {e}")
        return False


def _stream_response(user_message: str, mode: str = "fast"):
    """Générateur qui yield les tokens un par un pendant la génération."""
    global _model, _device, _tokenizer
    
    if _model is None or _tokenizer is None:
        yield {"error": "Erreur: modèle CORTEX non disponible", "finished": True}
        return
    
    try:
        # Encoder le message
        encoded = _tokenizer.encode(user_message, add_bos=True, add_eos=False)
        max_len = len(encoded) + 15 + 5  # 15 tokens + marge
        
        # Pré-allouer le tensor
        input_ids = torch.zeros(1, max_len, dtype=torch.long, device=_device)
        input_ids[0, :len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=_device)
        
        current_pos = len(encoded)
        
        # Configuration selon le mode
        original_max_retries = _model.cerveau.max_retries
        _model.cerveau.max_retries = 1  # Optimisé pour le streaming
        
        if mode == "fast":
            override_level = 1
        elif mode == "standard":
            override_level = 3
        else:
            override_level = None
        
        try:
            current_text = ""
            with torch.no_grad():
                for i in range(15):  # 15 tokens max
                    current_input = input_ids[:, :current_pos + 1]

                    # Verrou pris UNIQUEMENT pour ce pas - voir bridge.py
                    with _verrou_modele:
                        result = _model(current_input, override_level=override_level, cascade=False)
                    logits = result['logits']

                    next_token = _echantillonner(logits[:, -1, :], deja_generes=current_input)
                    input_ids[0, current_pos + 1] = next_token.squeeze()
                    current_pos += 1
                    new_token_id = next_token.item()
                    if new_token_id == _tokenizer.EOS_ID:
                        break

                    new_tokens = [new_token_id]
                    new_text = _tokenizer.decode(new_tokens, skip_special=True)
                    current_text += new_text

                    # Yield le token
                    yield {
                        "chunk": new_text,
                        "current_text": current_text,
                        "finished": False
                    }

                    if new_token_id == _tokenizer.EOS_ID:
                        break
        finally:
            _model.cerveau.max_retries = original_max_retries
        
        # Message de fin
        yield {
            "chunk": "",
            "current_text": current_text,
            "finished": True
        }
        
    except Exception as e:
        yield {"error": str(e), "finished": True}


def _process_message(user_message: str, max_new_tokens: int = 15) -> str:
    """Traite un message avec le modèle CORTEX en mode fast (optimisé)."""
    global _model, _device, _tokenizer
    
    if _model is None or _tokenizer is None:
        return "Erreur: modèle CORTEX non disponible"
    
    try:
        import time
        start_time = time.time()
        
        # Encoder le message (optimisé)
        encoded = _tokenizer.encode(user_message, add_bos=True, add_eos=False)
        max_len = len(encoded) + max_new_tokens + 5  # Marge de sécurité
        
        # Pré-allouer le tensor pour éviter torch.cat dans la boucle
        input_ids = torch.zeros(1, max_len, dtype=torch.long, device=_device)
        input_ids[0, :len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=_device)
        
        current_pos = len(encoded)
        
        # Configuration fast : réduire les boucles de feedback pour la vitesse
        original_max_retries = _model.cerveau.max_retries
        _model.cerveau.max_retries = 1  # Réduire à 1 tentative au lieu de 3
        
        try:
            # Mode fast : CORTEX complet mais optimisé
            with torch.no_grad():
                for i in range(max_new_tokens):
                    # Slice la taille actuelle (évite recréation de tensor)
                    current_input = input_ids[:, :current_pos + 1]

                    # Verrou pris UNIQUEMENT pour ce pas (voir bridge.py
                    # pour le raisonnement complet) - laisse une chance
                    # aux requetes standard/thinking de s'intercaler entre
                    # deux tokens fast, au lieu de devoir attendre la fin
                    # complete de la reponse fast.
                    with _verrou_modele:
                        result = _model(current_input, override_level=1, cascade=False)
                    logits = result['logits']

                    # Echantillonnage (temperature+top-k) au lieu du greedy pur -
                    # casse les boucles de repetition ("ma ma ma ma...")
                    next_token = _echantillonner(logits[:, -1, :], deja_generes=current_input)
                    input_ids[0, current_pos + 1] = next_token.squeeze()
                    current_pos += 1

                    # Arrêt si EOS
                    if next_token.item() == _tokenizer.EOS_ID:
                        break
        finally:
            # Restaurer la configuration originale
            _model.cerveau.max_retries = original_max_retries
        
        # Décoder uniquement les nouveaux tokens
        new_tokens = input_ids[0, len(encoded):current_pos].tolist()
        response = _tokenizer.decode(new_tokens, skip_special=True)
        
        elapsed = time.time() - start_time
        print(f"[FAST] Temps de génération : {elapsed*1000:.2f}ms ({current_pos - len(encoded)} tokens)")
        
        return response if response else "(pas de réponse générée)"
        
    except Exception as e:
        return f"Erreur lors de la génération : {e}"



def _start_socket_server(host='127.0.0.1', port=5001):
    """Démarre un serveur socket léger pour les requêtes fast."""
    global _server_running
    
    if not _init_fast():
        return False
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        server_socket.settimeout(5.0)
        _server_running = True
        
        print(f"[FAST] Serveur démarré sur {host}:{port}")
        
        while _server_running:
            try:
                client_socket, address = server_socket.accept()
                print(f"[FAST] Connexion de {address}")
                
                # Recevoir la requête
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    client_socket.close()
                    continue
                
                # Parser la requête JSON
                try:
                    request = json.loads(data)
                    user_message = request.get('message', '')
                    max_tokens = request.get('max_tokens', 15)  # 15 tokens pour des réponses plus complètes
                    
                    # Traiter la requête
                    response = _process_message(user_message, max_tokens)
                    
                    # Envoyer la réponse
                    response_data = json.dumps({'response': response})
                    client_socket.send(response_data.encode('utf-8'))
                    
                except json.JSONDecodeError:
                    error_response = json.dumps({'error': 'Invalid JSON'})
                    client_socket.send(error_response.encode('utf-8'))
                
                client_socket.close()
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[FAST] Erreur de traitement : {e}")
                
    except Exception as e:
        print(f"[FAST] Erreur du serveur : {e}")
    finally:
        server_socket.close()
        print("[FAST] Serveur arrêté")


def _send_to_fast_server(user_message: str, max_tokens: int = 5) -> str:
    """Envoie une requête au serveur fast et retourne la réponse."""
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 15s : le verrou n'est plus tenu que pendant CHAQUE pas de
        # generation individuel (pas toute une reponse standard/thinking
        # d'un coup) - voir bridge.py/_run_batched_generation. Le pire
        # cas d'attente est donc la duree d'un seul forward pass en
        # cours ailleurs (quelques secondes), pas plusieurs minutes.
        client_socket.settimeout(15.0)
        client_socket.connect(('127.0.0.1', 5001))
        
        # Envoyer la requête
        request = json.dumps({'message': user_message, 'max_tokens': max_tokens})
        client_socket.send(request.encode('utf-8'))
        
        # Recevoir la réponse
        response_data = client_socket.recv(8192).decode('utf-8')
        response = json.loads(response_data)
        
        client_socket.close()
        
        return response.get('response', response.get('error', 'Erreur inconnue'))
        
    except Exception as e:
        return f"Erreur de connexion au serveur fast : {e}"


if __name__ == "__main__":
    # Mode serveur : python -m cortex.fast
    _start_socket_server()