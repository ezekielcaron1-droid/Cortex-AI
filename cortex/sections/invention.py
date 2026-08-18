"""
cortex/sections/invention.py
Section INV : L'Invention

Crée des idées nouvelles À PARTIR DE RIEN. Aucun poids entraînable.
Apprend par l'expérience (banque persistante sur disque).

Mécanismes :
    1. Génération Pure    — bruit structuré dans l'espace latent
    2. Mutation           — perturbation aléatoire de représentations existantes
    3. Croisement         — fusion inter-sections (CO, RF, I)
    4. Extrapolation      — projeter AU-DELÀ des représentations connues
    5. Guidée             — utilise les expériences passées, affinée par les
                            schémas condensés (K-means) si disponibles

Sélection Darwinienne : génère N candidats, score chacun, garde le meilleur,
stocke le résultat dans la banque d'expérience. Pas de gradient descent.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.config import CortexConfig


class BanqueExperience:
    """Mémoire persistante des inventions passées."""

    def __init__(self, save_path: str, max_entries: int = 10_000):
        self.save_path = save_path
        self.max_entries = max_entries
        self.entries = []
        self._load()

    def _load(self):
        if os.path.exists(self.save_path):
            try:
                data = torch.load(self.save_path, weights_only=False)
                self.entries = data.get('entries', [])
            except Exception:
                self.entries = []

    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save({'entries': self.entries}, self.save_path)

    def add(self, direction: torch.Tensor, score: float, strategie: str):
        direction = direction.detach().cpu()
        if direction.dim() == 1:
            direction = direction.unsqueeze(0)

        seuil_sim = 0.95
        seuil_conflit = 0.3

        for entry in self.entries:
            edir = entry['direction']
            if edir.dim() == 1:
                edir = edir.unsqueeze(0)

            sim = F.cosine_similarity(direction, edir).item()

            if sim > seuil_sim:
                diff_score = abs(score - entry['score'])

                if diff_score < seuil_conflit:
                    w = entry.get('weight', 1)
                    entry['direction'] = (entry['direction'] * w + direction.squeeze(0)) / (w + 1)
                    entry['score'] = (entry['score'] * w + score) / (w + 1)
                    entry['weight'] = w + 1
                    self.save()
                    return
                else:
                    if 'nuances' not in entry:
                        entry['nuances'] = []
                    vecteur_nuance = direction.squeeze(0) - entry['direction']
                    entry['nuances'].append({
                        'vecteur': vecteur_nuance,
                        'score_critique': score
                    })
                    self.save()
                    return

        self.entries.append({
            'direction': direction.squeeze(0),
            'score': score,
            'strategie': strategie,
            'weight': 1,
            'nuances': []
        })

        if len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda e: e['score'] * (1 + 0.1 * e.get('weight', 1)), reverse=True)
            self.entries = self.entries[:self.max_entries // 2]

        self.save()

    def get_meilleures_directions(self, top_k: int = 10):
        if not self.entries:
            return []
        tri = sorted(self.entries, key=lambda e: e['score'], reverse=True)
        return [e['direction'] for e in tri[:top_k]]

    def get_meilleure_strategie(self) -> str:
        if not self.entries:
            return 'pure'
        scores_par_strat = {}
        counts = {}
        for e in self.entries:
            s = e['strategie']
            scores_par_strat[s] = scores_par_strat.get(s, 0.0) + e['score']
            counts[s] = counts.get(s, 0) + 1
        moyennes = {s: scores_par_strat[s] / counts[s] for s in scores_par_strat}
        return max(moyennes, key=moyennes.get)

    def __len__(self) -> int:
        return len(self.entries)


class Invention(nn.Module):
    """Section INV — Invention & Création d'idées nouvelles.

    Aucun poids entraînable. Apprend par l'expérience (BanqueExperience) et
    affine sa guidance grâce aux schémas condensés (K-means) calculés par
    analyse_schemas.py, si cortex/data/schemas.pt existe.
    """

    def __init__(self, config: CortexConfig, n_candidats: int = 5,
                 experience_path: str = None):
        super().__init__()
        self.d_embed = config.d_embed
        self.n_candidats = n_candidats
        self.temperature = 1.0

        rotation = torch.randn(config.d_embed, config.d_embed)
        rotation, _ = torch.linalg.qr(rotation)
        self.register_buffer('rotation_matrix', rotation)

        rotation2 = torch.randn(config.d_embed, config.d_embed)
        rotation2, _ = torch.linalg.qr(rotation2)
        self.register_buffer('rotation_matrix_2', rotation2)

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if experience_path is None:
            experience_path = os.path.join(base, 'data', 'experience_bank.pt')
        self.experience = BanqueExperience(experience_path)

        # Schémas condensés (K-means) : chargés en permanence désormais,
        # avant : uniquement actifs pendant que run_sara_v4_rapide.py
        # tournait, via un monkey-patch temporaire.
        schemas_path = os.path.join(base, 'data', 'schemas.pt')
        if os.path.exists(schemas_path):
            data_s = torch.load(schemas_path, weights_only=False)
            self.register_buffer('schemas_centroids', data_s['centroids'])
        else:
            self.schemas_centroids = None

        self._derniere_direction = None
        self._derniere_strategie = None

    def _strategie_pure(self, B, T, D, device):
        bruit = torch.randn(B, T, D, device=device) * self.temperature
        idee = torch.matmul(bruit, self.rotation_matrix)
        return idee

    def _strategie_mutation(self, base, force: float = 0.4):
        B, T, D = base.shape
        device = base.device
        bruit = torch.randn_like(base) * force
        perm = torch.randperm(D, device=device)
        permute = base[..., perm]
        alpha = torch.rand(1, device=device).item()
        idee = alpha * base + (1.0 - alpha) * permute + bruit
        return idee

    def _strategie_croisement(self, co, rf, img):
        B, T, D = co.shape
        device = co.device
        assignation = torch.randint(0, 3, (D,), device=device)
        masque_co = (assignation == 0).float().unsqueeze(0).unsqueeze(0)
        masque_rf = (assignation == 1).float().unsqueeze(0).unsqueeze(0)
        masque_i = (assignation == 2).float().unsqueeze(0).unsqueeze(0)
        idee = co * masque_co + rf * masque_rf + img * masque_i
        idee = idee + torch.randn_like(idee) * 0.1
        return idee

    def _strategie_extrapolation(self, co, rf):
        direction = rf - co
        facteur = 1.0 + torch.rand(1).item() * 1.5
        idee = co + direction * facteur
        rotation_legere = torch.matmul(idee, self.rotation_matrix_2)
        melange = 0.8 * idee + 0.2 * rotation_legere
        return melange

    def _strategie_guidee(self, B, T, D, device):
        """Guidée par l'expérience passée, affinée par les schémas condensés."""
        meilleures = self.experience.get_meilleures_directions(top_k=5)

        if not meilleures:
            return self._strategie_pure(B, T, D, device)

        idx = torch.randint(0, len(meilleures), (1,)).item()
        direction_base = meilleures[idx].to(device)

        if self.schemas_centroids is not None:
            sims = F.cosine_similarity(direction_base.unsqueeze(0), self.schemas_centroids)
            weights = F.softmax(sims / 0.1, dim=0)
            schema_cible = torch.sum(weights.unsqueeze(1) * self.schemas_centroids, dim=0)
            direction_base = 0.7 * direction_base + 0.3 * schema_cible

        base = direction_base.unsqueeze(0).unsqueeze(0).expand(B, T, D)
        bruit = torch.randn(B, T, D, device=device) * self.temperature * 0.4
        idee = base + bruit
        return idee

    def _scorer_candidat(self, candidat, references):
        candidat_flat = candidat.reshape(-1)
        distances = []
        for ref in references:
            ref_flat = ref.reshape(-1)
            sim = F.cosine_similarity(candidat_flat.unsqueeze(0), ref_flat.unsqueeze(0))
            distances.append(1.0 - sim.item())
        nouveaute = sum(distances) / len(distances) if distances else 1.0

        var_par_dim = candidat.var(dim=1).mean(dim=0)
        ratio_var = var_par_dim.std() / (var_par_dim.mean() + 1e-8)
        coherence = min(ratio_var.item(), 1.0)

        score = 0.6 * nouveaute + 0.4 * coherence
        return score

    def forward(self, co_output, rf_output, i_output):
        B, T, D = co_output.shape
        device = co_output.device
        references = [co_output, rf_output, i_output]

        candidats = []
        strategies_utilisees = []

        candidats.append(self._strategie_pure(B, T, D, device))
        strategies_utilisees.append('pure')

        candidats.append(self._strategie_mutation(rf_output))
        strategies_utilisees.append('mutation')

        candidats.append(self._strategie_croisement(co_output, rf_output, i_output))
        strategies_utilisees.append('croisement')

        candidats.append(self._strategie_extrapolation(co_output, rf_output))
        strategies_utilisees.append('extrapolation')

        candidats.append(self._strategie_guidee(B, T, D, device))
        strategies_utilisees.append('guidee')

        meilleur_score = -1.0
        meilleur_idx = 0
        for i, candidat in enumerate(candidats):
            score = self._scorer_candidat(candidat, references)
            if score > meilleur_score:
                meilleur_score = score
                meilleur_idx = i

        invention = candidats[meilleur_idx]
        strategie_gagnante = strategies_utilisees[meilleur_idx]

        self._derniere_direction = invention.mean(dim=0).mean(dim=0)
        self._derniere_strategie = strategie_gagnante

        return invention

    def record_feedback(self, score: float):
        if self._derniere_direction is not None:
            self.experience.add(
                direction=self._derniere_direction,
                score=score,
                strategie=self._derniere_strategie,
            )
            self._derniere_direction = None
            self._derniere_strategie = None

    def recharger_schemas(self):
        """Recharge schemas.pt depuis le disque — utile après qu'un cycle
        SARA / analyse_schemas.py a mis à jour le fichier, sans avoir à
        redémarrer le processus (voir apprentissage_force.py)."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        schemas_path = os.path.join(base, 'data', 'schemas.pt')
        if os.path.exists(schemas_path):
            device = self.rotation_matrix.device  # buffer toujours present, contrairement a parameters()
            data_s = torch.load(schemas_path, weights_only=False)
            self.schemas_centroids = data_s['centroids'].to(device)
            return True
        return False

    def get_stats(self) -> dict:
        n = len(self.experience)
        if n == 0:
            return {'total_experiences': 0, 'meilleure_strategie': 'aucune', 'score_moyen': 0.0}
        scores = [e['score'] for e in self.experience.entries]
        return {
            'total_experiences': n,
            'meilleure_strategie': self.experience.get_meilleure_strategie(),
            'score_moyen': sum(scores) / len(scores),
            'score_max': max(scores),
            'score_min': min(scores),
        }
