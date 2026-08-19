# CORTEX — Recursive Fractal Mixture-of-Experts (MoE)

> **Projet de recherche expérimental en IA / Deep Learning**  
> Conception et entraînement d'une architecture hiérarchique récursive de plus d'un milliard de paramètres sous contraintes de calcul grand public (VRAM ≤ 6 Go).

---

## 🌟 Vue d'ensemble

**CORTEX** explore une approche alternative aux architectures Transformer denses conventionnelles. En s'appuyant sur un graphe d'arbres récursifs (`FractalNode`) et un routage dynamique guidé par la complexité (`FractalGate`), le modèle active sélectivement ses sous-experts à la volée.

```
Input Tokens (BPE 32k) ──► Traducteur (T) ──► Compréhension 3D (CO) ──► Nœud Fractal (RF / MoE) ──► Décodeur (S)
```

## 🛠️ Innovations & Solutions d'Ingénierie

1. **Offloading Dynamique CPU $\leftrightarrow$ GPU :** Instanciation paresseuse de l'arbre fractal ($5^N$ nœuds) maintenant les sous-branches inactives sur la RAM hôte et ne montant sur GPU que les experts actifs sans rompre le graphe d'autograd PyTorch.
2. **Synchronisation Tenseur-Device de l'Optimiseur :** Algorithme personnalisé pour synchroniser dynamiquement les états d'accumulation AdamW (`exp_avg`, `exp_avg_sq`) avec le device physique de chaque tenseur de poids.
3. **Stabilisation du Routage :** Perte auxiliaire d'équilibrage de charge pour contrer l'effondrement prématuré des portes de sélection d'experts.

## 📊 Résultats Empiriques

* **Cycle de pré-entraînement :** 100 000 micro-pas (batch effectif de 8 avec accumulation) sur un corpus francophone de 320 millions de caractères (~75M tokens).
* **Convergence de la Loss :** Passage de **1.09** (initial) à **0.19** (final).

---

## 📂 Structure du Répertoire

```text
├── cortex/                     # Package principal de l'architecture
│   ├── core/                   # Blocs transformer et nœud fractal récursif
│   ├── modules/                # Gating, mémoire locale, vérification
│   ├── sections/               # Sections (Traducteur, Compréhension, Réflexion...)
│   ├── bridge.py               # Moteur d'inférence & échantillonnage logits
│   ├── config.py               # Hyperparamètres du modèle
│   ├── model.py                # Assemblage global CortexModel
│   └── tokenizer.py            # Tokenizer BPE CamemBERT
├── train.py                    # Script d'entraînement avec gestion mémoire
├── cortex_server.py            # Serveur Flask d'interaction
├── NOTE_TECHNIQUE_CORTEX.md    # Synthèse technique de recherche
└── requirements.txt            # Dépendances du projet
```

## 🚀 Démarrage Rapide

```bash
# 1. Cloner le projet
git clone https://github.com/ezekielcaron1-droid/Cortex-AI.git
cd Cortex-AI

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'entraînement
python train.py
```

---
*Auteur : Ezekiel Caron Weewauters (14 ans) — Projet CORTEX / Fractal Labs.*
