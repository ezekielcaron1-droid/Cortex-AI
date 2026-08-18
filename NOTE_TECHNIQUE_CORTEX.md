# CORTEX : Architecture Mixture-of-Experts Récursive et Fractale sous Contraintes Matérielles

**Auteur :** Ezekiel Caron WeeWauters (14 ans) — Projet CORTEX / Fractal Labs  
**Domaine :** Deep Learning, Systèmes & Architectures de Modèles de Langage  
**Dépôt Officiel :** [github.com/ezekielcaron1-droid/Cortex-AI](https://github.com/ezekielcaron1-droid/Cortex-AI)  
**Date :** Août 2026  

---

## 1. Résumé & Vision (Abstract)

Le projet **CORTEX** explore une approche alternative aux transformers denses monolithiques : une **architecture hiérarchique récursive et fractale** (*Fractal Mixture-of-Experts*). 

L'objectif de cette recherche empirique est double :
1. Évaluer la faisabilité d'un réseau dynamique où la profondeur de calcul s'adapte à la complexité de l'entrée via des routeurs de branchement (*FractalGate*).
2. Concevoir un moteur d'exécution et d'entraînement capable de faire tourner un graphe dépassant **1 milliard de paramètres** sur une configuration grand public contrainte (**VRAM 6 Go / GTX 1660 / RTX 5060**).

---

## 2. Architecture du Modèle

```
Input Tokens (CamemBERT BPE, vocab 32k)
      │
      ▼
┌───────────────────────────────────────────────────────────┐
│ Section T : Traducteur d'Entrée (Embeddings + 3D Proj)    │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Section CO : Compréhension Multi-Plans (4 plans × 128d)   │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Section RF : Nœud Fractal Racine (Niveau 0)               │
│                                                           │
│  TransformerBlock ──► MiniBrainVerifier ──► FractalGate   │
│                                                   │       │
│                ┌──────────────────────────────────┴─────┐ │
│                │ 5 Enfants Fractaux (Niveau 1, MoE)     │ │
│                │   └── Recursion jusqu'à Niveau N...    │ │
│                └────────────────────────────────────────┘ │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Section S : Traducteur de Sortie & Projection Vocabulaire │
└───────────────────────────────────────────────────────────┘
```

### Caractéristiques architecturales clés :
* **Nœud Fractal Récursif (`FractalNode`) :** Chaque nœud encapsule un bloc décodeur, une mémoire court terme locale, un vérificateur d'erreurs de signal (*MiniBrainVerifier*) et un routeur vers $K$ sous-experts.
* **Instanciation Paresseuse & Élagage :** L'arbre complet ($5^N$ nœuds, >1,09 milliard de paramètres) n'est pas alloué statiquement en mémoire GPU mais instancié à la volée.

---

## 3. Ingénierie Système sous Contraintes de VRAM

Faire converger un modèle de plus d'un milliard de paramètres avec rétropropagation complète sur un GPU de 6 Go a nécessité le développement de solutions bas niveau spécifiques :

1. **Offloading Dynamique & Respect de l'Autograd :**
   * Les sous-branches inactives restent résidentes en RAM hôte (CPU).
   * Les experts sélectionnés par le routeur sont transvasés sur GPU uniquement pour leur passe `forward`.
   * Pour l'entraînement, les tenseurs restent ancrés sur GPU jusqu'à l'exécution complète de `loss.backward()`, puis sont déchargés via `decharger_enfants_gpu()` après `optimizer.step()`.

2. **Synchronisation Tenseur-Device de l'Optimiseur (AdamW) :**
   * Correction du désalignement de device : synchronisation dynamique des accumulateurs de momentum et variance (`exp_avg`, `exp_avg_sq`) pour chaque paramètre réveillé ou endormi, évitant les crashs `RuntimeError` et les OOM par rafale.

3. **Stabilisation du Routage (Anti-Effondrement) :**
   * Intégration d'une perte auxiliaire d'équilibrage de charge pour empêcher le *FractalGate* de s'effondrer sur un seul expert trivial.

---

## 4. Résultats Empiriques & Observations

* **Volume d'entraînement :** 100 000 micro-pas continus avec accumulation de gradients (batch effectif 8) sur un corpus francophone de 320 millions de caractères (~75M tokens).
* **Dynamique de convergence :**
  * Étape 0 $\rightarrow$ Loss initiale : $\approx 1.09$
  * Étape 60 000 $\rightarrow$ Loss stabilisée : $\approx 0.45$
  * Étape 100 000 $\rightarrow$ Loss finale : $\approx 0.19$ (avec pics de mémorisation locale à $0.0075$).
* **Observations / Défis ouverts :**
  * *Apprentissage du vocabulaire :* Parfaite acquisition du lexique et des tokens BPE français.
  * *Défi de phase & sémantique :* Nécessité d'un alignement de phase d'horloge interne et d'un volume de données massif pour passer de la simple complétion lexicale à la cohérence sémantique complexe et au raisonnement structuré.

---

## 5. Perspectives & Contact

Cette recherche prouve qu'il est possible de concevoir et d'entraîner des architectures non conventionnelles complexes sur des ressources matérielles très modestes grâce à une optimisation fine des flux de mémoire.

Je serais honoré de recueillir l'avis technique, les conseils ou les retours critiques de chercheurs et ingénieurs travaillant sur les modèles de pointe.

* **Auteur :** Ezekiel Caron WeeWauters  
* **Projet :** CORTEX / Fractal Labs  
* **Dépôt Git :** https://github.com/ezekielcaron1-droid/Cortex-AI  
