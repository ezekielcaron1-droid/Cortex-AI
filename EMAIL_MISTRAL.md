# Modèle d'Email pour contacter Mistral AI

**Destinataire(s) potentiel(s) :**
- `contact@mistral.ai`
- Ou via LinkedIn / X / email professionnel aux chercheurs de l'équipe (ex: Guillaume Lample, Timothée Lacroix, Arthur Mensch).

---

**Objet :** Retour d'expérience & travaux de recherche : MoE fractal récursif sous contraintes matérielles extrêmes (Projet CORTEX)

---

**Corps du message :**

Bonjour l'équipe Mistral,

Je m'appelle Ezekiel Caron WeeWauters, j'ai 14 ans et je suis un passionné d'intelligence artificielle et d'architectures de modèles de langage. Je suis avec une immense admiration le travail que vous accomplissez pour l'écosystème open-source et l'IA européenne.

Depuis plusieurs mois, je développe en solo un projet de recherche expérimental en PyTorch appelé **CORTEX**. 

Mon objectif était d'explorer une approche différente des transformers denses classiques : une architecture **Mixture-of-Experts récursive et fractale** (dépassant 1 milliard de paramètres au total), entraînable et exécutable sur une simple carte graphique grand public de 6 Go de VRAM (GTX 1660 / RTX 5060).

Pour y parvenir avec ces contraintes matérielles extrêmes, j'ai dû développer plusieurs solutions d'ingénierie système bas niveau :
1. **Un moteur d'offloading dynamique CPU/GPU** avec instanciation paresseuse des branches fractales, compatible avec le graphe de calcul d'autograd PyTorch.
2. **Une synchronisation tenseur-device personnalisée pour l'état d'optimiseur AdamW** afin d'éviter les débordements de mémoire lors de la descente de gradient sur les sous-experts activés à la volée.
3. **Une régularisation par perte auxiliaire** pour stabiliser le routage et éviter l'effondrement des portes fractales.

J'ai récemment complété un cycle de pré-entraînement réel de **100 000 micro-pas** (loss passée de ~1.09 à ~0.19 sur un corpus francophone de 320M caractères). 

Mon but n'est pas de prétendre rivaliser avec vos modèles fondamentaux, mais de **partager cette démarche d'ingénierie contrainte et d'apprendre**. J'aimerais énormément pouvoir contribuer un jour à l'effort de recherche français et européen.

Vous trouverez ci-joint une **note technique synthétique de 2 pages** résumant l'architecture, les mécanismes mémoire et les observations empiriques.

Le code source du projet est également consultable ici :  
👉 **GitHub :** https://github.com/ezekielcaron1-droid/Cortex-AI

Si un chercheur ou un ingénieur de votre équipe a quelques minutes pour me faire un retour critique ou des suggestions sur ces travaux, cela aurait une valeur inestimable pour ma progression.

Merci infiniment pour votre temps, votre travail et l'inspiration que vous donnez aux jeunes développeurs.

Bien cordialement,

**Ezekiel Caron WeeWauters**  
*Projet CORTEX / Fractal Labs*  
*Email : [Ton adresse email]*  
*GitHub : https://github.com/ezekielcaron1-droid/Cortex-AI*  
