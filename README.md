# TP 2 – Deep Learning avec PyTorch (mode GPU)

**Université Abdelmalek Essaadi – Faculté des Sciences et Techniques de Tanger**
Département Génie Informatique – Master SITBD – Deep Learning
Encadrant : Pr. ELAACHAK Lotfi
Étudiante : Ahadri Botaina

---

## 1. Objectif

Se familiariser avec PyTorch pour construire des architectures de vision par ordinateur (CNN, Faster R-CNN, modèles pré-entraînés, Vision Transformer) et les entraîner sur GPU, puis comparer leurs performances sur le dataset **MNIST**.

## 2. Contenu du dépôt

| Fichier | Description |
|---|---|
| `TP2_DeepLearning.ipynb` | Notebook complet (code, résultats, analyses) exécuté sur Kaggle avec GPU |
| `results.csv` | Tableau final des métriques de tous les modèles |
| `README.md` | Rapport (ce fichier) |

## 3. Environnement

- Plateforme : **Kaggle Notebooks** (GPU activé, Internet activé)
- Python, **PyTorch**, torchvision, scikit-learn, pandas, matplotlib
- Graine aléatoire : `SEED = 42`

## 4. Dataset

**MNIST** : 70 000 images de chiffres manuscrits en niveaux de gris de 28×28 pixels, réparties en 10 classes (0 à 9).
- 60 000 images d'entraînement, 10 000 images de test.
- Source : https://www.kaggle.com/datasets/hojjatk/mnist-dataset (chargé ici via `torchvision.datasets.MNIST`, mêmes données).
- Prétraitement : conversion en tenseur puis normalisation (moyenne 0,1307 ; écart-type 0,3081).

## 5. Métriques utilisées

- **Accuracy** et **F1-score macro** sur le jeu de test
- **Loss** sur le jeu de test (cross-entropy pour les classifieurs ; somme des losses de détection pour Faster R-CNN)
- **Temps d'entraînement** (hors évaluation) et **nombre de paramètres**

---

## 6. Partie 1 – Modèles convolutifs

### 6.1 CNN from scratch

| Élément | Choix |
|---|---|
| Couches | Conv(1→32, 3×3, stride 1, padding 1) + BatchNorm + ReLU → MaxPool 2×2 → Conv(32→64, 3×3, stride 1, padding 1) + BatchNorm + ReLU → MaxPool 2×2 → Dropout 0,25 → FC 3136→128 + ReLU + Dropout 0,5 → FC 128→10 |
| Optimiseur | Adam, lr = 1e-3 |
| Régularisation | Weight decay 1e-4, Dropout, BatchNorm |
| Scheduler | StepLR (γ = 0,5 toutes les 2 époques) |
| Loss | CrossEntropy |
| Entraînement | 5 époques, batch 128, GPU |
| Paramètres | 421 834 |

### 6.2 Faster R-CNN

Faster R-CNN est un modèle de **détection d'objets**. Pour l'appliquer à MNIST, le problème est transformé en détection :
- chaque chiffre 28×28 est placé à une position aléatoire sur un canevas noir de 96×96 ;
- la cible est la **boîte englobante** du chiffre et sa classe (`label + 1`, la classe 0 étant le fond) ;
- à l'inférence, la classe de la détection au **meilleur score** sert de prédiction (aucune détection = prédiction fausse).

| Élément | Choix |
|---|---|
| Modèle | `fasterrcnn_mobilenet_v3_large_320_fpn`, backbone pré-entraîné ImageNet, 11 classes |
| Optimiseur | SGD, lr = 0,01, momentum 0,9, weight decay 1e-4, StepLR |
| Données | Sous-ensemble : 8 000 images d'entraînement, 2 000 images de test |
| Entraînement | 5 époques, batch 32, gradient clipping |
| Paramètres | 18 976 354 |

> La loss de détection (RPN + classification + régression de boîtes) n'est pas directement comparable à la cross-entropy du CNN.

### 6.3 Comparaison CNN vs Faster R-CNN

| Modèle | Accuracy | F1 macro | Loss test | Temps (s) | Paramètres |
|---|---|---|---|---|---|
| CNN | 0,9919 | 0,9918 | 0,0255 | 44,5 | 421 834 |
| Faster R-CNN | 0,9865 | 0,9862 | 0,3012 | 406,7 | 18 976 354 |

- Le CNN est un peu plus précis (+0,54 point), environ **9× plus rapide** et **45× plus léger**.
- Le CNN atteint 98,05 % dès la 1re époque. Faster R-CNN démarre à 71,0 % et se stabilise vers 98,7 % aux époques 4 et 5.
- Faster R-CNN localise correctement les chiffres (boîtes bien centrées) et les classes sont correctes sur les exemples visualisés.
- Faster R-CNN est conçu pour localiser et classer des objets dans une scène ; pour de la classification de chiffres, il est surdimensionné.

### 6.4 Fine-tuning de VGG16 et AlexNet (transfer learning)

| Élément | Choix |
|---|---|
| Modèles | VGG16 et AlexNet avec poids **ImageNet** |
| Adaptation des images | Redimensionnement 64×64 et répétition du canal sur 3 canaux |
| Stratégie | Couches convolutives (`features`) **gelées** ; dernière couche remplacée (1000 → 10) ; entraînement des couches FC |
| Optimiseur | Adam, lr = 1e-4, weight decay 1e-4 |
| Données | Sous-ensemble de 20 000 images d'entraînement ; test sur les 10 000 images |
| Entraînement | 3 époques, batch 64 |

### 6.5 Comparaison des 4 modèles de la partie 1

| Modèle | Accuracy | F1 macro | Loss test | Temps (s) | Params (total) | Params (entraînés) |
|---|---|---|---|---|---|---|
| CNN | 0,9919 | 0,9918 | 0,0255 | 44,5 | 421 834 | 421 834 |
| Faster R-CNN | 0,9865 | 0,9862 | 0,3012 | 406,7 | 18 976 354 | 18 917 458 |
| AlexNet (fine-tuned) | 0,9402 | 0,9391 | 0,1746 | 25,7 | 57 044 810 | 54 575 114 |
| VGG16 (fine-tuned) | 0,9615 | 0,9612 | 0,1180 | 71,6 | 134 301 514 | 119 586 826 |

**Conclusion de la partie 1**
- Classement par accuracy : **CNN > Faster R-CNN > VGG16 > AlexNet**.
- VGG16 gagne environ 2 points sur AlexNet mais est 2,8× plus lent et 2,4× plus lourd.
- Le fine-tuning ne bat pas le CNN sur MNIST : les features d'ImageNet (images naturelles en couleur) sont peu adaptées à des chiffres en niveaux de gris ; les images 28×28 sont agrandies artificiellement ; les convolutions sont gelées ; l'entraînement est court (3 époques, 20 000 images).
- Les modèles pré-entraînés ont 135 à 320 fois plus de paramètres que le CNN pour une précision inférieure. Le transfer learning est surtout utile quand les données sont rares ou complexes.
- Pistes d'amélioration : dégeler les derniers blocs convolutifs, entraîner plus longtemps.

---

## 7. Partie 2 – Vision Transformer (ViT) from scratch

Implémentation d'après le tutoriel *Vision Transformers from scratch (PyTorch)* :
1. **Patchify** : l'image 28×28 est découpée en 49 patchs de 4×4 pixels (vecteurs de dimension 16).
2. **Projection linéaire** de chaque patch vers la dimension cachée `d`.
3. Ajout d'un **class token** et d'un **encodage positionnel sinusoïdal**.
4. **Blocs Transformer** : LayerNorm → Multi-Head Self-Attention (+ résiduel) → LayerNorm → MLP (+ résiduel).
5. **Tête de classification** appliquée sur le class token.

| Hyper-paramètre | Valeur |
|---|---|
| Taille de patch | 4×4 (49 patchs + 1 class token) |
| Dimension cachée `d` | 64 |
| Têtes d'attention | 4 |
| Blocs Transformer | 4 (MLP ratio 4, activation GELU) |
| Optimiseur | AdamW, lr = 1e-3, weight decay 1e-2 |
| Scheduler | CosineAnnealingLR |
| Entraînement | 10 époques, batch 128 |
| Paramètres | 201 866 |

Évolution de l'accuracy de test du ViT : 94,83 % (époque 1) → 97,22 % (époque 3) → 98,03 % (époque 6) → **98,51 %** (époque 10).

## 8. Résultats finaux (tous les modèles)

| Modèle | Accuracy | F1 macro | Loss test | Temps (s) | Params (total) | Époques |
|---|---|---|---|---|---|---|
| **CNN** | **0,9919** | **0,9918** | **0,0255** | 44,5 | 421 834 | 5 |
| Faster R-CNN | 0,9865 | 0,9862 | 0,3012 | 406,7 | 18 976 354 | 5 |
| ViT (from scratch) | 0,9851 | 0,9849 | 0,0516 | 89,1 | 201 866 | 10 |
| VGG16 (fine-tuned) | 0,9615 | 0,9612 | 0,1180 | 71,6 | 134 301 514 | 3 |
| AlexNet (fine-tuned) | 0,9402 | 0,9391 | 0,1746 | 25,7 | 57 044 810 | 3 |

## 9. Interprétation et comparaison avec la partie 1

- **Performance** : le ViT entraîné from scratch atteint 98,51 %. Il est derrière le CNN (écart de 0,68 point) et quasi à égalité avec Faster R-CNN (0,14 point, non significatif vu la variabilité des runs). Il dépasse nettement VGG16 et AlexNet fine-tunés, sans aucun pré-entraînement.
- **Efficacité** : avec 0,20 M de paramètres, c'est le plus petit modèle (la moitié du CNN). Face à Faster R-CNN, il a environ 94× moins de paramètres et s'entraîne environ 4,6× plus vite pour une précision quasi identique.
- **Convergence** : le ViT démarre plus lentement (94,8 % à l'époque 1 contre 98,05 % pour le CNN) et nécessite plus d'époques.
- **Généralisation** : l'écart entre la loss d'entraînement (0,0080) et la loss de test (0,0516) indique un surapprentissage plus marqué que pour le CNN ; de la data augmentation ou davantage de régularisation pourraient aider.
- **Biais inductif** : le CNN intègre la localité et l'invariance par translation (convolutions + pooling), idéal pour MNIST. Le ViT doit apprendre ces propriétés via l'attention entre patchs, d'où une convergence plus lente et un besoin de plus de données.
- **Coût** : temps par époque quasi identique (≈ 8,9 s), mais le ViT a besoin de 2× plus d'époques.
- **Conclusion générale** : sur un dataset simple comme MNIST, un petit CNN reste le meilleur compromis précision / temps / taille. Le ViT est très compétitif sans pré-entraînement et montre son intérêt surtout sur de grands datasets ou avec du pré-entraînement.

## 10. Limites et remarques

- Les modèles n'ont pas été entraînés avec le même nombre d'époques (5, 3 ou 10) ni sur les mêmes sous-ensembles (Faster R-CNN : 8 000 images, VGG16/AlexNet : 20 000 images). La comparaison est donc **indicative**.
- Les résultats varient légèrement d'un run à l'autre (non-déterminisme du GPU, initialisation aléatoire). Lors d'un run précédent, Faster R-CNN avait obtenu 96,05 % au lieu de 98,65 % : des écarts de quelques dixièmes de point ne sont pas significatifs.
- Le temps d'entraînement dépend du GPU utilisé (Kaggle).

## 11. Reproduire les résultats

1. Ouvrir `TP2_DeepLearning.ipynb` sur Kaggle (*File > Import notebook*) ou Google Colab.
2. Activer le **GPU** (Kaggle : *Session options > Accelerator > GPU*) et **Internet : On** (téléchargement de MNIST et des poids pré-entraînés).
3. Exécuter toutes les cellules (*Run All*). Le fichier `results.csv` est généré à la fin.

## 12. Synthèse personnelle

Ce TP m'a permis de me familiariser avec PyTorch : création de `Dataset` et `DataLoader`, définition de modèles avec `nn.Module`, écriture d'une boucle d'entraînement et exécution sur GPU (`.to(device)`).

J'ai construit un CNN en choisissant les hyper-paramètres (kernel, padding, stride, pooling, optimiseur, régularisation par Dropout, weight decay et BatchNorm). J'ai compris la différence entre **classification** et **détection d'objets** en adaptant MNIST à Faster R-CNN (boîtes englobantes, RPN, RoI heads). J'ai appliqué le **transfer learning** avec VGG16 et AlexNet (geler les features, remplacer la dernière couche) et constaté qu'un modèle pré-entraîné n'est pas toujours meilleur qu'un petit modèle adapté à la tâche. Enfin, j'ai implémenté un **Vision Transformer** from scratch (patchs, class token, encodage positionnel, self-attention) et compris le rôle du biais inductif des CNN.

J'ai aussi appris à comparer des modèles avec plusieurs métriques (accuracy, F1, loss, temps, nombre de paramètres) et à rester prudent sur la validité d'une comparaison quand les protocoles d'entraînement diffèrent.

## 13. Références

- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, 2020.
- Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, 2015.
- Tutoriel : *Vision Transformers from scratch (PyTorch): a step-by-step guide* (Medium).
- Documentation PyTorch / torchvision.
