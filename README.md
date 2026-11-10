# w2vec — Word2Vec (skip-gram + negative sampling) from scratch

Implémentation de **word2vec** (skip-gram avec échantillonnage négatif) en **NumPy pur** : gradients écrits à la main, sans PyTorch ni TensorFlow. Les embeddings sont appris sur *Le Comte de Monte-Cristo* puis comparés à [Gensim](https://radimrehurek.com/gensim/).

Projet réalisé dans le cadre d'un TP de machine learning. Le but n'est pas seulement de faire tourner l'algorithme, mais de **mesurer l'écart avec Gensim, de comprendre d'où il vient et de le réduire pas à pas** (voir [Perspectives](#perspectives)).

## Résultats

Évaluation sur 100 triplets `(m, m+, m-)` : un triplet est réussi si `cos(m, m+) > cos(m, m-)`. Moyenne ± écart-type sur **10 graines** (initialisation de W et ordre des exemples), même dataset pour tous les runs.

| Modèle | Taux de réussite | Perte finale | Temps / run (CPU) |
|---|---|---|---|
| w2v from scratch, init. de W uniforme sur [0, 1] (baseline) | 74,1 % ± 3,5 | 2,811 | ≈ 5 min |
| w2v from scratch, init. de W centrée (piste 1) | 75,5 % ± 1,7 | 2,797 | ≈ 5-6 min |
| Gensim (skip-gram, voir config ci-dessous) | **84,4 % ± 2,6** | n/a | 5 à 13 s |

Configuration commune : `n=100`, `L=2`, `k=10`, `η=0.1`, 5 itérations, `minc=5`.
Gensim : `Word2Vec(vector_size=100, window=5, min_count=1, sg=1, negative=5, alpha=0.1, epochs=5, workers=1)` sur le corpus déjà filtré.

**Lecture des résultats**

- L'écart avec Gensim (≈ 9 points) dépasse nettement la variabilité entre graines. C'est l'écart que ce projet cherche à expliquer et à réduire.
- Le jeu d'évaluation ne contient que 100 triplets (1 triplet = 1 point). Les différences de 1 à 2 points entre deux variantes de w2v **ne sont pas significatives** ; je ne conclus que sur des écarts plus larges ou sur la tendance.
- La configuration Gensim ci-dessus **n'est pas encore alignée** sur celle de w2v (`window=5` côté Gensim, qui est une distance maximale de chaque côté, contre `L=2` ; `negative=5` contre `k=10`). La comparer à paramètres égaux est la première étape des perspectives.

## Ce que fait le code

Le pipeline s'enchaîne en cinq étapes :

- `filter_rare_words` : supprime les mots de moins de 5 occurrences.
- `generate_positive_samples` : fenêtre ±L, soit 876 832 paires (cible, contexte).
- `generate_negative_samples` : k=10 négatifs par paire, tirés selon P(w) ∝ count(w)^0.75.
- `w2v` : descente de gradient (SGD) sur les matrices W et C, puis sauvegarde des embeddings au format `.txt`.
- `eval_w2v` : similarité cosinus sur les triplets d'évaluation.

- **Vocabulaire** : 13 023 mots distincts, 3 156 conservés (≥ 5 occurrences), 233 476 positions cibles.
- **Négatifs** : distincts entre eux et différents de la cible et du contexte. Le dataset peut être sauvegardé en JSON (`save_dataset` / `load_dataset`) pour comparer plusieurs runs sur exactement les mêmes données.
- **Entraînement** : deux matrices, `W` (mots cibles, initialisée dans [-0,5/n ; 0,5/n]) et `C` (contextes, initialisée à 0 comme dans word2vec). Une paire positive et ses `k` négatifs constituent un pas de descente de gradient.

Pour une paire `(w, c⁺)` et ses négatifs `c⁻ᵢ`, la perte est

$$\mathcal{L} = -\log\sigma(w\cdot c^+) - \sum_{i=1}^{k}\log\sigma(-w\cdot c^-_i)$$

et les mises à jour (taux `η`) implémentées dans `src/w2v.py` sont

$$c^+ \leftarrow c^+ - \eta\,\big(\sigma(w\cdot c^+)-1\big)\,w \qquad c^-_i \leftarrow c^-_i - \eta\,\sigma(w\cdot c^-_i)\,w$$

$$w \leftarrow w - \eta\Big[\big(\sigma(w\cdot c^+)-1\big)\,c^+ + \sum_{i=1}^{k}\sigma(w\cdot c^-_i)\,c^-_i\Big]$$

Le suivi de la perte peut être décomposé en terme positif et terme négatif (`loss_details`).

## Structure du dépôt

```
src/
  data.py        filtrage du vocabulaire, échantillons positifs/négatifs, dictionnaires, sauvegarde du dataset
  w2v.py         apprentissage (SGNS), sauvegarde des embeddings, pipeline complet, ligne de commande
  eval_w2v.py    lecture des embeddings, similarité cosinus, évaluation sur triplets (1 ou plusieurs runs)
notebooks/
  01_dataset_preperation.ipynb   génération du dataset (positifs + négatifs)
  02_experiments.ipynb           runs sur 10 graines, baseline Gensim, pistes d'amélioration
data/
  Le_comte_de_Monte_Cristo.tok            corpus tokenisé
  Le_comte_de_Monte_Cristo.100.sim.txt    100 triplets d'évaluation
```

## Reproduire

```bash
git clone https://github.com/slimani-meriem/w2vec.git && cd w2vec
pip install -r requirements.txt
mkdir -p outputs

# Entraînement (≈ 5 min sur CPU) : produit outputs/embeddings.txt
python src/w2v.py data/Le_comte_de_Monte_Cristo.tok outputs/embeddings.txt \
    -n 100 -L 2 -k 10 --eta 0.1 -e 5 --minc 5 --seed 0gi

# Évaluation
python src/eval_w2v.py outputs/embeddings.txt data/Le_comte_de_Monte_Cristo.100.sim.txt
```

La ligne de commande utilise l'initialisation actuelle de `W` (centrée, piste 1). Les notebooks sont prévus pour Google Colab : ils clonent le dépôt et enregistrent les sorties sur Drive. Le dataset (`dataset.json`, ≈ 130 Mo) n'est pas versionné : il se régénère avec le notebook 01.

## Pistes déjà explorées

| Piste | Modification | Résultat | Ce que j'en retiens |
|---|---|---|---|
| 1 | Initialisation de `W` centrée sur 0 : `(U[0,1] − 0,5) / n` au lieu de `U[0,1]` | 74,1 % → 75,5 % ; perte 2,811 → 2,797 (10 graines) | Gain modeste, dans le bruit de mesure. La variabilité entre graines semble plus faible (± 1,7 contre ± 3,5), à confirmer. |
| 2 | `η` divisé par 5 (0,02), cumulé avec la piste 1, **1 seule graine** | Perte 2,797 → 2,597 ; taux 77 % → 78 % ; avec 15 itérations : perte 2,453, taux 78 % | La perte baisse nettement sans gain mesurable sur le taux : sur ce jeu de triplets, la perte d'entraînement n'est pas un bon indicateur de la qualité. |

Observation sur la perte (graine 0, baseline) : le terme positif pèse ≈ 1,92 sur 2,81, soit environ 68 % de la perte totale. Ce déséquilibre est cohérent avec l'idée que les mots très fréquents génèrent beaucoup de paires positives peu informatives (voir perspective 3).

## Perspectives

**Objectif : amener w2v au niveau de Gensim** (84,4 % sur ce jeu), mesuré sur 10 graines et à configuration comparable. Je considérerai l'objectif atteint quand l'écart devient inférieur à la variabilité entre graines (≈ 2-3 points). Méthode : une modification à la fois, mêmes données, 10 graines, résultats consignés.

1. **Comparer à armes égales.** Relancer Gensim avec `window=2` et `negative=10`, puis le « dégrader » vers mon implémentation en désactivant ses optimisations une par une (`sample=0` pour le sous-échantillonnage, `min_alpha=alpha` pour un taux constant). Cela indique quelle optimisation explique quelle part de l'écart, avant d'en implémenter une.
2. **Taux d'apprentissage décroissant.** Décroissance linéaire de `η₀` vers ≈ 10⁻⁴ comme dans Gensim, au lieu d'un `η` constant qui laisse la perte bruitée autour d'un plateau.
3. **Sous-échantillonnage des mots fréquents** (`t ≈ 10⁻³`) : écarter une partie des paires dont la cible est un mot très fréquent (« le », « de »), pour réduire le poids des positifs peu informatifs.
4. **Renouveler les négatifs à chaque itération.** Aujourd'hui ils sont tirés une fois pour toutes dans le dataset : chaque paire revoit les mêmes 10 négatifs à chaque passage. C'est une hypothèse à tester, Gensim les tire à la volée.
5. **Hyperparamètres** : taille de fenêtre `L`, dimension `n ∈ {50, 100, 200}`, grille sur `η`, plus d'itérations avec un `η` plus faible.
6. **Initialisation de `C`** légèrement non nulle (priorité basse).

## Licence

MIT, voir [LICENSE](LICENSE).