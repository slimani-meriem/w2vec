"""
data.py : préparation des données et sampling pour word2vec (skip-gram with negative sampling).
"""
import json
from collections import Counter
import numpy as np

ENCODING = "utf-8"


# ---------------------------------------------------------------------------
# 1. Filtrage du vocabulaire
# ---------------------------------------------------------------------------
def filter_rare_words(file, out_file, minc=5, deleted_file=None):
    """
    Supprime du corpus les mots dont le nombre d'occurrences est < minc.

    :param file: chemin du corpus segmenté (un texte tokenisé, séparé par des espaces)
    :param out_file: chemin du corpus filtré à écrire
    :param minc: nombre minimal d'occurrences pour conserver un mot
    :param deleted_file: (optionnel) chemin où écrire la liste des mots supprimés
    :return: (words_count, valid_words)
             words_count : Counter des occurrences dans le corpus brut
             valid_words : liste des mots conservés
    """
    with open(file, "r", encoding=ENCODING) as f:
        lines = f.readlines()

    words_count = Counter(word for line in lines for word in line.split())

    valid_words = [w for w, c in words_count.items() if c >= minc]
    valid_set = set(valid_words)

    filtered_lines = [
        " ".join(w for w in line.split() if w in valid_set) for line in lines
    ]
    with open(out_file, "w", encoding=ENCODING) as f:
        f.write("\n".join(filtered_lines))

    if deleted_file is not None:
        invalid_words = [w for w, c in words_count.items() if c < minc]
        with open(deleted_file, "w", encoding=ENCODING) as f:
            f.write("\n".join(invalid_words))

    return words_count, valid_words


# ---------------------------------------------------------------------------
# 2. Exemples positifs
# ---------------------------------------------------------------------------
def generate_positive_samples(file, L=2):
    """
    Parcourt le corpus filtré avec une fenêtre de taille 2L+1.

    :param file: chemin du corpus filtré (une phrase par ligne)
    :param L: taille des contextes gauche et droit
    :return: liste de (mot_cible, [mots_du_contexte]), ligne par ligne
    """
    positive_samples = []

    with open(file, "r", encoding=ENCODING) as f:
        for line in f:
            sentence = line.split()
            for i, target_word in enumerate(sentence):
                start = max(0, i - L)
                end = min(len(sentence), i + L + 1)
                context_words = [sentence[j] for j in range(start, end) if j != i]
                positive_samples.append((target_word, context_words))

    return positive_samples


# ---------------------------------------------------------------------------
# 3. Exemples négatifs
# ---------------------------------------------------------------------------
def generate_negative_samples(positive_samples, words_count, valid_words,
                              alpha=0.75, k=10, seed=None):
    """
    Pour chaque paire positive (cible, contexte), tire k mots négatifs selon la
    distribution unigramme lissée P(w) ~ count(w)^alpha.
    Les négatifs sont distincts entre eux et différents de la cible et du contexte.

    :param positive_samples: sortie de generate_positive_samples
    :param words_count: Counter des occurrences (sortie de filter_rare_words)
    :param valid_words: liste des mots du vocabulaire
    :param alpha: exposant de lissage (0.75 dans word2vec)
    :param k: nombre d'exemples négatifs par exemple positif
    :param seed: graine aléatoire (reproductibilité)
    :return: liste de (mot_cible, mot_contexte_positif, [k mots négatifs])
    """
    rng = np.random.default_rng(seed)

    vocab = np.array(valid_words)
    weights = np.array([words_count[w] ** alpha for w in valid_words], dtype=float)
    probabilities = weights / weights.sum()

    flat_positives = [
        (target, ctxt) for target, contexts in positive_samples for ctxt in contexts
    ]
    N = len(flat_positives)

    # Tirage vectorisé de N x k indices, converti en mots ensuite
    sampled = vocab[rng.choice(len(vocab), size=(N, k), p=probabilities)]

    dataset = []
    for i, (target_word, ctxt_w) in enumerate(flat_positives):
        negatives = set(sampled[i].tolist())
        negatives.discard(ctxt_w)
        negatives.discard(target_word)
        while len(negatives) < k:
            new_word = valid_words[rng.choice(len(valid_words), p=probabilities)]
            if new_word != ctxt_w and new_word != target_word:
                negatives.add(new_word)
        dataset.append((target_word, ctxt_w, list(negatives)))

    return dataset


# ---------------------------------------------------------------------------
# 4. Dictionnaires mot <-> indice
# ---------------------------------------------------------------------------
def create_dictionary(vocab):
    """
    :param vocab: liste des mots du vocabulaire
    :return: (word2id, id2word)
    """
    word2id = {word: index for index, word in enumerate(vocab)}
    id2word = {index: word for index, word in enumerate(vocab)}
    return word2id, id2word


def create_dataset_dictionary(vocab, dataset):
    """
    Convertit le dataset de mots en dataset d'indices.

    :param vocab: liste des mots du vocabulaire
    :param dataset: liste de (cible, positif, [négatifs]) en mots
    :return: liste de (idx_cible, idx_positif, [idx_négatifs])
    """
    word2id, _ = create_dictionary(vocab)
    return [
        (word2id[target], word2id[pos], [word2id[w] for w in negs])
        for target, pos, negs in dataset
    ]


# ---------------------------------------------------------------------------
# 5. Sauvegarde / chargement du dataset (reproductibilité des entraînements)
# ---------------------------------------------------------------------------
def save_dataset(dataset, path):
    """Écrit le dataset (cible, positif, négatifs) au format JSON."""
    structured = [
        {"target_word": t, "positive_example": p, "neg_examples": negs}
        for t, p, negs in dataset
    ]
    with open(path, "w", encoding=ENCODING) as f:
        json.dump(structured, f, ensure_ascii=False)


def load_dataset(path):
    """Relit un dataset écrit par save_dataset."""
    with open(path, "r", encoding=ENCODING) as f:
        structured = json.load(f)
    return [(d["target_word"], d["positive_example"], d["neg_examples"])
            for d in structured]