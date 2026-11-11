"""
w2v.py : apprentissage d'embeddings (skip-gram with negative sampling).

Utilisable de deux façons :
  - en script :   python src/w2v.py corpus.tok out.txt -n 100 -L 2 -k 10 --eta 0.1 -e 5 --minc 5
  - en module :   from w2v import w2v, save_embeddings   (depuis un notebook)

"""
import argparse
import numpy as np

from data import (
    ENCODING,
    create_dataset_dictionary,
    create_dictionary,
    filter_rare_words,
    generate_negative_samples,
    generate_positive_samples,
    load_dataset,
    save_dataset,
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



# ---------------------------------------------------------------------------
# 1. Apprentissage
# ---------------------------------------------------------------------------
def w2v(dataset, vocab, n=100, max_iter=5, eta=0.1, seed=None,
        shuffle=True, verbose=True,  loss_details=None, eval_fn=None):
    """
    Apprend les embeddings par descente de gradient (un exemple positif et ses
    k négatifs = un mini-batch).

    :param dataset: liste de (mot_cible, mot_positif, [mots_négatifs])
    :param vocab: liste des mots pour lesquels on apprend un embedding
    :param n: dimension des embeddings
    :param max_iter: nombre d'itérations
    :param eta: taux d'apprentissage
    :param seed: graine aléatoire (initialisation + mélange)
    :param shuffle: mélange l'ordre des exemples à chaque itération
    :param verbose: affiche la perte moyenne à chaque itération
    :param loss_details : dictionnaire optionnel, rempli avec le détail de la perte
    :return: (W, losses) où W est la matrice des embeddings des mots cibles
             (taille len(vocab) x n) et losses la perte moyenne par itération
    """
    rng = np.random.default_rng(seed)

    N = len(vocab)
    # --------------------------------
    # 1ere methode d'initialisation :
    # W = rng.random((N, n)) # valeurs positives dans  [0,1] avec une moyenne de 0.5
    # --------------------------------

    W = (rng.random((N, n)) - 0.5) / n # les valeurs sont comprises entre -0,5/n et +0,5/n, centrées sur 0.
    C = np.zeros((N, n))                 # contextes initialisés à 0 (comme word2vec)

    dataset_indices = create_dataset_dictionary(vocab, dataset)
    order = np.arange(len(dataset_indices))

    losses = []
    losses_pos, losses_neg = [], []      # perte moyenne du terme positif et du terme négatif, par itération
    eval_scores = [] # pour garder une trace des scores d'évaluation au cours des itérations

    for it in range(max_iter):
        if shuffle:
            rng.shuffle(order)

        total_loss = 0.0
        total_pos, total_neg = 0.0, 0.0  # cumuls des deux termes pour cette itération
        for idx in order:
            w_idx, c_pos_idx, c_neg_indices = dataset_indices[idx]

            emb_w = W[w_idx]
            emb_c_pos = C[c_pos_idx]
            embs_c_neg = C[c_neg_indices]            # (k, n)

            # sigma(m.c_pos) - 1   et   sigma(m.c_neg)
            err_c_pos = sigmoid(np.dot(emb_w, emb_c_pos)) - 1.0
            err_c_neg = sigmoid(np.dot(embs_c_neg, emb_w))   # (k,)

            # Perte, réutiliser les sigmoïdes déjà calculées
            total_loss -= (np.log(err_c_pos + 1.0 + 1e-10)
                           + np.sum(np.log(1.0 - err_c_neg + 1e-10))) # les 1e-10 est pour éviter les log(0)

            if loss_details is not None:
                total_pos -= np.log(err_c_pos + 1.0 + 1e-10)
                total_neg -= np.sum(np.log(1.0 - err_c_neg + 1e-10))

            # Gradient par rapport au mot cible
            grad_w = err_c_pos * emb_c_pos + np.dot(err_c_neg, embs_c_neg)

            # Mises à jour
            C[c_pos_idx] -= eta * err_c_pos * emb_w
            C[c_neg_indices] -= eta * err_c_neg[:, None] * emb_w
            W[w_idx] -= eta * grad_w

        mean_loss = total_loss / len(dataset_indices)
        losses.append(mean_loss)
        if loss_details is not None:
            losses_pos.append(total_pos / len(dataset_indices))
            losses_neg.append(total_neg / len(dataset_indices))
        if eval_fn is not None:
            score = eval_fn(W)
            eval_scores.append(score)
        if verbose:
            print(f"Itération {it + 1}/{max_iter} - perte moyenne : {mean_loss:.4f}")

    if loss_details is not None:
        loss_details["pos"] = losses_pos
        loss_details["neg"] = losses_neg

    if eval_fn is not None:
        return W, losses, eval_scores

    return W, losses


# ---------------------------------------------------------------------------
# 3. Sauvegarde des embeddings
# ---------------------------------------------------------------------------
def save_embeddings(W, vocab, filename, precision=4):
    """
    Écrit les embeddings : première ligne "N d", puis "mot v1 v2 ... vd".

    :param W: matrice des embeddings (N x d)
    :param vocab: liste des mots, dans le même ordre que les lignes de W
    :param filename: chemin exact du fichier à écrire
    :param precision: nombre de décimales conservées
    """
    num_words, dim = W.shape
    with open(filename, "w", encoding=ENCODING) as f:
        f.write(f"{num_words} {dim}\n")
        for word, vec in zip(vocab, W):
            f.write(word + " " + " ".join(f"{x:.{precision}f}" for x in vec) + "\n")


# ---------------------------------------------------------------------------
# 4. Pipeline complet (pour le script et réutilisable depuis un notebook)
# ---------------------------------------------------------------------------
def train_from_corpus(corpus, output, n=100, L=2, k=10, eta=0.1, e=5, minc=5,
                      alpha=0.75, seed=None, filtered_file=None,
                      dataset_file=None, save_dataset_file=None, verbose=True):
    """
    Corpus brut -> fichier d'embeddings.

    :param corpus: chemin du corpus segmenté
    :param output: chemin du fichier d'embeddings produit
    :param filtered_file: chemin du corpus filtré (par défaut output + ".filtered.txt")
    :param dataset_file: si fourni, recharge un dataset JSON existant au lieu d'en
                         générer un (pour comparer des runs sur les mêmes données)
    :param save_dataset_file: si fourni, enregistre le dataset généré en JSON
    :return: (W, valid_words, losses)
    """
    filtered_file = filtered_file or output + ".filtered.txt"
    words_count, valid_words = filter_rare_words(corpus, filtered_file, minc=minc)

    if dataset_file is not None:
        dataset = load_dataset(dataset_file)
    else:
        positives = generate_positive_samples(filtered_file, L=L)
        dataset = generate_negative_samples(positives, words_count, valid_words,
                                            alpha=alpha, k=k, seed=seed)
        if save_dataset_file is not None:
            save_dataset(dataset, save_dataset_file)

    W, losses = w2v(dataset, valid_words, n=n, max_iter=e, eta=eta,
                    seed=seed, verbose=verbose)
    save_embeddings(W, valid_words, output)
    return W, valid_words, losses


# ---------------------------------------------------------------------------
# 5. Ligne de commande
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Entraîne des embeddings word2vec (SGNS).")
    p.add_argument("corpus", help="corpus segmenté (.tok)")
    p.add_argument("output", help="fichier d'embeddings à produire")
    p.add_argument("-n", type=int, default=100, help="dimension des embeddings")
    p.add_argument("-L", type=int, default=2, help="taille du contexte gauche/droit")
    p.add_argument("-k", type=int, default=10, help="négatifs par positif")
    p.add_argument("--eta", type=float, default=0.1, help="taux d'apprentissage")
    p.add_argument("-e", type=int, default=5, help="nombre d'itérations")
    p.add_argument("--minc", type=int, default=5, help="occurrences minimales d'un mot")
    p.add_argument("--alpha", type=float, default=0.75, help="lissage du tirage négatif")
    p.add_argument("--seed", type=int, default=None, help="graine aléatoire")
    p.add_argument("--filtered", default=None, help="chemin du corpus filtré")
    p.add_argument("--dataset", default=None, help="dataset JSON existant à réutiliser")
    p.add_argument("--save-dataset", default=None, help="où enregistrer le dataset généré")
    a = p.parse_args()

    train_from_corpus(a.corpus, a.output, n=a.n, L=a.L, k=a.k, eta=a.eta, e=a.e,
                      minc=a.minc, alpha=a.alpha, seed=a.seed,
                      filtered_file=a.filtered, dataset_file=a.dataset,
                      save_dataset_file=a.save_dataset)


if __name__ == "__main__":
    main()