"""
train_embeddings.py : entraîne des embeddings maison (w2v) et des embeddings Gensim (baseline)
sur le même corpus filtré, et sauvegarde chacun dans son propre fichier.

Le corpus est un fichier texte tokenisé (une phrase par ligne.
Les étapes de préparation (filtrage, exemples positifs et négatifs) utilisent les fonctions de data.py.

Usage :
    python train_embeddings.py --text corpus.txt --out-dir outputs --dim 100 --window 5 --epochs 15 --eta 0.05

Fichiers produits dans --out-dir :
    corpus_filtered.txt                        corpus après suppression des mots rares
    dataset_<corpus>_L<L>_k<k>_minc<m>_seed<s>.json   dataset (cache, relu s'il existe)
    embeddings_seed{seed}.txt                  embeddings maison
    gensim_embeddings_seed{seed}.txt           embeddings Gensim (baseline)
"""
import argparse
import os
import time

import numpy as np
from gensim.models import Word2Vec

from data import (ENCODING, filter_rare_words, generate_positive_samples,
                      generate_negative_samples, save_dataset, load_dataset)
from w2v import w2v, save_embeddings

def get_dataset(filtered_path, words_count, valid_words, args):
    """Construit le dataset (cible, positif, [négatifs]), ou le relit s'il a déjà été sauvegardé."""
    corpus_name = os.path.splitext(os.path.basename(args.text))[0]
    path = os.path.join(args.out_dir, f"dataset_{corpus_name}_L{args.window}_k{args.k}"
                                      f"_minc{args.min_count}_seed{args.data_seed}.json")
    if os.path.exists(path):
        print(f"dataset relu depuis {path}")
        return load_dataset(path)

    positives = generate_positive_samples(filtered_path, L=args.window)
    dataset = generate_negative_samples(positives, words_count, valid_words,
                                        alpha=args.alpha, k=args.k, seed=args.data_seed)
    save_dataset(dataset, path)
    print(f"dataset sauvegardé dans {path}")
    return dataset


def read_sentences(path):
    """Corpus filtré -> liste de listes de mots (format attendu par Gensim)."""
    with open(path, encoding=ENCODING) as f:
        return [line.split() for line in f if line.strip()]


def train_gensim(sentences, valid_words, args, seed):
    """Skip-gram avec negative sampling, réglé pour se rapprocher de notre implémentation."""
    model = Word2Vec(
        sentences=sentences,
        vector_size=args.dim,
        window=args.window,           # même L que generate_positive_samples
        shrink_windows=False,         # fenêtre fixe comme la nôtre
        min_count=1,                  # le corpus est déjà filtré par filter_rare_words
        sg=1,                         # skip-gram
        negative=args.k,
        ns_exponent=args.alpha,       # même lissage unigramme^alpha pour les négatifs
        alpha=args.eta,               # taux initial
        min_alpha=args.eta_min,       # taux final (décroissance linéaire)
        epochs=args.epochs,
        sample=args.gensim_sample,    # 0 = pas de sous-échantillonnage des mots fréquents
        seed=seed,
        workers=1,                    # 1 worker => résultat reproductible
    )

    # Ordre de valid_words conservé, uniquement les mots présents dans le modèle
    words = [w for w in valid_words if w in model.wv.key_to_index]
    W = np.vstack([model.wv[w] for w in words])
    return W, words


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", required=True, help="corpus tokenisé (une phrase par ligne)")
    p.add_argument("--out-dir", default="outputs", help="dossier de sortie")
    p.add_argument("--dim", type=int, default=100, help="dimension des embeddings")
    p.add_argument("--window", type=int, default=5, help="taille L du contexte de chaque côté")
    p.add_argument("--k", type=int, default=10, help="nombre de négatifs par exemple positif")
    p.add_argument("--alpha", type=float, default=0.75, help="exposant du tirage des négatifs")
    p.add_argument("--min-count", type=int, default=5, help="fréquence minimale d'un mot")
    p.add_argument("--epochs", type=int, default=15, help="nombre d'itérations")
    p.add_argument("--eta", type=float, default=0.05, help="taux d'apprentissage initial")
    p.add_argument("--eta-min", type=float, default=1e-4, help="taux final (décroissance linéaire)")
    p.add_argument("--n-runs", type=int, default=1, help="nombre de seeds d'entraînement (0 .. n-1)")
    p.add_argument("--data-seed", type=int, default=0, help="seed du tirage des négatifs (dataset)")
    p.add_argument("--gensim-sample", type=float, default=0.0, help="paramètre sample de Gensim")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- préparation des données (data.py) -----------------------------------------------
    filtered_path = os.path.join(args.out_dir, "corpus_filtered.txt")
    words_count, valid_words = filter_rare_words(args.text, filtered_path, minc=args.min_count)
    dataset = get_dataset(filtered_path, words_count, valid_words, args)
    sentences = read_sentences(filtered_path)
    print(f"{len(sentences)} phrases | {len(valid_words)} mots | {len(dataset)} exemples")

    # --- entraînements ---------------------------------------------------------------------
    for seed in range(args.n_runs):
        # embeddings maison
        t0 = time.time()
        W, losses = w2v(dataset, valid_words, n=args.dim, max_iter=args.epochs, eta=args.eta,
                        eta_min=args.eta_min, seed=seed, verbose=False)
        path = os.path.join(args.out_dir, f"embeddings_seed{seed}.txt")
        save_embeddings(W, valid_words, path)
        print(f"[maison] seed {seed} : perte finale = {losses[-1]:.3f} | {time.time() - t0:.0f} s | {path}")

        # embeddings Gensim
        t0 = time.time()
        W_gs, words_gs = train_gensim(sentences, valid_words, args, seed)
        path_gs = os.path.join(args.out_dir, f"gensim_embeddings_seed{seed}.txt")
        save_embeddings(W_gs, words_gs, path_gs)
        print(f"[gensim] seed {seed} : {len(words_gs)} mots | {time.time() - t0:.0f} s | {path_gs}")


if __name__ == "__main__":
    main()