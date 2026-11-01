"""
eval_w2v.py : évaluation d'embeddings sur des triplets (m, m+, m-).

Un triplet est réussi si cos(m, m+) > cos(m, m-).

Utilisable de deux façons :
  - en script :   python src/eval_w2v.py embeddings.txt triplets.sim
  - en module :   from eval_w2v import get_embeddings, evaluate   (depuis un notebook)

"""
import argparse

import numpy as np

ENCODING = "utf-8"


# ---------------------------------------------------------------------------
# 1. Lecture des embeddings (format de la section 5 du sujet)
# ---------------------------------------------------------------------------
def get_embeddings(path):
    """
    Lit un fichier d'embeddings : première ligne "N d", puis "mot v1 ... vd".

    :param path: chemin du fichier d'embeddings
    :return: dictionnaire {mot: vecteur numpy de taille d}
    """
    embeddings = {}
    with open(path, "r", encoding=ENCODING) as f:
        next(f)  # ligne d'en-tête "N d"
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            embeddings[parts[0]] = np.array(parts[1:], dtype=float)
    return embeddings


# ---------------------------------------------------------------------------
# 2. Similarité
# ---------------------------------------------------------------------------
def cosinus_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# 3. Évaluation
# ---------------------------------------------------------------------------
def evaluate(embeddings, eval_file, skip_oov=False):
    """
    Calcule le taux de triplets (m, m+, m-) on voudrait que cos(m, m+) > cos(m, m-).

    :param embeddings: dictionnaire {mot: vecteur} (sortie de get_embeddings)
    :param eval_file: chemin du fichier de triplets (trois mots par ligne)
    :param skip_oov: si False, un triplet contenant un mot absent des embeddings
                     compte comme un échec ; si True, il est ignoré du calcul
    :return: dictionnaire avec les clés
             "rate"    : taux de réussite
             "correct" : nombre de triplets réussis
             "total"   : nombre de triplets pris en compte dans le taux
             "oov"     : nombre de triplets contenant un mot inconnu
    """
    correct, total, oov = 0, 0, 0

    with open(eval_file, "r", encoding=ENCODING) as f:
        for line in f:
            words = line.split()
            if len(words) != 3:
                continue  # ligne vide ou mal formée
            m, m_pos, m_neg = words

            if not all(w in embeddings for w in words):
                oov += 1
                if not skip_oov:
                    total += 1   # compté comme un échec
                continue

            total += 1
            sim_pos = cosinus_similarity(embeddings[m], embeddings[m_pos])
            sim_neg = cosinus_similarity(embeddings[m], embeddings[m_neg])
            if sim_pos > sim_neg:
                correct += 1

    rate = correct / total if total > 0 else 0.0
    return {"rate": rate, "correct": correct, "total": total, "oov": oov}


def evaluate_runs(embedding_files, eval_file, skip_oov=False):
    """
    Évalue plusieurs fichiers d'embeddings (par exemple 10 runs avec des graines
    différentes) et renvoie la moyenne et l'écart type des taux de réussite.

    :param embedding_files: liste de chemins de fichiers d'embeddings
    :param eval_file: chemin du fichier de triplets
    :return: (rates, mean, std)
    """
    rates = [evaluate(get_embeddings(p), eval_file, skip_oov)["rate"]
             for p in embedding_files]
    return rates, float(np.mean(rates)), float(np.std(rates))


# ---------------------------------------------------------------------------
# 4. Ligne de commande
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Évalue des embeddings sur des triplets.")
    p.add_argument("embeddings", help="fichier d'embeddings (format de la section 5)")
    p.add_argument("eval_file", help="fichier de triplets (m m+ m-)")
    p.add_argument("--skip-oov", action="store_true",
                   help="ignorer les triplets contenant un mot inconnu "
                        "(par défaut ils comptent comme des échecs)")
    a = p.parse_args()

    res = evaluate(get_embeddings(a.embeddings), a.eval_file, skip_oov=a.skip_oov)
    print(f"Taux de réussite : {res['rate']:.3f} "
          f"({res['correct']}/{res['total']} triplets, {res['oov']} avec mot inconnu)")


if __name__ == "__main__":
    main()