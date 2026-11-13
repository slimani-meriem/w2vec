class KDNode:
    def __init__(self, point, k, left=None, right=None, word=None):
        self.word = word
        self.point = point # k-dimensional point dans notre cas ca sera l'embedding
        self.k = k # garde une trace de l'axe de compraison
        self.left = left
        self.right = right

def build_kd_tree(embeddings, depth=0):
    """embeddings (une liste)  matrice (Nxn) d'embeddings (n étant la dimension de l'embedding)"""

    if len(embeddings) == 0:
        return None

    n = len(embeddings[0])
    axis = depth % n
    sorted_embeddings = sorted(embeddings, key=lambda emb: emb[axis])
    median_idx = len(sorted_embeddings) // 2
    median_embedding = sorted_embeddings[median_idx]

    node = KDNode(median_embedding, axis)
    node.left = build_kd_tree(sorted_embeddings[:median_idx], depth + 1)
    node.right = build_kd_tree(sorted_embeddings[median_idx + 1:], depth + 1)

    return node

def distance_squared(emb1, emb2):
    return sum((x-y) ** 2 for x, y in zip(emb1, emb2))

def nearest_neighbor(root, target, depth=0, best=None):
    "target being the embedding for whih we want to fin the nearest neighbor"
    if root is None:
        return best

    n = len(target)
    axis = depth % n

    next_best = None
    next_sub_tree = None

    if best is None or distance_squared(target, root.point) < distance_squared(target, best.point):
        next_best = root
    else:
        next_best = best

    if target[axis] < root.point[axis]:
        next_sub_tree = root.left
        other_sub_tree = root.right
    else :
        next_sub_tree = root.right
        other_sub_tree = root.left

    best = nearest_neighbor(next_sub_tree, target, depth + 1, next_best)
    # abs(target[axis] - root.point[axis]) ** 2 : distance entre la cible et la frontière de séparation du noeud
    if distance_squared(target, best.point) > abs(target[axis] - root.point[axis]) ** 2:
        best = nearest_neighbor(other_sub_tree, target, depth + 1, best)

    return best



