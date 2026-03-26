import math
import torch
import torch.nn.functional as F


class MemoryCluster:
    """A node in the hierarchical memory tree.

    Leaf clusters hold raw vectors (Tensors).
    Non-leaf clusters hold child MemoryCluster objects.
    """

    __slots__ = ("centroid", "children", "is_leaf")

    def __init__(self, is_leaf=True):
        self.centroid = None          # (d_model,) tensor or None
        self.children = []            # List[Tensor] if leaf, List[MemoryCluster] if not
        self.is_leaf = is_leaf

    def recompute_centroid(self):
        """Recompute centroid as mean of all descendant leaf vectors."""
        vecs = self._collect_vectors()
        if vecs:
            self.centroid = torch.stack(vecs).mean(dim=0)

    def _collect_vectors(self):
        """Recursively collect all leaf vectors under this cluster."""
        if self.is_leaf:
            return list(self.children)
        vecs = []
        for child in self.children:
            vecs.extend(child._collect_vectors())
        return vecs

    @property
    def size(self):
        return len(self.children)


def _cosine_scores(query, vectors):
    """Cosine similarity between query (d_model,) and a list of vectors."""
    if not vectors:
        return torch.tensor([])
    stacked = torch.stack(vectors)
    return F.cosine_similarity(query.unsqueeze(0), stacked, dim=-1)


def _kmeans_cosine(vectors, k, max_iters=20):
    """Simple k-means clustering using cosine similarity."""
    n = len(vectors)
    if n <= k:
        return [[v] for v in vectors]

    stacked = torch.stack(vectors)
    indices = torch.randperm(n)[:k]
    centroids = stacked[indices].clone()

    for _ in range(max_iters):
        sims = F.cosine_similarity(
            stacked.unsqueeze(1), centroids.unsqueeze(0), dim=-1
        )
        assignments = sims.argmax(dim=1)

        new_centroids = torch.zeros_like(centroids)
        for ci in range(k):
            mask = assignments == ci
            if mask.any():
                new_centroids[ci] = stacked[mask].mean(dim=0)
            else:
                new_centroids[ci] = stacked[torch.randint(n, (1,))].squeeze(0)

        if torch.allclose(centroids, new_centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids

    sims = F.cosine_similarity(
        stacked.unsqueeze(1), centroids.unsqueeze(0), dim=-1
    )
    assignments = sims.argmax(dim=1)
    clusters = [[] for _ in range(k)]
    for i, ci in enumerate(assignments.tolist()):
        clusters[ci].append(vectors[i])
    return [c for c in clusters if c]


class HierarchicalMemory:
    """Hierarchical memory with beam search retrieval.

    Tree structure grows organically:
    - Starts as a flat list of leaf clusters in top_clusters
    - When a leaf overflows, it's split into multiple leaves via k-means
    - When top_clusters overflows, leaves are grouped into non-leaf wrappers
    - Navigation always follows the actual tree (is_leaf checks), not n_levels

    n_levels is a soft depth hint used only for _recluster_top decisions.
    """

    def __init__(self, n_levels, max_cluster_size, max_cluster_size_after,
                 beam_width, d_model):
        self.n_levels = n_levels
        self.max_cluster_size = max_cluster_size
        self.max_cluster_size_after = max_cluster_size_after
        self.beam_width = beam_width
        self.d_model = d_model
        self.top_clusters = []
        self._total_vectors = 0

    # ---- Insert ----

    def insert(self, vector):
        """Insert a vector into the nearest leaf cluster.

        Navigates by following the actual tree structure (is_leaf),
        then rebalances any overflowing nodes.
        """
        vec = vector.detach().cpu()

        if not self.top_clusters:
            cluster = MemoryCluster(is_leaf=True)
            cluster.children.append(vec)
            cluster.centroid = vec.clone()
            self.top_clusters.append(cluster)
            self._total_vectors += 1
            return

        # Navigate to nearest leaf — follow actual tree, ignore n_levels
        path = []  # [(children_list, index_in_list)]
        current_list = self.top_clusters
        while True:
            centroids = [c.centroid for c in current_list]
            scores = _cosine_scores(vec, centroids)
            idx = scores.argmax().item()
            path.append((current_list, idx))
            node = current_list[idx]
            if node.is_leaf:
                break
            current_list = node.children

        # Add vector to leaf
        leaf = path[-1][0][path[-1][1]]
        leaf.children.append(vec)
        self._total_vectors += 1

        # Recompute centroids bottom-up along the path
        for plist, pidx in reversed(path):
            plist[pidx].recompute_centroid()

        # Rebalance overflows (walks tree independently of stale path)
        self._rebalance()

    # ---- Rebalance ----

    def _rebalance(self):
        """Walk the tree bottom-up and split any overflowing nodes."""
        self._rebalance_list(self.top_clusters)
        if len(self.top_clusters) > self.max_cluster_size:
            self._recluster_top()

    def _rebalance_list(self, cluster_list):
        """Recursively rebalance clusters in a list."""
        i = 0
        while i < len(cluster_list):
            cluster = cluster_list[i]
            # Recurse into non-leaf children first (bottom-up)
            if not cluster.is_leaf:
                self._rebalance_list(cluster.children)
            # Check if this cluster overflows
            if cluster.size > self.max_cluster_size:
                self._split_cluster(cluster_list, i)
                # Don't increment — new clusters start at position i
                continue
            i += 1

    def _split_cluster(self, parent_list, idx):
        """Split an overflowing cluster into multiple sub-clusters."""
        cluster = parent_list[idx]
        k = math.ceil(cluster.size / self.max_cluster_size_after)

        if cluster.is_leaf:
            groups = _kmeans_cosine(cluster.children, k)
            new_clusters = []
            for group in groups:
                c = MemoryCluster(is_leaf=True)
                c.children = group
                c.recompute_centroid()
                new_clusters.append(c)
        else:
            child_centroids = [ch.centroid for ch in cluster.children]
            groups = _kmeans_cosine(child_centroids, k)
            centroid_to_cluster = {id(ch.centroid): ch for ch in cluster.children}
            new_clusters = []
            for group in groups:
                wrapper = MemoryCluster(is_leaf=False)
                wrapper.children = [centroid_to_cluster[id(c)] for c in group]
                wrapper.recompute_centroid()
                new_clusters.append(wrapper)

        # Replace the single cluster with new sub-clusters
        parent_list[idx:idx+1] = new_clusters

    def _recluster_top(self):
        """Group overflowing top_clusters into non-leaf wrappers."""
        k = math.ceil(len(self.top_clusters) / self.max_cluster_size_after)
        centroids = [c.centroid for c in self.top_clusters]
        groups = _kmeans_cosine(centroids, k)

        centroid_to_cluster = {id(c.centroid): c for c in self.top_clusters}

        new_top = []
        for group in groups:
            wrapper = MemoryCluster(is_leaf=False)
            wrapper.children = [centroid_to_cluster[id(cent)] for cent in group]
            wrapper.recompute_centroid()
            new_top.append(wrapper)
        self.top_clusters = new_top

    # ---- Search ----

    def search(self, query):
        """Beam search retrieval from top to bottom.

        Returns beam_width vectors most similar to query.

        Args:
            query: (d_model,) tensor

        Returns:
            dict with "vectors": (beam_width, d_model) tensor
        """
        n = self.beam_width
        device = query.device
        query_cpu = query.detach().cpu()

        if not self.top_clusters:
            return {"vectors": torch.zeros(n, self.d_model, device=device)}

        # Seed beam with top-level clusters
        beam = []
        for cluster in self.top_clusters:
            score = F.cosine_similarity(
                query_cpu.unsqueeze(0), cluster.centroid.unsqueeze(0)
            ).item()
            beam.append((score, cluster))

        beam.sort(key=lambda x: x[0], reverse=True)
        beam = beam[:self.beam_width]

        # Expand until all beam items are raw tensors
        while any(isinstance(item, MemoryCluster) for _, item in beam):
            candidates = []
            for score, item in beam:
                if isinstance(item, MemoryCluster):
                    if item.is_leaf:
                        for vec in item.children:
                            s = F.cosine_similarity(
                                query_cpu.unsqueeze(0), vec.unsqueeze(0)
                            ).item()
                            candidates.append((s, vec))
                    else:
                        for child in item.children:
                            s = F.cosine_similarity(
                                query_cpu.unsqueeze(0), child.centroid.unsqueeze(0)
                            ).item()
                            candidates.append((s, child))
                else:
                    candidates.append((score, item))

            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:max(self.beam_width, n)]

        result_vecs = [item for _, item in beam[:n] if isinstance(item, torch.Tensor)]

        # Pad if fewer than n vectors available
        while len(result_vecs) < n:
            result_vecs.append(torch.zeros(self.d_model))

        stacked = torch.stack(result_vecs).to(device)
        return {"vectors": stacked}

    # ---- Batch operations ----

    def insert_batch(self, vectors):
        """Insert a batch of vectors into memory.

        Args:
            vectors: (B, d_model) tensor
        """
        for i in range(vectors.size(0)):
            self.insert(vectors[i])

    def search_batch(self, queries):
        """Batch beam-search retrieval.

        Args:
            queries: (B, d_model) tensor

        Returns:
            (B, beam_width, d_model) tensor
        """
        results = []
        for i in range(queries.size(0)):
            result = self.search(queries[i])
            results.append(result["vectors"])
        return torch.stack(results)

    # ---- Utilities ----

    def clear(self):
        """Reset memory to empty state."""
        self.top_clusters = []
        self._total_vectors = 0

    @property
    def size(self):
        return self._total_vectors

    @property
    def n_clusters(self):
        return len(self.top_clusters)

    # ---- Serialization ----

    def state_dict(self):
        return {
            "top_clusters": [self._serialize_cluster(c) for c in self.top_clusters],
            "total_vectors": self._total_vectors,
        }

    def load_state_dict(self, state):
        self.top_clusters = [self._deserialize_cluster(d) for d in state["top_clusters"]]
        self._total_vectors = state["total_vectors"]

    @staticmethod
    def _serialize_cluster(cluster):
        data = {"is_leaf": cluster.is_leaf,
                "centroid": cluster.centroid if cluster.centroid is not None else None}
        if cluster.is_leaf:
            data["children"] = [v.clone() for v in cluster.children]
        else:
            data["children"] = [HierarchicalMemory._serialize_cluster(c) for c in cluster.children]
        return data

    @staticmethod
    def _deserialize_cluster(data):
        cluster = MemoryCluster(is_leaf=data["is_leaf"])
        cluster.centroid = data["centroid"]
        if data["is_leaf"]:
            cluster.children = list(data["children"])
        else:
            cluster.children = [HierarchicalMemory._deserialize_cluster(d) for d in data["children"]]
        return cluster
