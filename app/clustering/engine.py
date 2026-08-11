from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from app.config import Settings, get_settings
from app.graph.base import GraphStore
from app.memory.base import VectorStore
from app.memory.centroid_index import InMemoryCentroidIndex
from app.schemas import Cluster


class ClusterEngine:
    """Per-class visual clustering with centroid indexes in Qdrant + Neo4j + RAM."""

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: Optional[GraphStore] = None,
        settings: Optional[Settings] = None,
        centroid_index: Optional[InMemoryCentroidIndex] = None,
    ) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.settings = settings or get_settings()
        self.centroid_index = centroid_index or InMemoryCentroidIndex(self.settings)

    def rebuild_for_class(self, class_name: str) -> list[Cluster]:
        from sklearn.cluster import AgglomerativeClustering

        points = self.vector_store.list_observations_for_class(class_name)
        if not points:
            return []

        vectors = []
        object_ids = []
        for p in points:
            vec = p.get("vector") or []
            if not vec:
                continue
            vectors.append(vec)
            oid = (p.get("payload") or {}).get("object_id")
            object_ids.append(oid)

        if len(vectors) < self.settings.clustering.min_samples_for_rebuild:
            # Single class-level cluster
            centroid = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            unique_objects = sorted({o for o in object_ids if o})
            cluster_id = f"cluster_{class_name}_0"
            cluster = Cluster(
                cluster_id=cluster_id,
                name=f"{class_name}_0",
                class_name=class_name,
                object_count=len(unique_objects),
                object_ids=unique_objects,
                centroid=centroid.tolist(),
            )
            self._persist([cluster])
            return [cluster]

        n = min(
            self.settings.clustering.n_clusters_per_class,
            len(vectors),
        )
        n = max(1, n)
        X = np.asarray(vectors, dtype=np.float32)
        # Normalize rows
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms

        if n == 1:
            labels = np.zeros(len(X), dtype=int)
        else:
            model = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
            labels = model.fit_predict(X)

        by_label: dict[int, list[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            by_label[int(lab)].append(idx)

        clusters: list[Cluster] = []
        for lab, indices in sorted(by_label.items()):
            members = [object_ids[i] for i in indices if object_ids[i]]
            unique_objects = sorted(set(members))
            centroid = np.mean(X[indices], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            cluster_id = f"cluster_{class_name}_{lab}"
            clusters.append(
                Cluster(
                    cluster_id=cluster_id,
                    name=f"{class_name}_{lab}",
                    class_name=class_name,
                    object_count=len(unique_objects),
                    object_ids=unique_objects,
                    centroid=centroid.tolist(),
                )
            )
        self._persist(clusters)
        return clusters

    def rebuild_all(self, class_names: list[str]) -> dict[str, list[Cluster]]:
        return {c: self.rebuild_for_class(c) for c in class_names}

    def _persist(self, clusters: list[Cluster]) -> None:
        for c in clusters:
            if not c.centroid:
                continue
            payload = {
                "cluster_id": c.cluster_id,
                "name": c.name,
                "class_name": c.class_name,
                "object_ids": c.object_ids,
                "object_count": c.object_count,
            }
            self.vector_store.upsert_cluster(
                cluster_id=c.cluster_id,
                vector=c.centroid,
                payload=payload,
            )
            self.centroid_index.upsert_cluster_model(c)
            if self.graph_store:
                self.graph_store.upsert_cluster_node(
                    cluster_id=c.cluster_id,
                    name=c.name,
                    class_name=c.class_name,
                    object_count=c.object_count,
                )
                for oid in c.object_ids:
                    self.graph_store.link_object_to_cluster(oid, c.cluster_id)
