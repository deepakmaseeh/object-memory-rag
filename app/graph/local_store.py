from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from app.config import Settings, get_settings
from app.graph.base import GraphStore
from app.schemas import ImageRecord, MemoryObject, Observation, Scene
from app.schemas.identity import ProductSignature


class LocalGraphStore(GraphStore):
    """
    File-backed graph memory used when Neo4j is unavailable (no Docker).
    Implements the same GraphStore contract for local development and E2E verification.
    """

    def __init__(self, settings: Optional[Settings] = None, path: Optional[Path] = None) -> None:
        self.settings = settings or get_settings()
        self.path = path or (
            self.settings.resolve_path(self.settings.paths.storage) / "local_graph.json"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "images": {},
            "objects": {},
            "observations": {},
            "scenes": {},
            "clusters": {},
            "locations": {},
            "attributes": {},
            "product_signatures": {},
            "links": {
                "observed_as": [],  # (object_id, observation_id)
                "from_image": [],
                "in_scene": [],
                "member_of": [],
                "located_at": [],
                "has_attribute": [],
                "instance_of": [],  # (object_id, product_signature_id)
            },
        }

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")

    def ensure_schema(self) -> None:
        with self._lock:
            scene = Scene(
                scene_id=self.settings.default_scene.scene_id,
                name=self.settings.default_scene.name,
            )
            self.ensure_scene(scene)
            self._save()

    def upsert_image(self, image: ImageRecord) -> None:
        with self._lock:
            self._data["images"][image.image_id] = image.model_dump(mode="json")
            self._save()

    def upsert_object(self, obj: MemoryObject) -> None:
        with self._lock:
            existing = self._data["objects"].get(obj.object_id, {})
            count = int(existing.get("observation_count") or 0)
            data = obj.model_dump(mode="json")
            if existing:
                data["created_at"] = existing.get("created_at") or data["created_at"]
                data["observation_count"] = count
                # Merge attributes (new keys win)
                old_attrs = existing.get("attributes") or {}
                new_attrs = data.get("attributes") or {}
                if isinstance(old_attrs, dict) and isinstance(new_attrs, dict):
                    data["attributes"] = {**old_attrs, **new_attrs}
            else:
                data["observation_count"] = 0
            self._data["objects"][obj.object_id] = data
            self._save()

    def set_object_attributes(self, object_id: str, attributes: dict[str, Any]) -> None:
        """Persist attribute map on object and HAS_ATTRIBUTE-style links."""
        with self._lock:
            obj = self._data["objects"].setdefault(
                object_id, {"object_id": object_id, "attributes": {}}
            )
            attrs = dict(obj.get("attributes") or {})
            attrs.update({str(k): v for k, v in (attributes or {}).items()})
            obj["attributes"] = attrs
            for key, value in attrs.items():
                name = f"{key}:{value}"
                self._data["attributes"][name] = {
                    "name": str(key),
                    "value": value,
                }
                pair = [object_id, name]
                if pair not in self._data["links"]["has_attribute"]:
                    self._data["links"]["has_attribute"].append(pair)
            self._save()

    def upsert_product_signature(self, product: ProductSignature) -> None:
        with self._lock:
            if "product_signatures" not in self._data:
                self._data["product_signatures"] = {}
            self._data["product_signatures"][product.product_signature_id] = (
                product.model_dump(mode="json")
            )
            self._save()

    def link_object_to_product(self, object_id: str, product_signature_id: str) -> None:
        with self._lock:
            if object_id in self._data["objects"]:
                self._data["objects"][object_id]["product_signature_id"] = (
                    product_signature_id
                )
            pair = [object_id, product_signature_id]
            links = self._data["links"]
            if "instance_of" not in links:
                links["instance_of"] = []
            if pair not in links["instance_of"]:
                links["instance_of"].append(pair)
            self._save()

    def get_product_signature(self, product_signature_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            ps = (self._data.get("product_signatures") or {}).get(product_signature_id)
            return dict(ps) if ps else None

    def list_objects_for_product(self, product_signature_id: str) -> list[dict[str, Any]]:
        with self._lock:
            oids = [
                o
                for o, p in self._data["links"].get("instance_of", [])
                if p == product_signature_id
            ]
            return [dict(self._data["objects"][oid]) for oid in oids if oid in self._data["objects"]]

    def create_observation(
        self,
        observation: Observation,
        object_id: str,
        scene_id: Optional[str] = None,
        location_name: Optional[str] = None,
    ) -> None:
        scene_id = scene_id or observation.scene_id or self.settings.default_scene.scene_id
        with self._lock:
            obs = observation.model_dump(mode="json")
            obs["object_id"] = object_id
            obs["scene_id"] = scene_id
            self._data["observations"][observation.observation_id] = obs
            if object_id not in self._data["objects"]:
                self._data["objects"][object_id] = {
                    "object_id": object_id,
                    "class_id": observation.class_id,
                    "class_name": observation.class_name,
                    "observation_count": 0,
                    "last_seen": obs["timestamp"],
                }
            obj = self._data["objects"][object_id]
            obj["last_seen"] = obs["timestamp"]
            obj["observation_count"] = int(obj.get("observation_count") or 0) + 1
            obj["class_name"] = observation.class_name
            obj["class_id"] = observation.class_id
            links = self._data["links"]
            pair = [object_id, observation.observation_id]
            if pair not in links["observed_as"]:
                links["observed_as"].append(pair)
            img_pair = [observation.observation_id, observation.image_id]
            if img_pair not in links["from_image"]:
                links["from_image"].append(img_pair)
            scene_pair = [observation.observation_id, scene_id]
            if scene_pair not in links["in_scene"]:
                links["in_scene"].append(scene_pair)
            if scene_id not in self._data["scenes"]:
                self._data["scenes"][scene_id] = {"scene_id": scene_id, "name": scene_id}
            if location_name:
                self._data["locations"][location_name] = {"name": location_name}
                loc_pair = [object_id, location_name]
                if loc_pair not in links["located_at"]:
                    links["located_at"].append(loc_pair)
            self._save()

    def ensure_scene(self, scene: Scene) -> None:
        with self._lock:
            self._data["scenes"][scene.scene_id] = scene.model_dump(mode="json")
            self._save()

    def link_object_to_cluster(self, object_id: str, cluster_id: str) -> None:
        with self._lock:
            if object_id in self._data["objects"]:
                self._data["objects"][object_id]["cluster_id"] = cluster_id
            pair = [object_id, cluster_id]
            if pair not in self._data["links"]["member_of"]:
                self._data["links"]["member_of"].append(pair)
            self._save()

    def upsert_cluster_node(
        self,
        cluster_id: str,
        name: str,
        class_name: Optional[str] = None,
        object_count: int = 0,
    ) -> None:
        with self._lock:
            self._data["clusters"][cluster_id] = {
                "cluster_id": cluster_id,
                "name": name,
                "class_name": class_name,
                "object_count": object_count,
            }
            self._save()

    def get_object(self, object_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            obj = self._data["objects"].get(object_id)
            return dict(obj) if obj else None

    def get_object_history(self, object_id: str) -> dict[str, Any]:
        with self._lock:
            obj = self._data["objects"].get(object_id)
            if not obj:
                return {}
            obs_ids = [
                oid for src, oid in self._data["links"]["observed_as"] if src == object_id
            ]
            observations = []
            for oid in obs_ids:
                o = self._data["observations"].get(oid)
                if not o:
                    continue
                scene_id = o.get("scene_id")
                scene_name = (self._data["scenes"].get(scene_id) or {}).get("name")
                observations.append({**o, "scene_name": scene_name})
            observations.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
            locations = [
                loc for oid, loc in self._data["links"]["located_at"] if oid == object_id
            ]
            clusters = [
                cid for oid, cid in self._data["links"]["member_of"] if oid == object_id
            ]
            return {
                "object": dict(obj),
                "observations": observations,
                "locations": locations,
                "clusters": clusters,
                "attributes": dict(obj.get("attributes") or {}),
            }

    def latest_observation(self, object_id: str) -> Optional[dict[str, Any]]:
        history = self.get_object_history(object_id)
        obs = history.get("observations") or []
        return obs[0] if obs else None

    def search_objects_by_class(self, class_name: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = []
            for obj in self._data["objects"].values():
                if str(obj.get("class_name", "")).lower() != class_name.lower():
                    continue
                latest = None
                for src, oid in self._data["links"]["observed_as"]:
                    if src == obj["object_id"]:
                        o = self._data["observations"].get(oid)
                        if o and (
                            latest is None
                            or (o.get("timestamp") or "") > (latest.get("timestamp") or "")
                        ):
                            latest = o
                scene_name = None
                if latest and latest.get("scene_id"):
                    scene_name = (self._data["scenes"].get(latest["scene_id"]) or {}).get(
                        "name"
                    )
                rows.append(
                    {
                        "object": dict(obj),
                        "last_seen": obj.get("last_seen") or (latest or {}).get("timestamp"),
                        "last_scene": scene_name,
                    }
                )
            rows.sort(key=lambda r: r.get("last_seen") or "", reverse=True)
            return rows[:limit]

    def list_objects(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = []
            for obj in self._data["objects"].values():
                oid = obj.get("object_id")
                history = {
                    "object": dict(obj),
                    "observation_count": int(obj.get("observation_count") or 0),
                    "locations": [
                        loc for o, loc in self._data["links"]["located_at"] if o == oid
                    ],
                    "clusters": [
                        c for o, c in self._data["links"]["member_of"] if o == oid
                    ],
                }
                rows.append(history)
            rows.sort(
                key=lambda r: (r["object"].get("last_seen") or r["object"].get("created_at") or ""),
                reverse=True,
            )
            return rows[:limit]

    def list_clusters(self) -> list[dict[str, Any]]:
        with self._lock:
            out = []
            for cid, c in self._data["clusters"].items():
                oids = [o for o, cl in self._data["links"]["member_of"] if cl == cid]
                out.append(
                    {
                        "cluster_id": cid,
                        "name": c.get("name") or cid,
                        "class_name": c.get("class_name"),
                        "object_count": c.get("object_count") or len(oids),
                        "object_ids": oids,
                    }
                )
            return out

    def export_graph(self, object_id: Optional[str] = None, limit: int = 200) -> dict[str, Any]:
        """Export nodes/edges for UI visualization."""
        with self._lock:
            nodes: list[dict[str, Any]] = []
            edges: list[dict[str, Any]] = []
            seen_n: set[str] = set()

            def add_node(nid: str, ntype: str, label: str, data: Optional[dict] = None):
                if nid in seen_n:
                    return
                seen_n.add(nid)
                nodes.append(
                    {
                        "id": nid,
                        "type": ntype,
                        "label": label,
                        "data": data or {},
                    }
                )

            def add_edge(src: str, tgt: str, etype: str):
                edges.append(
                    {
                        "id": f"{src}-{etype}-{tgt}",
                        "source": src,
                        "target": tgt,
                        "type": etype,
                    }
                )

            object_ids = (
                [object_id]
                if object_id
                else list(self._data["objects"].keys())[:limit]
            )
            for oid in object_ids:
                obj = self._data["objects"].get(oid)
                if not obj:
                    continue
                add_node(f"obj:{oid}", "Object", oid, obj)
                for src, obs_id in self._data["links"]["observed_as"]:
                    if src != oid:
                        continue
                    obs = self._data["observations"].get(obs_id) or {}
                    add_node(f"obs:{obs_id}", "Observation", obs_id, obs)
                    add_edge(f"obj:{oid}", f"obs:{obs_id}", "OBSERVED_AS")
                    img_id = obs.get("image_id")
                    if img_id:
                        img = self._data["images"].get(img_id) or {"image_id": img_id}
                        add_node(f"img:{img_id}", "Image", img_id, img)
                        add_edge(f"obs:{obs_id}", f"img:{img_id}", "FROM_IMAGE")
                    scene_id = obs.get("scene_id")
                    if scene_id:
                        sc = self._data["scenes"].get(scene_id) or {
                            "scene_id": scene_id,
                            "name": scene_id,
                        }
                        add_node(f"scene:{scene_id}", "Scene", sc.get("name") or scene_id, sc)
                        add_edge(f"obs:{obs_id}", f"scene:{scene_id}", "IN_SCENE")
                for o, cl in self._data["links"]["member_of"]:
                    if o != oid:
                        continue
                    c = self._data["clusters"].get(cl) or {"cluster_id": cl}
                    add_node(f"cluster:{cl}", "Cluster", cl, c)
                    add_edge(f"obj:{oid}", f"cluster:{cl}", "MEMBER_OF")
                for o, loc in self._data["links"]["located_at"]:
                    if o != oid:
                        continue
                    add_node(f"loc:{loc}", "Location", loc, {"name": loc})
                    add_edge(f"obj:{oid}", f"loc:{loc}", "AT_LOCATION")
                for o, attr_name in self._data["links"].get("has_attribute", []):
                    if o != oid:
                        continue
                    attr = self._data["attributes"].get(attr_name) or {"name": attr_name}
                    add_node(
                        f"attr:{attr_name}",
                        "Attribute",
                        attr_name,
                        attr if isinstance(attr, dict) else {"name": attr_name},
                    )
                    add_edge(f"obj:{oid}", f"attr:{attr_name}", "HAS_ATTRIBUTE")
                pid = obj.get("product_signature_id")
                if pid:
                    ps = (self._data.get("product_signatures") or {}).get(pid) or {
                        "product_signature_id": pid
                    }
                    add_node(f"product:{pid}", "ProductSignature", pid, ps)
                    add_edge(f"obj:{oid}", f"product:{pid}", "INSTANCE_OF")
            return {"nodes": nodes, "edges": edges}

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "objects": len(self._data["objects"]),
                "observations": len(self._data["observations"]),
                "clusters": len(self._data["clusters"]),
                "images": len(self._data["images"]),
                "scenes": len(self._data["scenes"]),
            }

    def health(self) -> bool:
        return True

    def close(self) -> None:
        with self._lock:
            self._save()
