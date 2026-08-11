from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from neo4j import GraphDatabase

from app.config import Settings, get_settings
from app.graph.base import GraphStore
from app.schemas import ImageRecord, MemoryObject, Observation, Scene


class Neo4jGraphStore(GraphStore):
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._driver = GraphDatabase.driver(
            self.settings.neo4j.uri,
            auth=(self.settings.neo4j.user, self.settings.neo4j.password),
            connection_timeout=3.0,
            max_connection_lifetime=30,
        )

    def ensure_schema(self) -> None:
        statements = self._load_cypher_statements()
        with self._driver.session() as session:
            for stmt in statements:
                session.run(stmt)
            # Default scene
            scene = Scene(
                scene_id=self.settings.default_scene.scene_id,
                name=self.settings.default_scene.name,
            )
            self.ensure_scene(scene)

    def _load_cypher_statements(self) -> list[str]:
        path = self.settings.root_dir / "docker" / "init-neo4j.cypher"
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        statements: list[str] = []
        buf: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            buf.append(line)
            if stripped.endswith(";"):
                statements.append("\n".join(buf).rstrip(";").strip())
                buf = []
        if buf:
            statements.append("\n".join(buf).strip())
        return statements

    def upsert_image(self, image: ImageRecord) -> None:
        query = """
        MERGE (i:Image {image_id: $image_id})
        SET i.original_path = $original_path,
            i.width = $width,
            i.height = $height,
            i.timestamp = $timestamp,
            i.content_type = $content_type
        """
        with self._driver.session() as session:
            session.run(
                query,
                image_id=image.image_id,
                original_path=image.original_path,
                width=image.width,
                height=image.height,
                timestamp=image.timestamp.isoformat(),
                content_type=image.content_type,
            )

    def upsert_object(self, obj: MemoryObject) -> None:
        # Do not overwrite observation_count with a stale client value;
        # create_observation increments it authoritatively.
        query = """
        MERGE (o:Object {object_id: $object_id})
        ON CREATE SET o.created_at = $created_at,
                      o.observation_count = 0
        SET o.class_id = $class_id,
            o.class_name = $class_name,
            o.last_seen = $last_seen,
            o.cluster_id = $cluster_id,
            o.attributes = $attributes
        """
        import json

        with self._driver.session() as session:
            session.run(
                query,
                object_id=obj.object_id,
                class_id=obj.class_id,
                class_name=obj.class_name,
                created_at=obj.created_at.isoformat(),
                last_seen=obj.last_seen.isoformat(),
                cluster_id=obj.cluster_id,
                attributes=json.dumps(obj.attributes or {}),
            )

    def set_object_attributes(self, object_id: str, attributes: dict) -> None:
        import json

        query = """
        MATCH (o:Object {object_id: $object_id})
        SET o.attributes = $attributes
        WITH o
        UNWIND $pairs AS pair
        MERGE (a:Attribute {name: pair.key, value: toString(pair.value)})
        MERGE (o)-[:HAS_ATTRIBUTE]->(a)
        """
        pairs = [{"key": str(k), "value": v} for k, v in (attributes or {}).items()]
        with self._driver.session() as session:
            session.run(
                query,
                object_id=object_id,
                attributes=json.dumps(attributes or {}),
                pairs=pairs,
            )

    def ensure_scene(self, scene: Scene) -> None:
        query = """
        MERGE (s:Scene {scene_id: $scene_id})
        SET s.name = $name,
            s.description = $description,
            s.created_at = coalesce(s.created_at, $created_at)
        """
        with self._driver.session() as session:
            session.run(
                query,
                scene_id=scene.scene_id,
                name=scene.name,
                description=scene.description,
                created_at=scene.created_at.isoformat(),
            )

    def create_observation(
        self,
        observation: Observation,
        object_id: str,
        scene_id: Optional[str] = None,
        location_name: Optional[str] = None,
    ) -> None:
        scene_id = scene_id or observation.scene_id or self.settings.default_scene.scene_id
        query = """
        MERGE (obj:Object {object_id: $object_id})
        MERGE (obs:Observation {observation_id: $observation_id})
        SET obs.image_id = $image_id,
            obs.class_id = $class_id,
            obs.class_name = $class_name,
            obs.bbox = $bbox,
            obs.confidence = $confidence,
            obs.timestamp = $timestamp,
            obs.mask_path = $mask_path,
            obs.crop_path = $crop_path,
            obs.object_id = $object_id,
            obs.scene_id = $scene_id
        MERGE (obj)-[:OBSERVED_AS]->(obs)
        WITH obj, obs
        MERGE (img:Image {image_id: $image_id})
        MERGE (obs)-[:FROM_IMAGE]->(img)
        WITH obj, obs
        MERGE (s:Scene {scene_id: $scene_id})
        ON CREATE SET s.name = $scene_id, s.created_at = $timestamp
        MERGE (obs)-[:IN_SCENE]->(s)
        WITH obj, s
        SET obj.last_seen = $timestamp,
            obj.observation_count = coalesce(obj.observation_count, 0) + 1
        """
        params = {
            "object_id": object_id,
            "observation_id": observation.observation_id,
            "image_id": observation.image_id,
            "class_id": observation.class_id,
            "class_name": observation.class_name,
            "bbox": observation.bbox.as_list(),
            "confidence": observation.confidence,
            "timestamp": observation.timestamp.isoformat(),
            "mask_path": observation.mask_path,
            "crop_path": observation.crop_path,
            "scene_id": scene_id,
        }
        with self._driver.session() as session:
            session.run(query, **params)
            if location_name:
                session.run(
                    """
                    MATCH (obj:Object {object_id: $object_id})
                    MERGE (l:Location {name: $location_name})
                    MERGE (obj)-[:LOCATED_AT]->(l)
                    """,
                    object_id=object_id,
                    location_name=location_name,
                )

    def link_object_to_cluster(self, object_id: str, cluster_id: str) -> None:
        query = """
        MATCH (o:Object {object_id: $object_id})
        MERGE (c:Cluster {cluster_id: $cluster_id})
        MERGE (o)-[:MEMBER_OF]->(c)
        SET o.cluster_id = $cluster_id
        """
        with self._driver.session() as session:
            session.run(query, object_id=object_id, cluster_id=cluster_id)

    def upsert_cluster_node(
        self,
        cluster_id: str,
        name: str,
        class_name: Optional[str] = None,
        object_count: int = 0,
    ) -> None:
        query = """
        MERGE (c:Cluster {cluster_id: $cluster_id})
        SET c.name = $name,
            c.class_name = $class_name,
            c.object_count = $object_count
        """
        with self._driver.session() as session:
            session.run(
                query,
                cluster_id=cluster_id,
                name=name,
                class_name=class_name,
                object_count=object_count,
            )

    def get_object(self, object_id: str) -> Optional[dict[str, Any]]:
        query = """
        MATCH (o:Object {object_id: $object_id})
        RETURN o {.*} AS obj
        """
        with self._driver.session() as session:
            rec = session.run(query, object_id=object_id).single()
            return dict(rec["obj"]) if rec else None

    def get_object_history(self, object_id: str) -> dict[str, Any]:
        query = """
        MATCH (o:Object {object_id: $object_id})
        OPTIONAL MATCH (o)-[:OBSERVED_AS]->(obs:Observation)
        OPTIONAL MATCH (obs)-[:IN_SCENE]->(s:Scene)
        OPTIONAL MATCH (o)-[:LOCATED_AT]->(l:Location)
        OPTIONAL MATCH (o)-[:MEMBER_OF]->(c:Cluster)
        WITH o,
             collect(DISTINCT obs {.*, scene_name: s.name}) AS observations,
             collect(DISTINCT l.name) AS locations,
             collect(DISTINCT c.cluster_id) AS clusters
        RETURN o {.*} AS object,
               observations,
               locations,
               clusters
        """
        with self._driver.session() as session:
            rec = session.run(query, object_id=object_id).single()
            if not rec:
                return {}
            observations = sorted(
                [dict(x) for x in rec["observations"] if x and x.get("observation_id")],
                key=lambda x: x.get("timestamp") or "",
                reverse=True,
            )
            return {
                "object": dict(rec["object"]),
                "observations": observations,
                "locations": [x for x in rec["locations"] if x],
                "clusters": [x for x in rec["clusters"] if x],
            }

    def latest_observation(self, object_id: str) -> Optional[dict[str, Any]]:
        query = """
        MATCH (o:Object {object_id: $object_id})-[:OBSERVED_AS]->(obs:Observation)
        OPTIONAL MATCH (obs)-[:IN_SCENE]->(s:Scene)
        RETURN obs {.*, scene_name: s.name, scene_id: s.scene_id} AS observation
        ORDER BY obs.timestamp DESC
        LIMIT 1
        """
        with self._driver.session() as session:
            rec = session.run(query, object_id=object_id).single()
            return dict(rec["observation"]) if rec and rec["observation"] else None

    def search_objects_by_class(self, class_name: str, limit: int = 20) -> list[dict[str, Any]]:
        query = """
        MATCH (o:Object)
        WHERE toLower(o.class_name) = toLower($class_name)
        OPTIONAL MATCH (o)-[:OBSERVED_AS]->(obs:Observation)-[:IN_SCENE]->(s:Scene)
        WITH o, obs, s
        ORDER BY obs.timestamp DESC
        WITH o, collect({obs: obs, scene: s})[0] AS latest
        RETURN o {.*} AS object,
               latest.obs.timestamp AS last_seen,
               latest.scene.name AS last_scene
        ORDER BY coalesce(o.last_seen, '') DESC
        LIMIT $limit
        """
        with self._driver.session() as session:
            rows = session.run(query, class_name=class_name, limit=limit)
            return [
                {
                    "object": dict(r["object"]),
                    "last_seen": r["last_seen"],
                    "last_scene": r["last_scene"],
                }
                for r in rows
            ]

    def list_objects(self, limit: int = 200) -> list[dict[str, Any]]:
        query = """
        MATCH (o:Object)
        OPTIONAL MATCH (o)-[:LOCATED_AT]->(l:Location)
        OPTIONAL MATCH (o)-[:MEMBER_OF]->(c:Cluster)
        RETURN o {.*} AS object,
               collect(DISTINCT l.name) AS locations,
               collect(DISTINCT c.cluster_id) AS clusters
        ORDER BY coalesce(o.last_seen, o.created_at, '') DESC
        LIMIT $limit
        """
        with self._driver.session() as session:
            rows = session.run(query, limit=limit)
            return [
                {
                    "object": dict(r["object"]),
                    "observation_count": int((r["object"] or {}).get("observation_count") or 0),
                    "locations": [x for x in r["locations"] if x],
                    "clusters": [x for x in r["clusters"] if x],
                }
                for r in rows
            ]

    def list_clusters(self) -> list[dict[str, Any]]:
        query = """
        MATCH (c:Cluster)
        OPTIONAL MATCH (o:Object)-[:MEMBER_OF]->(c)
        RETURN c {.*} AS cluster, collect(DISTINCT o.object_id) AS object_ids
        """
        with self._driver.session() as session:
            rows = session.run(query)
            out = []
            for r in rows:
                c = dict(r["cluster"] or {})
                oids = [x for x in r["object_ids"] if x]
                out.append(
                    {
                        "cluster_id": c.get("cluster_id"),
                        "name": c.get("name") or c.get("cluster_id"),
                        "class_name": c.get("class_name"),
                        "object_count": c.get("object_count") or len(oids),
                        "object_ids": oids,
                    }
                )
            return out

    def export_graph(self, object_id: Optional[str] = None, limit: int = 200) -> dict[str, Any]:
        # Minimal Neo4j export focused on one object or a sample
        if object_id:
            hist = self.get_object_history(object_id)
            if not hist:
                return {"nodes": [], "edges": []}
            nodes = []
            edges = []
            obj = hist["object"]
            oid = obj["object_id"]
            nodes.append({"id": f"obj:{oid}", "type": "Object", "label": oid, "data": obj})
            for obs in hist.get("observations") or []:
                oid_obs = obs.get("observation_id")
                if not oid_obs:
                    continue
                nodes.append(
                    {
                        "id": f"obs:{oid_obs}",
                        "type": "Observation",
                        "label": oid_obs,
                        "data": obs,
                    }
                )
                edges.append(
                    {
                        "id": f"obj:{oid}-OBSERVED_AS-obs:{oid_obs}",
                        "source": f"obj:{oid}",
                        "target": f"obs:{oid_obs}",
                        "type": "OBSERVED_AS",
                    }
                )
            for loc in hist.get("locations") or []:
                nodes.append(
                    {"id": f"loc:{loc}", "type": "Location", "label": loc, "data": {"name": loc}}
                )
                edges.append(
                    {
                        "id": f"obj:{oid}-AT_LOCATION-loc:{loc}",
                        "source": f"obj:{oid}",
                        "target": f"loc:{loc}",
                        "type": "AT_LOCATION",
                    }
                )
            for cl in hist.get("clusters") or []:
                nodes.append(
                    {
                        "id": f"cluster:{cl}",
                        "type": "Cluster",
                        "label": cl,
                        "data": {"cluster_id": cl},
                    }
                )
                edges.append(
                    {
                        "id": f"obj:{oid}-MEMBER_OF-cluster:{cl}",
                        "source": f"obj:{oid}",
                        "target": f"cluster:{cl}",
                        "type": "MEMBER_OF",
                    }
                )
            return {"nodes": nodes, "edges": edges}

        objects = self.list_objects(limit=min(limit, 50))
        nodes = []
        edges = []
        for row in objects:
            obj = row["object"]
            oid = obj.get("object_id")
            if not oid:
                continue
            nodes.append({"id": f"obj:{oid}", "type": "Object", "label": oid, "data": obj})
            for cl in row.get("clusters") or []:
                nodes.append(
                    {
                        "id": f"cluster:{cl}",
                        "type": "Cluster",
                        "label": cl,
                        "data": {"cluster_id": cl},
                    }
                )
                edges.append(
                    {
                        "id": f"obj:{oid}-MEMBER_OF-cluster:{cl}",
                        "source": f"obj:{oid}",
                        "target": f"cluster:{cl}",
                        "type": "MEMBER_OF",
                    }
                )
        # de-dupe nodes
        seen = set()
        uniq_nodes = []
        for n in nodes:
            if n["id"] in seen:
                continue
            seen.add(n["id"])
            uniq_nodes.append(n)
        return {"nodes": uniq_nodes, "edges": edges}

    def stats(self) -> dict[str, int]:
        with self._driver.session() as session:
            objects = session.run("MATCH (o:Object) RETURN count(o) AS n").single()["n"]
            observations = session.run(
                "MATCH (o:Observation) RETURN count(o) AS n"
            ).single()["n"]
            clusters = session.run("MATCH (c:Cluster) RETURN count(c) AS n").single()["n"]
            images = session.run("MATCH (i:Image) RETURN count(i) AS n").single()["n"]
            scenes = session.run("MATCH (s:Scene) RETURN count(s) AS n").single()["n"]
            return {
                "objects": int(objects),
                "observations": int(observations),
                "clusters": int(clusters),
                "images": int(images),
                "scenes": int(scenes),
            }

    def health(self) -> bool:
        try:
            with self._driver.session() as session:
                session.run("RETURN 1 AS ok").single()
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._driver.close()
