// Run via scripts/init_neo4j.py (cypher executed against live bolt connection)
// Constraints and indexes for the object-memory graph

CREATE CONSTRAINT object_id IF NOT EXISTS
FOR (o:Object) REQUIRE o.object_id IS UNIQUE;

CREATE CONSTRAINT observation_id IF NOT EXISTS
FOR (o:Observation) REQUIRE o.observation_id IS UNIQUE;

CREATE CONSTRAINT scene_id IF NOT EXISTS
FOR (s:Scene) REQUIRE s.scene_id IS UNIQUE;

CREATE CONSTRAINT cluster_id IF NOT EXISTS
FOR (c:Cluster) REQUIRE c.cluster_id IS UNIQUE;

CREATE CONSTRAINT attribute_name IF NOT EXISTS
FOR (a:Attribute) REQUIRE a.name IS UNIQUE;

CREATE CONSTRAINT location_name IF NOT EXISTS
FOR (l:Location) REQUIRE l.name IS UNIQUE;

CREATE CONSTRAINT image_id IF NOT EXISTS
FOR (i:Image) REQUIRE i.image_id IS UNIQUE;

CREATE INDEX object_class_name IF NOT EXISTS
FOR (o:Object) ON (o.class_name);

CREATE INDEX observation_timestamp IF NOT EXISTS
FOR (o:Observation) ON (o.timestamp);
