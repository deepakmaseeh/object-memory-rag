export type ComponentStatus = {
  name: string
  status: string
  detail?: string
  meta?: Record<string, unknown>
}

export type HealthStatus = {
  status: string
  qdrant: string
  neo4j: string
  ollama: string
  details: Record<string, unknown>
  components: ComponentStatus[]
}

export type ObservationMatch = {
  observation_id: string
  object_id: string
  class_name: string
  confidence: number
  matched_existing_object: boolean
  is_new: boolean
  decision?: string
  cluster_id?: string | null
  similarity: number
  attributes?: Record<string, unknown>
  location?: string | null
  scene_id?: string | null
  image_id?: string | null
  crop_path?: string | null
  mask_path?: string | null
  memory_saved?: boolean
  embedding_stored?: boolean
  graph_updated?: boolean
  cluster_assigned?: boolean
  candidate_scores?: Array<{ object_id: string; score: number; cluster_id?: string | null }>
}

export type PipelineLatencies = {
  yolo_ms: number
  sam_ms: number
  embedding_ms: number
  cluster_lookup_ms: number
  identity_resolution_ms: number
  neo4j_update_ms: number
  vlm_ms: number
  perception_ms: number
  total_ms: number
}

export type ProcessImageResult = {
  image_id: string
  original_path: string
  detection_count: number
  observations: Array<Record<string, unknown>>
  objects: Array<Record<string, unknown>>
  matches: ObservationMatch[]
  latencies: PipelineLatencies
  request_id?: string
  device?: string
  models?: Record<string, string>
}

export type MemoryResponse = {
  query: string
  answer: string
  context: Array<{
    object_id: string
    class_name: string
    similarity?: number | null
    last_scene?: string | null
    last_location?: string | null
    last_seen?: string | null
    observation_count: number
    summary?: string
    attributes?: Record<string, unknown>
    locations?: string[]
  }>
  raw_context?: string | null
}

export type Stats = {
  objects: number
  observations: number
  clusters: number
  images: number
  scenes: number
  clusters_ram: number
  qdrant_mode: string
  graph_backend: string
  device: string
}

export type GraphPayload = {
  nodes: Array<{ id: string; type: string; label: string; data?: Record<string, unknown> }>
  edges: Array<{ id: string; source: string; target: string; type: string }>
}
