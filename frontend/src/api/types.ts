export type IdentityScoreBreakdown = {
  overall_score?: number
  visual_score?: number
  text_score?: number
  brand_score?: number
  semantic_score?: number
  attribute_score?: number
  brand_conflict?: boolean
  decision?: string
  reason_codes?: string[]
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
  candidate_scores?: Array<{
    object_id: string
    score: number
    visual_score?: number
    text_score?: number
    brand_score?: number
    semantic_score?: number
    overall_score?: number
    brand_conflict?: boolean
    reason_codes?: string[]
    cluster_id?: string | null
  }>
  product_signature_id?: string | null
  product_label?: string | null
  identity_score?: IdentityScoreBreakdown | Record<string, unknown> | null
  reason_codes?: string[]
  object_signature?: Record<string, unknown> | null
  ocr_text?: string | null
  identity_path?: string | null
}

export type ProcessingStrength = 'auto' | 'light' | 'medium' | 'strong'

export type ProcessingOptions = {
  enhance_for_ai: boolean
  remove_background: boolean
  clean_for_auction: boolean
  remove_noise: boolean
  improve_resolution: boolean
}

export type RecognitionSource = 'original' | 'ai_enhanced' | 'auction'

export type ImageDerivatives = {
  original_path: string
  ai_enhanced_path?: string | null
  auction_path?: string | null
  transparent_preview_path?: string | null
  processing_meta?: Record<string, unknown>
}

export type PrepareImageResult = {
  image_id: string
  width: number
  height: number
  derivatives: ImageDerivatives
  preview_urls: Record<string, string>
  options: ProcessingOptions
  strength: ProcessingStrength
  preprocess_ms: number
  auction_ms: number
}

export type PipelineLatencies = {
  yolo_ms: number
  sam_ms: number
  embedding_ms: number
  ocr_ms?: number
  preprocess_ms?: number
  auction_ms?: number
  cluster_lookup_ms: number
  identity_resolution_ms: number
  identity_scoring_ms?: number
  neo4j_update_ms: number
  vlm_ms: number
  vlm_verify_ms?: number
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
  recognition_source?: RecognitionSource | string
  recognition_path?: string | null
  processing_options?: Record<string, unknown> | null
  derivatives?: ImageDerivatives | Record<string, unknown> | null
  preview_urls?: Record<string, string>
}

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
    product_signature_id?: string | null
    product_label?: string | null
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
