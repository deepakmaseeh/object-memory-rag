import type {
  GraphPayload,
  HealthStatus,
  MemoryResponse,
  PrepareImageResult,
  ProcessImageResult,
  ProcessingOptions,
  ProcessingStrength,
  RecognitionSource,
  Stats,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  // RAG + cold Ollama load can exceed 30s; chat must not look "stuck" without a clear timeout.
  const timeoutMs = path === '/query' ? 180_000 : 60_000
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller.signal,
    })
    if (!res.ok) {
      let detail = res.statusText
      try {
        const body = await res.json()
        detail = body.detail ?? JSON.stringify(body)
      } catch {
        /* ignore */
      }
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
    }
    return res.json() as Promise<T>
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(
        path === '/query'
          ? 'RAG query timed out (180s). Check Ollama is running and qwen3:8b is pulled.'
          : 'Request timed out.',
      )
    }
    if (err instanceof TypeError) {
      throw new Error(
        'Cannot reach API. Start backend on :8000 (`python main.py` with .venv-cuda).',
      )
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

export const api = {
  health: (loadModels = false) =>
    request<HealthStatus>(`/health?load_models=${loadModels}`),

  stats: () => request<Stats>('/stats'),

  system: () => request<Record<string, unknown>>('/system'),

  prepareImage: async (
    file: File,
    options: ProcessingOptions,
    strength: ProcessingStrength = 'auto',
    sceneId?: string,
  ) => {
    const form = new FormData()
    form.append('file', file)
    form.append('enhance_for_ai', String(options.enhance_for_ai))
    form.append('remove_background', String(options.remove_background))
    form.append('clean_for_auction', String(options.clean_for_auction))
    form.append('remove_noise', String(options.remove_noise))
    form.append('improve_resolution', String(options.improve_resolution))
    form.append('strength', strength)
    if (sceneId) form.append('scene_id', sceneId)
    return request<PrepareImageResult>('/process/prepare', {
      method: 'POST',
      body: form,
    })
  },

  recognizeImage: (
    imageId: string,
    recognitionSource: RecognitionSource = 'original',
    location = 'Desk',
    forceVlm = false,
    removeBackground = false,
  ) =>
    request<ProcessImageResult>('/process/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_id: imageId,
        recognition_source: recognitionSource,
        location_name: location,
        force_vlm: forceVlm,
        remove_background: removeBackground,
      }),
    }),

  processImage: async (file: File, location = 'Desk', forceVlm = false) => {
    const form = new FormData()
    form.append('file', file)
    form.append('location_name', location)
    form.append('force_vlm', String(forceVlm))
    return request<ProcessImageResult>('/process/image', {
      method: 'POST',
      body: form,
    })
  },

  query: (q: string) =>
    request<MemoryResponse>('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q }),
    }),

  listObjects: (limit = 200) =>
    request<{ count: number; objects: Array<Record<string, unknown>> }>(
      `/objects?limit=${limit}`,
    ),

  getObject: (id: string) => request<Record<string, unknown>>(`/objects/${encodeURIComponent(id)}`),

  listClusters: () =>
    request<{ count: number; clusters: Array<Record<string, unknown>> }>('/clusters'),

  getCluster: (id: string) =>
    request<Record<string, unknown>>(`/clusters/${encodeURIComponent(id)}`),

  graph: (objectId?: string) => {
    const q = objectId ? `?object_id=${encodeURIComponent(objectId)}` : ''
    return request<GraphPayload>(`/graph${q}`)
  },

  media: {
    crop: (observationId: string) => `${API_BASE}/media/crop/${encodeURIComponent(observationId)}`,
    mask: (observationId: string) => `${API_BASE}/media/mask/${encodeURIComponent(observationId)}`,
    raw: (imageId: string) => `${API_BASE}/media/raw/${encodeURIComponent(imageId)}`,
    processed: (imageId: string, kind: string) =>
      `${API_BASE}/media/processed/${encodeURIComponent(imageId)}/${encodeURIComponent(kind)}`,
    object: (objectId: string, kind: string) =>
      `${API_BASE}/media/object/${encodeURIComponent(objectId)}/${encodeURIComponent(kind)}`,
  },
}
