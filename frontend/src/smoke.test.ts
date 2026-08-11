import { describe, expect, it } from 'vitest'
import type { HealthStatus } from './api/types'

describe('health status mapping', () => {
  it('accepts READY/DEGRADED/FAILED', () => {
    const allowed = new Set(['READY', 'DEGRADED', 'FAILED', 'NOT_CONFIGURED'])
    const sample: HealthStatus = {
      status: 'READY',
      qdrant: 'READY',
      neo4j: 'DEGRADED',
      ollama: 'READY',
      details: {},
      components: [],
    }
    expect(allowed.has(sample.status)).toBe(true)
  })
})
