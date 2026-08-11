import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams, Link } from 'react-router-dom'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { api } from '../api/client'
import { Card, EmptyState, ErrorBox, PageHeader } from '../components/ui'

const typeColor: Record<string, string> = {
  Object: '#38bdf8',
  Observation: '#a3e635',
  Image: '#fbbf24',
  Scene: '#c084fc',
  Cluster: '#fb7185',
  Location: '#2dd4bf',
  Attribute: '#94a3b8',
}

export function GraphPage() {
  const [params] = useSearchParams()
  const objectId = params.get('object') || undefined
  const q = useQuery({
    queryKey: ['graph', objectId || 'all'],
    queryFn: () => api.graph(objectId),
  })

  const { nodes, edges } = useMemo(() => {
    const rawNodes = q.data?.nodes || []
    const rawEdges = q.data?.edges || []
    const byType = new Map<string, number>()
    const nodes: Node[] = rawNodes.map((n) => {
      const t = n.type || 'Object'
      const col = byType.get(t) || 0
      byType.set(t, col + 1)
      const row = Math.floor(col / 4)
      const x = (col % 4) * 220 + (Object.keys(typeColor).indexOf(t) % 3) * 40
      const y = Object.keys(typeColor).indexOf(t) * 120 + row * 70
      return {
        id: n.id,
        position: { x, y },
        data: {
          label: `${n.type}: ${n.label}`,
        },
        style: {
          border: `1px solid ${typeColor[t] || '#64748b'}`,
          background: '#0f172a',
          color: '#e2e8f0',
          fontSize: 11,
          borderRadius: 8,
          padding: 8,
          width: 180,
        },
      }
    })
    const edges: Edge[] = rawEdges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      label: e.type,
      style: { stroke: '#475569' },
      labelStyle: { fill: '#94a3b8', fontSize: 10 },
    }))
    return { nodes, edges }
  }, [q.data])

  return (
    <div>
      <PageHeader
        title="Memory Graph"
        subtitle={objectId ? `Subgraph for ${objectId}` : 'Objects, observations, scenes, clusters, locations'}
      />
      {objectId ? (
        <p className="text-xs mb-3">
          <Link className="text-sky-300 underline" to="/graph">Show full graph sample</Link>
        </p>
      ) : null}
      {q.isError && <ErrorBox error={q.error} />}
      <Card className="h-[70vh] p-0 overflow-hidden">
        {q.isLoading ? (
          <EmptyState message="Loading graph…" />
        ) : !nodes.length ? (
          <EmptyState message="No graph data yet. Process images to populate memory." />
        ) : (
          <ReactFlow nodes={nodes} edges={edges} fitView>
            <Background color="#1e293b" gap={18} />
            <MiniMap
              style={{ background: '#0f172a' }}
              nodeColor={(n) => {
                const t = String(n.data?.label || '').split(':')[0]
                return typeColor[t] || '#64748b'
              }}
            />
            <Controls />
          </ReactFlow>
        )}
      </Card>
      {q.data?.nodes?.length ? (
        <p className="text-xs text-slate-500 mt-2 mono">
          nodes={q.data.nodes.length} edges={q.data.edges.length}
        </p>
      ) : null}
    </div>
  )
}
