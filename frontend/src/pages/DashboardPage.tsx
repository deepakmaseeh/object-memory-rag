import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { Card, EmptyState, ErrorBox, PageHeader, StatusChip } from '../components/ui'

export function DashboardPage() {
  const health = useQuery({ queryKey: ['health'], queryFn: () => api.health(false), refetchInterval: 15000 })
  const stats = useQuery({ queryKey: ['stats'], queryFn: () => api.stats(), refetchInterval: 15000 })

  return (
    <div>
      <PageHeader
        title="Dashboard"
        subtitle="System health, memory counts, and GPU/runtime status"
      />

      {health.isError && <ErrorBox error={health.error} />}
      {stats.isError && <ErrorBox error={stats.error} />}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {[
          ['Objects', stats.data?.objects],
          ['Observations', stats.data?.observations],
          ['Clusters', stats.data?.clusters],
          ['Images', stats.data?.images],
        ].map(([label, value]) => (
          <Card key={String(label)}>
            <div className="text-xs text-slate-400">{label}</div>
            <div className="text-2xl font-semibold mono mt-1">
              {value ?? (stats.isLoading ? '…' : 0)}
            </div>
          </Card>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card title="Overall status">
          {health.isLoading ? (
            <EmptyState message="Loading health…" />
          ) : health.data ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className="text-sm text-slate-400">Application</span>
                <StatusChip status={health.data.status} />
              </div>
              <div className="grid grid-cols-3 gap-2 text-sm">
                <div>Qdrant <StatusChip status={health.data.qdrant} /></div>
                <div>Graph <StatusChip status={health.data.neo4j} /></div>
                <div>Ollama <StatusChip status={health.data.ollama} /></div>
              </div>
              <dl className="text-xs space-y-1 text-slate-300 mono">
                <div>GPU: {String(health.data.details.gpu_name ?? 'n/a')}</div>
                <div>CUDA: {String(health.data.details.cuda_available ?? false)}</div>
                <div>Torch: {String(health.data.details.torch_version ?? 'n/a')}</div>
                <div>Device: {String(health.data.details.resolved_device ?? stats.data?.device ?? 'n/a')}</div>
                <div>VRAM: {String(health.data.details.vram_gb ?? 'n/a')} GB</div>
                <div>Qdrant mode: {stats.data?.qdrant_mode ?? 'n/a'}</div>
                <div>Graph backend: {stats.data?.graph_backend ?? 'n/a'}</div>
              </dl>
            </div>
          ) : null}
        </Card>

        <Card title="Components">
          {health.data?.components?.length ? (
            <ul className="space-y-2">
              {health.data.components.map((c) => (
                <li
                  key={c.name}
                  className="flex items-start justify-between gap-3 border-b border-slate-800/80 pb-2 last:border-0"
                >
                  <div>
                    <div className="text-sm font-medium capitalize">{c.name}</div>
                    <div className="text-xs text-slate-400 mono break-all">{c.detail}</div>
                  </div>
                  <StatusChip status={c.status} />
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState message="No component data yet. Is the backend running on :8000?" />
          )}
        </Card>
      </div>
    </div>
  )
}
