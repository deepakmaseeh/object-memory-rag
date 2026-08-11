import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { Card, EmptyState, ErrorBox, PageHeader, StatusChip } from '../components/ui'

export function SystemPage() {
  const system = useQuery({ queryKey: ['system'], queryFn: () => api.system() })
  const health = useQuery({
    queryKey: ['health-models'],
    queryFn: () => api.health(true),
    enabled: false,
  })

  return (
    <div>
      <PageHeader title="System / Debug" subtitle="Models, devices, paths, health" />
      {system.isError && <ErrorBox error={system.error} />}

      <div className="mb-4">
        <button
          onClick={() => health.refetch()}
          className="rounded border border-slate-600 bg-slate-800 px-3 py-2 text-sm hover:bg-slate-700"
          disabled={health.isFetching}
        >
          {health.isFetching ? 'Running health + model probe…' : 'Run Health Check (load models)'}
        </button>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card title="Runtime">
          {system.isLoading ? (
            <EmptyState message="Loading…" />
          ) : system.data ? (
            <pre className="text-xs mono text-slate-300 whitespace-pre-wrap overflow-auto max-h-[50vh]">
              {JSON.stringify(
                {
                  device: system.data.device,
                  qdrant_mode: system.data.qdrant_mode,
                  graph_backend: system.data.graph_backend,
                  models: system.data.models,
                  ollama: system.data.ollama,
                  paths: system.data.paths,
                },
                null,
                2,
              )}
            </pre>
          ) : null}
        </Card>

        <Card title="Health snapshot">
          {health.data ? (
            <div className="space-y-2">
              <div>
                Overall <StatusChip status={health.data.status} />
              </div>
              <ul className="space-y-2 text-sm">
                {health.data.components.map((c) => (
                  <li key={c.name} className="flex justify-between gap-2 border-b border-slate-800 pb-1">
                    <span>
                      {c.name}
                      <div className="text-xs text-slate-500 mono">{c.detail}</div>
                    </span>
                    <StatusChip status={c.status} />
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <EmptyState message="Run health check to probe YOLO/SAM/CLIP (may take a few seconds)." />
          )}
          {health.isError && <ErrorBox error={health.error} />}
        </Card>
      </div>
    </div>
  )
}
