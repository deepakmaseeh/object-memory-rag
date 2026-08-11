import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { Card, EmptyState, ErrorBox, PageHeader } from '../components/ui'

export function ClustersPage() {
  const q = useQuery({ queryKey: ['clusters'], queryFn: () => api.listClusters() })

  const byClass = new Map<string, Array<Record<string, unknown>>>()
  for (const c of q.data?.clusters || []) {
    const cls = String(c.class_name || 'unknown')
    if (!byClass.has(cls)) byClass.set(cls, [])
    byClass.get(cls)!.push(c)
  }

  return (
    <div>
      <PageHeader
        title="Clusters"
        subtitle="Fast memory structure: query → nearest clusters → candidates → identity"
      />
      {q.isError && <ErrorBox error={q.error} />}

      <Card title="Fast memory lookup flow" className="mb-4">
        <pre className="text-xs mono text-slate-300 whitespace-pre-wrap leading-relaxed">
{`query embedding
  → RAM-resident cluster centroids
  → top-N relevant clusters
  → candidate objects from those clusters
  → exact vector similarity
  → MATCH existing object  OR  CREATE new object`}
        </pre>
      </Card>

      <Card title="Cluster index">
        {q.isLoading ? (
          <EmptyState message="Loading clusters…" />
        ) : !q.data?.clusters?.length ? (
          <EmptyState message="No clusters yet. Process images to build the index." />
        ) : (
          <div className="space-y-5">
            {[...byClass.entries()].map(([cls, items]) => (
              <div key={cls}>
                <h3 className="text-sm font-medium capitalize text-sky-200 mb-2">
                  {cls}{' '}
                  <span className="text-slate-400 font-normal">
                    ({items.reduce((n, c) => n + Number(c.object_count || (c.object_ids as string[])?.length || 0), 0)} objects)
                  </span>
                </h3>
                <div className="grid md:grid-cols-2 gap-3">
                  {items.map((c) => {
                    const id = String(c.cluster_id)
                    const oids = (c.object_ids as string[]) || []
                    return (
                      <div key={id} className="rounded border border-slate-800 p-3 bg-slate-950/40">
                        <div className="mono text-xs text-slate-200">{id}</div>
                        <div className="text-xs text-slate-400 mt-1">
                          objects: {Number(c.object_count || oids.length)} · source: {String(c.source || 'graph')}
                        </div>
                        <ul className="mt-2 space-y-0.5 text-[11px] mono">
                          {oids.slice(0, 8).map((oid) => (
                            <li key={oid}>
                              <Link className="text-sky-300 underline" to={`/objects/${oid}`}>
                                {oid}
                              </Link>
                            </li>
                          ))}
                          {oids.length > 8 ? <li className="text-slate-500">… +{oids.length - 8} more</li> : null}
                        </ul>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
