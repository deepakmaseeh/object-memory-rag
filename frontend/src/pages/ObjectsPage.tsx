import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api/client'
import { Card, EmptyState, ErrorBox, PageHeader } from '../components/ui'

export function ObjectsPage() {
  const q = useQuery({ queryKey: ['objects'], queryFn: () => api.listObjects() })

  return (
    <div>
      <PageHeader title="Object Memory" subtitle="Persistent identities across observations" />
      {q.isError && <ErrorBox error={q.error} />}
      <Card>
        {q.isLoading ? (
          <EmptyState message="Loading objects…" />
        ) : !q.data?.objects?.length ? (
          <EmptyState message="No objects stored yet. Process an image first." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2 pr-3">Object ID</th>
                  <th className="py-2 pr-3">Class</th>
                  <th className="py-2 pr-3">Obs</th>
                  <th className="py-2 pr-3">Last seen</th>
                  <th className="py-2 pr-3">Cluster</th>
                  <th className="py-2">Locations</th>
                </tr>
              </thead>
              <tbody>
                {q.data.objects.map((row) => {
                  const obj = (row.object || {}) as Record<string, unknown>
                  const id = String(obj.object_id ?? '')
                  const clusters = (row.clusters as string[]) || []
                  const locations = (row.locations as string[]) || []
                  return (
                    <tr key={id} className="border-b border-slate-800/70 hover:bg-slate-800/30">
                      <td className="py-2 pr-3 mono text-xs">
                        <Link className="text-sky-300 underline" to={`/objects/${id}`}>
                          {id}
                        </Link>
                      </td>
                      <td className="py-2 pr-3 capitalize">{String(obj.class_name ?? '—')}</td>
                      <td className="py-2 pr-3 mono">{String(obj.observation_count ?? row.observation_count ?? 0)}</td>
                      <td className="py-2 pr-3 mono text-xs">{String(obj.last_seen ?? '—')}</td>
                      <td className="py-2 pr-3 mono text-xs">{clusters[0] || String(obj.cluster_id ?? '—')}</td>
                      <td className="py-2 text-xs">{locations.join(', ') || '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

export function ObjectDetailPage() {
  const { objectId = '' } = useParams()
  const q = useQuery({
    queryKey: ['object', objectId],
    queryFn: () => api.getObject(objectId),
    enabled: Boolean(objectId),
  })

  const obj = (q.data?.object || {}) as Record<string, unknown>
  const observations = (q.data?.observations as Array<Record<string, unknown>>) || []
  const locations = (q.data?.locations as string[]) || []
  const clusters = (q.data?.clusters as string[]) || []

  return (
    <div>
      <PageHeader title="Object detail" subtitle={objectId} />
      {q.isError && <ErrorBox error={q.error} />}
      {q.isLoading ? (
        <EmptyState message="Loading object…" />
      ) : !q.data ? (
        <EmptyState message="Object not found." />
      ) : (
        <div className="grid lg:grid-cols-[1fr_1.2fr] gap-4">
          <Card title="Identity">
            <dl className="text-sm space-y-2 mono">
              <div><span className="text-slate-400">class:</span> {String(obj.class_name)}</div>
              <div><span className="text-slate-400">object_id:</span> {String(obj.object_id)}</div>
              <div><span className="text-slate-400">first/created:</span> {String(obj.created_at ?? '—')}</div>
              <div><span className="text-slate-400">last_seen:</span> {String(obj.last_seen ?? '—')}</div>
              <div><span className="text-slate-400">observations:</span> {String(obj.observation_count ?? observations.length)}</div>
              <div><span className="text-slate-400">cluster:</span> {clusters.join(', ') || String(obj.cluster_id ?? '—')}</div>
              <div><span className="text-slate-400">locations:</span> {locations.join(', ') || '—'}</div>
            </dl>
            {Object.keys((q.data?.attributes as Record<string, unknown>) || (obj.attributes as object) || {}).length ? (
              <div className="mt-4">
                <div className="text-xs text-slate-400 mb-1">Attributes</div>
                <dl className="grid grid-cols-2 gap-1 text-xs">
                  {Object.entries(
                    (q.data?.attributes as Record<string, unknown>) ||
                      (obj.attributes as Record<string, unknown>) ||
                      {},
                  ).map(([k, v]) => (
                    <div key={k}>
                      <dt className="text-slate-500 capitalize">{k}</dt>
                      <dd>{String(v)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            ) : null}
            {clusters[0] ? (
              <Link className="inline-block mt-3 text-sm text-sky-300 underline" to={`/clusters`}>
                View clusters
              </Link>
            ) : null}
            <Link className="inline-block mt-2 ml-3 text-sm text-sky-300 underline" to={`/graph?object=${objectId}`}>
              Open in graph
            </Link>
          </Card>

          <Card title="Observation timeline">
            {!observations.length ? (
              <EmptyState message="No observations." />
            ) : (
              <ol className="space-y-3">
                {observations.map((obs) => {
                  const oid = String(obs.observation_id ?? '')
                  return (
                    <li key={oid} className="rounded border border-slate-800 p-3">
                      <div className="flex gap-3">
                        <img
                          src={api.media.crop(oid)}
                          alt="crop"
                          className="h-16 w-16 object-contain rounded border border-slate-800 bg-black/30"
                          onError={(e) => {
                            ;(e.target as HTMLImageElement).style.opacity = '0.2'
                          }}
                        />
                        <div className="text-xs mono space-y-0.5 break-all">
                          <div className="text-sky-200">{oid}</div>
                          <div>ts: {String(obs.timestamp ?? '—')}</div>
                          <div>scene: {String(obs.scene_name ?? obs.scene_id ?? '—')}</div>
                          <div>image: {String(obs.image_id ?? '—')}</div>
                          <div>conf: {String(obs.confidence ?? '—')}</div>
                        </div>
                      </div>
                    </li>
                  )
                })}
              </ol>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
