import { useMemo, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { ObservationMatch, ProcessImageResult } from '../api/types'
import { Card, EmptyState, ErrorBox, PageHeader, StatusChip } from '../components/ui'
import clsx from 'clsx'

type RunSnapshot = {
  label: string
  result: ProcessImageResult
  previewUrl: string
}

export function ProcessPage() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [location, setLocation] = useState('Desk')
  const [forceVlm, setForceVlm] = useState(false)
  const [first, setFirst] = useState<RunSnapshot | null>(null)
  const [second, setSecond] = useState<RunSnapshot | null>(null)
  const [active, setActive] = useState<ProcessImageResult | null>(null)
  const [drag, setDrag] = useState(false)

  const process = useMutation({
    mutationFn: async (payload: { file: File; asSecond: boolean }) => {
      const result = await api.processImage(payload.file, location, forceVlm)
      const url = URL.createObjectURL(payload.file)
      return { result, url, asSecond: payload.asSecond }
    },
    onSuccess: ({ result, url, asSecond }) => {
      setActive(result)
      if (asSecond) {
        setSecond({ label: 'SECOND RUN', result, previewUrl: url })
      } else {
        setFirst({ label: 'FIRST RUN', result, previewUrl: url })
        setSecond(null)
      }
    },
  })

  const matchSummary = useMemo(() => {
    if (!first || !second) return null
    const firstIds = new Set(first.result.matches.map((m) => m.object_id))
    const rows = second.result.matches.map((m) => ({
      ...m,
      reuse_ok: !m.is_new && firstIds.has(m.object_id),
    }))
    const duplicate_created = second.result.matches.some((m) => m.is_new)
    const ok = !duplicate_created && rows.every((r) => r.reuse_ok)
    return { rows, duplicate_created, ok }
  }, [first, second])

  function onFile(f: File | null) {
    setFile(f)
    setSecond(null)
    if (preview) URL.revokeObjectURL(preview)
    setPreview(f ? URL.createObjectURL(f) : null)
  }

  return (
    <div>
      <PageHeader
        title="Process Image"
        subtitle="Upload → YOLO → SAM → embedding → cluster identity → memory"
      />

      <div className="grid lg:grid-cols-[1.1fr_0.9fr] gap-4">
        <Card title="Input">
          <div
            className={clsx(
              'border border-dashed rounded-lg p-6 text-center transition cursor-pointer',
              drag ? 'border-sky-400 bg-sky-500/10' : 'border-slate-700 hover:border-slate-500',
            )}
            onDragOver={(e) => {
              e.preventDefault()
              setDrag(true)
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDrag(false)
              const f = e.dataTransfer.files?.[0]
              if (f) onFile(f)
            }}
            onClick={() => inputRef.current?.click()}
          >
            <p className="text-sm text-slate-300">
              {file ? file.name : 'Drag & drop an image, or click to browse'}
            </p>
            <p className="text-xs text-slate-500 mt-1">JPEG / PNG / WebP</p>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0] ?? null)}
            />
          </div>

          <div className="mt-4 flex flex-wrap items-end gap-3">
            <label className="text-xs text-slate-400">
              Location
              <input
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="mt-1 block w-40 rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              />
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-300 pb-2">
              <input type="checkbox" checked={forceVlm} onChange={(e) => setForceVlm(e.target.checked)} />
              Force VLM attributes
            </label>
            <button
              disabled={!file || process.isPending}
              onClick={() => file && process.mutate({ file, asSecond: false })}
              className="rounded bg-sky-600 hover:bg-sky-500 disabled:opacity-40 px-3 py-2 text-sm font-medium"
            >
              {process.isPending && !second ? 'Processing…' : 'Process Image'}
            </button>
            <button
              disabled={!file || process.isPending}
              onClick={() => file && process.mutate({ file, asSecond: true })}
              className="rounded border border-emerald-500/40 bg-emerald-500/10 hover:bg-emerald-500/20 disabled:opacity-40 px-3 py-2 text-sm text-emerald-200"
            >
              Process Same Image Again
            </button>
          </div>

          {process.isError && (
            <div className="mt-3">
              <ErrorBox error={process.error} />
            </div>
          )}

          {(preview || active) && (
            <div className="mt-4 rounded border border-slate-800 overflow-hidden bg-black/30">
              {preview ? (
                <img src={preview} alt="upload" className="max-h-[420px] w-full object-contain" />
              ) : null}
              {active ? (
                <div className="p-2 text-xs text-slate-400 flex flex-wrap gap-3 mono">
                  <span>image_id={active.image_id}</span>
                  <span>device={active.device}</span>
                  <span>detections={active.detection_count}</span>
                  <span>total={active.latencies.total_ms.toFixed(0)} ms</span>
                </div>
              ) : null}
            </div>
          )}
        </Card>

        <div className="space-y-4">
          <Card title="Second-loop identity">
            {!first ? (
              <EmptyState message="Process an image to create memory, then process the same image again." />
            ) : (
              <div className="space-y-3 text-sm">
                <RunBlock title="FIRST RUN" result={first.result} />
                {second ? <RunBlock title="SECOND RUN" result={second.result} /> : null}
                {matchSummary ? (
                  <div
                    className={clsx(
                      'rounded border p-3',
                      matchSummary.ok
                        ? 'border-emerald-500/40 bg-emerald-500/10'
                        : 'border-rose-500/40 bg-rose-500/10',
                    )}
                  >
                    <div className="font-medium mb-1">
                      {matchSummary.ok ? '✓ Previous objects recognized' : '✗ Identity mismatch'}
                    </div>
                    <div className="text-xs mono space-y-1">
                      <div>duplicate_created = {String(matchSummary.duplicate_created)}</div>
                      <div>matched_existing = {String(!matchSummary.duplicate_created)}</div>
                      {matchSummary.rows.map((r) => (
                        <div key={r.observation_id}>
                          {r.class_name} → {r.object_id} sim={r.similarity.toFixed(3)} new=
                          {String(r.is_new)} reuse={String(r.reuse_ok)}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            )}
          </Card>

          <Card title="Latencies (ms)">
            {active ? (
              <dl className="grid grid-cols-2 gap-2 text-xs mono">
                {Object.entries(active.latencies).map(([k, v]) => (
                  <div key={k} className="flex justify-between border-b border-slate-800/70 py-1">
                    <dt className="text-slate-400">{k}</dt>
                    <dd>{Number(v).toFixed(1)}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <EmptyState message="No timings yet" />
            )}
          </Card>
        </div>
      </div>

      {active && (
        <div className="mt-4">
          <Card title="Detected objects">
            {active.matches.length === 0 ? (
              <EmptyState message="No detections in this image." />
            ) : (
              <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">
                {active.matches.map((m) => (
                  <MatchCard key={m.observation_id} m={m} />
                ))}
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}

function RunBlock({ title, result }: { title: string; result: ProcessImageResult }) {
  return (
    <div className="rounded border border-slate-700 bg-slate-950/50 p-3">
      <div className="text-xs font-semibold text-sky-300 mb-2">{title}</div>
      <ul className="space-y-1 text-xs mono">
        {result.matches.map((m) => (
          <li key={m.observation_id}>
            {m.class_name} object_id={m.object_id} decision={m.decision || (m.is_new ? 'NEW' : 'KNOWN')}{' '}
            new_object={String(m.is_new)} sim={m.similarity.toFixed(3)}
          </li>
        ))}
      </ul>
    </div>
  )
}

function MatchCard({ m }: { m: ObservationMatch }) {
  const decision = (m.decision || (m.is_new ? 'NEW' : 'KNOWN')).toUpperCase()
  const isNew = decision === 'NEW' || decision === 'UNCERTAIN'
  const attrs = m.attributes || {}
  const attrEntries = Object.entries(attrs)

  return (
    <div
      className={clsx(
        'rounded border p-3',
        decision === 'KNOWN'
          ? 'border-emerald-500/40 bg-emerald-500/5'
          : decision === 'UNCERTAIN'
            ? 'border-amber-500/40 bg-amber-500/5'
            : 'border-sky-500/40 bg-sky-500/5',
      )}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-slate-400">
            {isNew ? 'New object detected' : 'Known object match'}
          </div>
          <div className="font-medium capitalize text-lg">{m.class_name}</div>
          <div className="text-xs text-slate-400">{(m.confidence * 100).toFixed(1)}% conf</div>
        </div>
        <StatusChip status={decision} />
      </div>

      <div className="grid grid-cols-2 gap-2 mb-3">
        <img
          src={api.media.crop(m.observation_id)}
          alt="crop"
          className="h-28 w-full object-contain rounded border border-slate-800 bg-black/40"
          onError={(e) => {
            ;(e.target as HTMLImageElement).style.opacity = '0.2'
          }}
        />
        <img
          src={api.media.mask(m.observation_id)}
          alt="mask"
          className="h-28 w-full object-contain rounded border border-slate-800 bg-black/40"
          onError={(e) => {
            ;(e.target as HTMLImageElement).style.opacity = '0.2'
          }}
        />
      </div>

      <div className="text-[11px] mono text-slate-300 space-y-0.5 break-all">
        <div>
          Object ID:{' '}
          <Link className="text-sky-300 underline" to={`/objects/${m.object_id}`}>
            {m.object_id}
          </Link>
        </div>
        <div>Observation: {m.observation_id}</div>
        <div>Cluster: {m.cluster_id || 'pending / assigned after rebuild'}</div>
        <div>Similarity: {m.similarity.toFixed(4)}</div>
        <div>Location: {m.location || '—'}</div>
        {m.candidate_scores?.length ? (
          <div className="text-slate-500">
            candidates:{' '}
            {m.candidate_scores
              .slice(0, 3)
              .map((c) => `${c.object_id}@${c.score.toFixed(2)}`)
              .join(', ')}
          </div>
        ) : null}
      </div>

      {attrEntries.length > 0 ? (
        <div className="mt-3 border-t border-slate-800 pt-2">
          <div className="text-xs font-medium text-slate-300 mb-1">Object details (VLM)</div>
          <dl className="grid grid-cols-2 gap-x-2 gap-y-1 text-[11px]">
            {attrEntries.map(([k, v]) => (
              <div key={k}>
                <dt className="text-slate-500 capitalize">{k}</dt>
                <dd className="text-slate-200">{String(v)}</dd>
              </div>
            ))}
          </dl>
        </div>
      ) : isNew ? (
        <div className="mt-3 text-[11px] text-slate-500">
          No VLM attributes yet (enable vision / check Ollama qwen2.5vl:3b).
        </div>
      ) : null}

      <ul className="mt-3 space-y-0.5 text-[11px] text-slate-300">
        <li>{m.memory_saved !== false ? '✓' : '○'} Object saved to memory</li>
        <li>{m.embedding_stored !== false ? '✓' : '○'} Embedding stored</li>
        <li>{m.graph_updated !== false ? '✓' : '○'} Graph updated</li>
        <li>{m.cluster_assigned ? '✓' : '○'} Cluster assigned</li>
        {decision === 'KNOWN' ? <li>✓ Previous object recognized · no duplicate created</li> : null}
        {decision === 'UNCERTAIN' ? (
          <li>△ Uncertain band · created new id (anti-merge bias)</li>
        ) : null}
      </ul>
    </div>
  )
}
