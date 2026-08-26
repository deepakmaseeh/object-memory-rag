import { useMemo, useRef, useState } from 'react'

import { useMutation } from '@tanstack/react-query'

import { Link } from 'react-router-dom'

import { api } from '../api/client'

import type {

  ObservationMatch,

  PrepareImageResult,

  ProcessingOptions,

  ProcessingStrength,

  ProcessImageResult,

  RecognitionSource,

} from '../api/types'

import { Card, EmptyState, ErrorBox, PageHeader, StatusChip } from '../components/ui'

import clsx from 'clsx'



type RunSnapshot = {

  label: string

  result: ProcessImageResult

  previewUrl: string

}



const DEFAULT_OPTIONS: ProcessingOptions = {

  enhance_for_ai: false,

  remove_background: false,

  clean_for_auction: false,

  remove_noise: false,

  improve_resolution: false,

}



const STRENGTHS: ProcessingStrength[] = ['auto', 'light', 'medium', 'strong']



export function ProcessPage() {

  const inputRef = useRef<HTMLInputElement>(null)

  const [file, setFile] = useState<File | null>(null)

  const [preview, setPreview] = useState<string | null>(null)

  const [location, setLocation] = useState('Desk')

  const [forceVlm, setForceVlm] = useState(false)

  const [options, setOptions] = useState<ProcessingOptions>({ ...DEFAULT_OPTIONS })

  const [strength, setStrength] = useState<ProcessingStrength>('auto')

  const [prepared, setPrepared] = useState<PrepareImageResult | null>(null)

  const [recognitionSource, setRecognitionSource] = useState<RecognitionSource>('original')

  const [first, setFirst] = useState<RunSnapshot | null>(null)

  const [second, setSecond] = useState<RunSnapshot | null>(null)

  const [active, setActive] = useState<ProcessImageResult | null>(null)

  const [drag, setDrag] = useState(false)



  const prepare = useMutation({

    mutationFn: async (payload: { file: File }) => {

      return api.prepareImage(payload.file, options, strength)

    },

    onSuccess: (result) => {

      setPrepared(result)

      setRecognitionSource('original')

      setActive(null)

      setFirst(null)

      setSecond(null)

    },

  })



  const recognize = useMutation({

    mutationFn: async (payload: { asSecond: boolean }) => {

      if (!prepared) throw new Error('Prepare an image first')

      const result = await api.recognizeImage(

        prepared.image_id,

        recognitionSource,

        location,

        forceVlm,

        options.remove_background,

      )

      return { result, asSecond: payload.asSecond }

    },

    onSuccess: ({ result, asSecond }) => {

      setActive(result)

      const url = preview || api.media.raw(result.image_id)

      if (asSecond && first) {

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

    setPrepared(null)

    setActive(null)

    setSecond(null)

    setFirst(null)

    if (preview) URL.revokeObjectURL(preview)

    setPreview(f ? URL.createObjectURL(f) : null)

  }



  function toggleOption(key: keyof ProcessingOptions) {

    setOptions((prev) => ({ ...prev, [key]: !prev[key] }))

  }



  const previewTiles = useMemo(() => {
    if (!prepared) return []
    const id = prepared.image_id
    return [
      { key: 'original', label: 'Original', url: api.media.raw(id), available: true },
      {
        key: 'ai_enhanced',
        label: 'AI Enhanced',
        url: api.media.processed(id, 'ai'),
        available: Boolean(prepared.derivatives.ai_enhanced_path),
      },
      {
        key: 'background_removed',
        label: 'Background Removed',
        url: api.media.processed(id, 'transparent_preview'),
        available: Boolean(prepared.derivatives.transparent_preview_path),
      },
      {
        key: 'auction',
        label: 'Auction Ready',
        url: api.media.processed(id, 'auction'),
        available: Boolean(prepared.derivatives.auction_path),
      },
    ]
  }, [prepared])

  const busy = prepare.isPending || recognize.isPending



  return (

    <div>

      <PageHeader

        title="Process Image"

        subtitle="Prepare (optional enhancements) → pick recognition source → YOLO → SAM → identity → memory"

      />



      <div className="grid lg:grid-cols-[1.1fr_0.9fr] gap-4">

        <Card title="Input & processing options">

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

            <p className="text-xs text-slate-500 mt-1">JPEG / PNG / WebP · raw file stays immutable</p>

            <input

              ref={inputRef}

              type="file"

              accept="image/*"

              className="hidden"

              onChange={(e) => onFile(e.target.files?.[0] ?? null)}

            />

          </div>



          <div className="mt-4 grid sm:grid-cols-2 gap-2 text-xs text-slate-300">

            {(

              [

                ['enhance_for_ai', 'Enhance for AI'],

                ['remove_background', 'Remove background (SAM after detect)'],

                ['clean_for_auction', 'Clean for auction'],

                ['remove_noise', 'Remove noise'],

                ['improve_resolution', 'Improve resolution'],

              ] as const

            ).map(([key, label]) => (

              <label key={key} className="flex items-start gap-2 rounded border border-slate-800 p-2">

                <input

                  type="checkbox"

                  className="mt-0.5"

                  checked={options[key]}

                  onChange={() => toggleOption(key)}

                />

                <span>{label}</span>

              </label>

            ))}

          </div>



          <div className="mt-3 flex flex-wrap items-end gap-3">

            <label className="text-xs text-slate-400">

              Strength

              <select

                value={strength}

                onChange={(e) => setStrength(e.target.value as ProcessingStrength)}

                className="mt-1 block w-32 rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm capitalize"

              >

                {STRENGTHS.map((s) => (

                  <option key={s} value={s}>

                    {s}

                  </option>

                ))}

              </select>

            </label>

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

          </div>



          <div className="mt-4 flex flex-wrap gap-2">

            <button

              disabled={!file || busy}

              onClick={() => file && prepare.mutate({ file })}

              className="rounded bg-sky-600 hover:bg-sky-500 disabled:opacity-40 px-3 py-2 text-sm font-medium"

            >

              {prepare.isPending ? 'Preparing…' : 'Process (Prepare)'}

            </button>

            <button

              disabled={!prepared || busy}

              onClick={() => recognize.mutate({ asSecond: false })}

              className="rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 px-3 py-2 text-sm font-medium"

            >

              {recognize.isPending && !second ? 'Analyzing…' : 'Run AI Analysis'}

            </button>

            <button

              disabled={!prepared || !first || busy}

              onClick={() => recognize.mutate({ asSecond: true })}

              className="rounded border border-emerald-500/40 bg-emerald-500/10 hover:bg-emerald-500/20 disabled:opacity-40 px-3 py-2 text-sm text-emerald-200"

            >

              Process Same Image Again

            </button>

          </div>



          {(prepare.isError || recognize.isError) && (

            <div className="mt-3">

              <ErrorBox error={prepare.error || recognize.error} />

            </div>

          )}



          {(preview || active) && (

            <div className="mt-4 rounded border border-slate-800 overflow-hidden bg-black/30">

              {preview ? (

                <img src={preview} alt="upload" className="max-h-[320px] w-full object-contain" />

              ) : null}

              {active ? (

                <div className="p-2 text-xs text-slate-400 flex flex-wrap gap-3 mono">

                  <span>image_id={active.image_id}</span>

                  <span>source={active.recognition_source || 'original'}</span>

                  <span>device={active.device}</span>

                  <span>detections={active.detection_count}</span>

                  <span>total={active.latencies.total_ms.toFixed(0)} ms</span>

                </div>

              ) : null}

            </div>

          )}

        </Card>



        <div className="space-y-4">

          <Card title="Preview & recognition source">

            {!prepared ? (

              <EmptyState message="Run Prepare to generate optional derivatives. All options default off — recognition uses the original." />

            ) : (

              <div className="space-y-4">

                <div className="grid grid-cols-2 gap-2">

                  {previewTiles.map((tile) => (

                    <div

                      key={tile.key}

                      className="rounded border border-slate-800 bg-slate-950/40 overflow-hidden"

                    >

                      <div className="text-[11px] uppercase tracking-wide text-slate-400 px-2 py-1 border-b border-slate-800">

                        {tile.label}

                      </div>

                      {tile.available ? (

                        <img

                          src={tile.url}

                          alt={tile.label}

                          className="h-28 w-full object-contain bg-black/40"

                        />

                      ) : (

                        <div className="h-28 flex items-center justify-center text-xs text-slate-500">

                          not generated

                        </div>

                      )}

                    </div>

                  ))}

                </div>



                <div className="flex flex-wrap gap-2">

                  <SourceButton

                    label="Use Original"

                    active={recognitionSource === 'original'}

                    onClick={() => setRecognitionSource('original')}

                  />

                  <SourceButton

                    label="Use AI Enhanced"

                    active={recognitionSource === 'ai_enhanced'}

                    disabled={!prepared.derivatives.ai_enhanced_path}

                    onClick={() => setRecognitionSource('ai_enhanced')}

                  />

                  <SourceButton

                    label="Use Auction Version"

                    active={recognitionSource === 'auction'}

                    disabled={!prepared.derivatives.auction_path}

                    warn="Discouraged for identity — optimized for listing presentation"

                    onClick={() => setRecognitionSource('auction')}

                  />

                </div>



                {recognitionSource === 'auction' ? (

                  <div className="text-xs text-amber-300/90 rounded border border-amber-500/30 bg-amber-500/10 p-2">

                    Auction frames are for presentation. Identity matching works best on original or AI-enhanced sources.

                  </div>

                ) : null}



                <div className="text-xs mono text-slate-400">

                  prepare={prepared.preprocess_ms.toFixed(0)} ms · auction={prepared.auction_ms.toFixed(0)} ms

                </div>

              </div>

            )}

          </Card>



          <Card title="Second-loop identity">

            {!first ? (

              <EmptyState message="Run AI Analysis, then Process Same Image Again to verify identity reuse." />

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



function SourceButton({

  label,

  active,

  disabled,

  warn,

  onClick,

}: {

  label: string

  active: boolean

  disabled?: boolean

  warn?: string

  onClick: () => void

}) {

  return (

    <button

      type="button"

      disabled={disabled}

      title={warn}

      onClick={onClick}

      className={clsx(

        'rounded px-3 py-1.5 text-xs border',

        active

          ? 'border-sky-400 bg-sky-500/20 text-sky-100'

          : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500',

        disabled && 'opacity-40 cursor-not-allowed',

      )}

    >

      {label}

    </button>

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

  const attrEntries = Object.entries(attrs).filter(([k]) => k !== 'ocr')

  const sig = (m.object_signature || {}) as Record<string, unknown>

  const semantic = (sig.semantic || {}) as Record<string, unknown>

  const identity = (m.identity_score || {}) as Record<string, unknown>

  const topCandidate = m.candidate_scores?.[0]

  const brandConflict =

    Boolean(identity.brand_conflict) ||

    Boolean(topCandidate?.brand_conflict) ||

    (m.reason_codes || []).includes('BRAND_CONFLICT')



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

            {decision === 'NEW' ? 'NEW OBJECT DETECTED' : decision === 'UNCERTAIN' ? 'UNCERTAIN IDENTITY' : 'KNOWN OBJECT MATCH'}

          </div>

          <div className="font-medium capitalize text-lg">{m.class_name}</div>

          <div className="text-xs text-slate-400">{(m.confidence * 100).toFixed(1)}% conf</div>

        </div>

        <StatusChip status={decision} />

      </div>



      <div className="grid grid-cols-2 gap-2 mb-3 text-[11px]">

        <div className="rounded border border-slate-800 p-2 bg-slate-950/40">

          <div className="text-slate-500 uppercase">Object class</div>

          <div className="text-slate-200 capitalize">{m.class_name}</div>

        </div>

        <div className="rounded border border-slate-800 p-2 bg-slate-950/40">

          <div className="text-slate-500 uppercase">Product</div>

          <div className="text-slate-200">{m.product_label || String(semantic.product_name || semantic.object_type || '—')}</div>

        </div>

        <div className="rounded border border-slate-800 p-2 bg-slate-950/40">

          <div className="text-slate-500 uppercase">Physical object</div>

          <div className="text-sky-300 mono">{m.object_id}</div>

        </div>

        <div className="rounded border border-slate-800 p-2 bg-slate-950/40">

          <div className="text-slate-500 uppercase">Identity status</div>

          <div className="text-slate-200">{decision} · path={m.identity_path || '—'}</div>

        </div>

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

          <div className="mt-2 border-t border-slate-800 pt-2 text-[11px]">

            <div className="text-slate-400 mb-1">Candidate scoring</div>

            {m.candidate_scores.slice(0, 2).map((c) => (

              <div key={c.object_id} className="mb-1 rounded bg-slate-900/60 p-2">

                <div className="mono text-slate-300">Candidate: {c.object_id}</div>

                <div className="grid grid-cols-2 gap-1 text-slate-400 mt-1">

                  <span>Visual: {((c.visual_score ?? c.score) * 100).toFixed(0)}%</span>

                  <span>Text: {((c.text_score ?? 0) * 100).toFixed(0)}%</span>

                  <span>Brand: {c.brand_conflict ? 'CONFLICT' : `${((c.brand_score ?? 0) * 100).toFixed(0)}%`}</span>

                  <span>Overall: {((c.overall_score ?? c.score) * 100).toFixed(0)}%</span>

                </div>

                <div className="text-slate-500">

                  Decision: {brandConflict && c === topCandidate ? 'UNCERTAIN (brand conflict)' : decision}

                </div>

              </div>

            ))}

          </div>

        ) : null}

        {m.reason_codes?.length ? (

          <div className="text-slate-500">reasons: {m.reason_codes.join(', ')}</div>

        ) : null}

      </div>



      {m.ocr_text ? (

        <div className="mt-2 text-[11px] text-slate-400">OCR: {m.ocr_text}</div>

      ) : null}



      {(semantic.brand as string) ? (

        <div className="mt-1 text-[11px] text-slate-300">Brand: {String(semantic.brand)}</div>

      ) : null}



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

        <li>✓ Original</li>

        <li>✓ Crop</li>

        <li>✓ Mask</li>

        <li>{m.embedding_stored !== false ? '✓' : '○'} Embedding</li>

        <li>{m.ocr_text ? '✓' : '○'} OCR</li>

        <li>{attrEntries.length ? '✓' : '○'} Semantic attributes</li>

        <li>{m.product_signature_id ? '✓' : '○'} Product signature</li>

        <li>{m.cluster_assigned ? '✓' : '○'} Cluster</li>

        <li>{m.graph_updated !== false ? '✓' : '○'} Graph</li>

        <li>{m.memory_saved !== false ? '✓' : '○'} Observation</li>

        {decision === 'KNOWN' ? <li>✓ Previous physical object recognized</li> : null}

        {decision === 'UNCERTAIN' ? (

          <li>△ Uncertain · anti-merge bias {brandConflict ? '(brand conflict)' : ''}</li>

        ) : null}

      </ul>

    </div>

  )

}

