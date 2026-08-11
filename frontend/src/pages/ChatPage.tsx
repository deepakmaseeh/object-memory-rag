import { useEffect, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { MemoryResponse } from '../api/types'
import { Card, EmptyState, ErrorBox, PageHeader } from '../components/ui'

type Message = {
  role: 'user' | 'assistant' | 'system'
  text: string
  sources?: MemoryResponse['context']
  raw?: string | null
}

export function ChatPage() {
  const [input, setInput] = useState('What is the cell phone?')
  const [messages, setMessages] = useState<Message[]>([])
  const bottomRef = useRef<HTMLDivElement | null>(null)

  const ask = useMutation({
    mutationFn: (q: string) => api.query(q),
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: data.answer || 'No answer returned from memory.',
          sources: data.context,
          raw: data.raw_context,
        },
      ])
    },
    onError: (err) => {
      const msg = err instanceof Error ? err.message : String(err)
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          text: `RAG query failed: ${msg}. Is the API running on :8000 and Ollama available?`,
        },
      ])
    },
  })

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, ask.isPending])

  function send() {
    const q = input.trim()
    if (!q || ask.isPending) return
    setMessages((prev) => [...prev, { role: 'user', text: q }])
    setInput('')
    ask.mutate(q)
  }

  return (
    <div>
      <PageHeader
        title="RAG Chat"
        subtitle="Answers only from stored object memory (qwen3:8b + graph/attributes)"
      />

      <Card className="min-h-[60vh] flex flex-col">
        <div className="flex-1 space-y-4 overflow-auto max-h-[55vh] pr-1">
          {!messages.length && !ask.isPending ? (
            <EmptyState message='Ask about objects in memory, e.g. “What is the cell phone?” or “Where did I last see the clock?”' />
          ) : (
            messages.map((m, i) => (
              <div
                key={`${m.role}-${i}`}
                className={
                  m.role === 'user'
                    ? 'ml-12 rounded-lg border border-sky-500/20 bg-sky-500/10 p-3'
                    : m.role === 'system'
                      ? 'rounded-lg border border-rose-500/30 bg-rose-500/10 p-3'
                      : 'mr-8 rounded-lg border border-slate-700 bg-slate-950/50 p-3'
                }
              >
                <div className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">
                  {m.role === 'user' ? 'You' : m.role === 'system' ? 'Error' : 'Memory'}
                </div>
                <div className="text-sm whitespace-pre-wrap">{m.text}</div>
                {m.sources?.length ? (
                  <div className="mt-3 border-t border-slate-800 pt-2">
                    <div className="text-xs text-slate-400 mb-1">Sources</div>
                    <ul className="space-y-1 text-xs mono">
                      {m.sources.map((s) => (
                        <li key={s.object_id}>
                          Object:{' '}
                          <Link className="text-sky-300 underline" to={`/objects/${s.object_id}`}>
                            {s.object_id}
                          </Link>{' '}
                          ({s.class_name}) · loc={s.last_location || '—'} · scene=
                          {s.last_scene || '—'} · last_seen={String(s.last_seen || '—')} · obs=
                          {s.observation_count}
                        </li>
                      ))}
                    </ul>
                    {m.raw ? (
                      <details className="mt-2">
                        <summary className="text-xs text-slate-500 cursor-pointer">Raw context</summary>
                        <pre className="mt-1 text-[11px] text-slate-400 whitespace-pre-wrap">{m.raw}</pre>
                      </details>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ))
          )}
          {ask.isPending ? (
            <div className="mr-8 rounded-lg border border-slate-700 bg-slate-950/50 p-3 text-sm text-slate-400">
              Retrieving memory and generating answer… (first Ollama call can take 15–60s)
            </div>
          ) : null}
          {ask.isError && !messages.some((m) => m.role === 'system') ? (
            <ErrorBox error={ask.error} />
          ) : null}
          <div ref={bottomRef} />
        </div>

        <div className="mt-4 flex gap-2 border-t border-slate-800 pt-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && send()}
            className="flex-1 rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
            placeholder="Ask memory…"
            disabled={ask.isPending}
          />
          <button
            onClick={send}
            disabled={ask.isPending || !input.trim()}
            className="rounded bg-sky-600 hover:bg-sky-500 disabled:opacity-40 px-4 py-2 text-sm font-medium"
          >
            {ask.isPending ? 'Thinking…' : 'Send'}
          </button>
        </div>
      </Card>
    </div>
  )
}
