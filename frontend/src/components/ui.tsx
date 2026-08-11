import { NavLink } from 'react-router-dom'
import clsx from 'clsx'
import {
  Activity,
  Box,
  Brain,
  GitBranch,
  LayoutDashboard,
  MessageSquare,
  Network,
  ScanSearch,
  Settings2,
} from 'lucide-react'

const links = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/process', label: 'Process Image', icon: ScanSearch },
  { to: '/objects', label: 'Objects', icon: Box },
  { to: '/clusters', label: 'Clusters', icon: Network },
  { to: '/graph', label: 'Memory Graph', icon: GitBranch },
  { to: '/chat', label: 'RAG Chat', icon: MessageSquare },
  { to: '/system', label: 'System', icon: Settings2 },
]

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen grid grid-cols-[240px_1fr]">
      <aside className="border-r border-slate-800/80 bg-slate-950/70 backdrop-blur px-4 py-5 flex flex-col gap-6">
        <div>
          <div className="flex items-center gap-2 text-sky-300">
            <Brain size={20} />
            <span className="font-semibold tracking-wide text-sm">OBJECT MEMORY</span>
          </div>
          <p className="text-xs text-slate-400 mt-1">Local RAG · YOLO · SAM · Ollama</p>
        </div>
        <nav className="flex flex-col gap-1">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-2 rounded-md px-3 py-2 text-sm transition',
                  isActive
                    ? 'bg-sky-500/15 text-sky-200 border border-sky-500/30'
                    : 'text-slate-300 hover:bg-slate-800/60 border border-transparent',
                )
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto text-[11px] text-slate-500 space-y-1">
          <div className="flex items-center gap-1"><Activity size={12} /> Backend :8000</div>
          <div>UI :5173</div>
          <div>Training frozen</div>
        </div>
      </aside>
      <main className="p-6 overflow-auto">{children}</main>
    </div>
  )
}

export function StatusChip({ status }: { status: string }) {
  const s = (status || '').toUpperCase()
  const color =
    s === 'READY' || s === 'OK' || s === 'UP'
      ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
      : s === 'DEGRADED'
        ? 'bg-amber-500/15 text-amber-200 border-amber-500/30'
        : s === 'FAILED' || s === 'DOWN' || s === 'UNAVAILABLE'
          ? 'bg-rose-500/15 text-rose-300 border-rose-500/30'
          : 'bg-slate-500/15 text-slate-300 border-slate-500/30'
  return (
    <span className={clsx('inline-flex items-center rounded border px-2 py-0.5 text-xs font-medium', color)}>
      {status}
    </span>
  )
}

export function PageHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <header className="mb-6">
      <h1 className="text-2xl font-semibold tracking-tight text-slate-100">{title}</h1>
      {subtitle ? <p className="text-sm text-slate-400 mt-1">{subtitle}</p> : null}
    </header>
  )
}

export function Card({
  title,
  children,
  className,
}: {
  title?: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <section
      className={clsx(
        'rounded-lg border border-slate-800 bg-slate-900/50 p-4 shadow-sm shadow-black/20',
        className,
      )}
    >
      {title ? <h2 className="text-sm font-medium text-slate-300 mb-3">{title}</h2> : null}
      {children}
    </section>
  )
}

export function EmptyState({ message }: { message: string }) {
  return <p className="text-sm text-slate-400 py-8 text-center">{message}</p>
}

export function ErrorBox({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error)
  return (
    <div className="rounded border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
      {msg}
    </div>
  )
}
