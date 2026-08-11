import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/ui'
import { DashboardPage } from './pages/DashboardPage'
import { ProcessPage } from './pages/ProcessPage'
import { ObjectsPage, ObjectDetailPage } from './pages/ObjectsPage'
import { ClustersPage } from './pages/ClustersPage'
import { GraphPage } from './pages/GraphPage'
import { ChatPage } from './pages/ChatPage'
import { SystemPage } from './pages/SystemPage'
import './index.css'

const qc = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 5000,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <AppShell>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/process" element={<ProcessPage />} />
            <Route path="/objects" element={<ObjectsPage />} />
            <Route path="/objects/:objectId" element={<ObjectDetailPage />} />
            <Route path="/clusters" element={<ClustersPage />} />
            <Route path="/graph" element={<GraphPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/system" element={<SystemPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AppShell>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
