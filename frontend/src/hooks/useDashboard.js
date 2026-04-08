// frontend/src/hooks/useDashboard.js

import { useState, useEffect, useCallback } from 'react'

const KPI_URL = '/api/kpi'
const AGENT_URLS = {
  agent1: '/api/agents/agent1',
  agent2: '/api/agents/agent2',
  agent3: '/api/agents/agent3',
  agent4: '/api/agents/agent4',
}
const POLL_INTERVAL_MS = 15_000

export default function useDashboard({ paused = false } = {}) {
  const [kpi, setKpi] = useState({})
  const [agents, setAgents] = useState({
    agent1: null,
    agent2: null,
    agent3: null,
    agent4: null,
  })

  const fetchAll = useCallback(async () => {
    try {
      const res = await fetch(KPI_URL)
      if (res.ok) setKpi(await res.json())
    } catch {
      // ignore
    }

    const results = await Promise.allSettled(
      Object.entries(AGENT_URLS).map(async ([key, url]) => {
        const res = await fetch(url)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return [key, await res.json()]
      })
    )

    const updates = {}
    results.forEach((result) => {
      if (result.status === 'fulfilled') {
        const [key, data] = result.value
        updates[key] = data
      }
    })

    if (Object.keys(updates).length > 0) {
      setAgents((prev) => ({ ...prev, ...updates }))
    }
  }, [])

  useEffect(() => {
    if (paused) return undefined

    fetchAll()
    const interval = setInterval(fetchAll, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [fetchAll, paused])

  return { kpi, agents, refreshDashboard: fetchAll }
}