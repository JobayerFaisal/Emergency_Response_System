import { useState, useEffect, useCallback } from 'react'

const KPI_URL    = '/api/kpi'
const AGENT_URLS = {
  agent1: '/api/agents/agent1',
  agent2: '/api/agents/agent2',
  agent3: '/api/agents/agent3',
  agent4: '/api/agents/agent4',
}
const POLL_INTERVAL_MS = 5_000

/**
 * useDashboard
 * Polls /api/kpi and each /api/agents/agentN endpoint on an interval.
 *
 * KPI shape expected from /api/kpi:
 * {
 *   incidentStatus, activeZones, criticalAlerts,
 *   deployedTeams, affectedPeople, severity, confidence
 * }
 *
 * Agent shapes expected from /api/agents/agentN:
 *   agent1: { detected, severity, confidence, trend, risk_factors, evidence_refs, weather, scores }
 *   agent2: { reports[], clusters{}, total_reports, critical_reports }
 *   agent3: { inventory[], volunteers[], total_volunteers, available_volunteers, deployed_volunteers }
 *   agent4: { missions[], total_missions, active_missions, completed_missions, failed_missions }
 */
export default function useDashboard() {
  const [kpi, setKpi] = useState({})
  const [agents, setAgents] = useState({
    agent1: null,
    agent2: null,
    agent3: null,
    agent4: null,
  })

  const fetchAll = useCallback(async () => {
    // Fetch KPI
    try {
      const res = await fetch(KPI_URL)
      if (res.ok) setKpi(await res.json())
    } catch { /* ignore */ }

    // Fetch each agent in parallel
    const results = await Promise.allSettled(
      Object.entries(AGENT_URLS).map(async ([key, url]) => {
        const res = await fetch(url)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return [key, await res.json()]
      })
    )

    const updates = {}
    results.forEach(result => {
      if (result.status === 'fulfilled') {
        const [key, data] = result.value
        updates[key] = data
      }
    })

    if (Object.keys(updates).length > 0) {
      setAgents(prev => ({ ...prev, ...updates }))
    }
  }, [])

  useEffect(() => {
    fetchAll()
    const interval = setInterval(fetchAll, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [fetchAll])

  return { kpi, agents }
}
