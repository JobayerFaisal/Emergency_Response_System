import { useState, useEffect, useCallback, useRef } from 'react'

const KPI_URL = '/api/kpi'
const AGENT_URLS = {
  agent1: '/api/agents/agent1',
  agent2: '/api/agents/agent2',
  agent3: '/api/agents/agent3',
  agent4: '/api/agents/agent4',
}

const POLL_INTERVAL_MS = 30000

export default function useDashboard({ paused = false, events = [] } = {}) {
  const [kpi, setKpi] = useState({})
  const [agents, setAgents] = useState({
    agent1: null,
    agent2: null,
    agent3: null,
    agent4: null,
  })

  const pausedRef = useRef(paused)
  useEffect(() => {
    pausedRef.current = paused
  }, [paused])

  const fetchAll = useCallback(async () => {
    if (pausedRef.current) return

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
  }, [paused, fetchAll])

  // ── Merge WS inventory_update events into agent3 state ─────────────────────
  // FIX: In replay mode, /api/agents/agent3 returns empty (by design).
  // Inventory is driven by WS `inventory_update` events from the replay
  // simulator. We watch the events array and rebuild agent3.inventory from
  // the latest inventory_update event whenever one arrives.
  const prevInventoryEventCountRef = useRef(0)
  useEffect(() => {
    const inventoryEvents = (events || []).filter(
      (ev) => ev.channel === 'inventory_update' || ev.type === 'inventory_update'
    )

    if (inventoryEvents.length === 0) return
    if (inventoryEvents.length === prevInventoryEventCountRef.current) return
    prevInventoryEventCountRef.current = inventoryEvents.length

    // Use the most recent inventory_update event
    const latest = inventoryEvents[inventoryEvents.length - 1]
    const p = latest.payload || {}

    // Map payload fields to the inventory[] shape Agent3Panel expects
    const RESOURCE_MAP = [
      { key: 'boats_available',  type: 'boat',      total: 6 },
      { key: 'medical_kits',     type: 'medicine',  total: 42 },
      { key: 'food_packs',       type: 'food',      total: 180 },
      { key: 'water_units',      type: 'water',     total: 260 },
    ]

    const inventory = RESOURCE_MAP
      .filter((m) => p[m.key] !== undefined)
      .map((m) => ({
        type:      m.type,
        available: Math.max(0, Number(p[m.key]) || 0),
        total:     m.total,
        unit:      '',
      }))

    if (inventory.length > 0) {
      setAgents((prev) => ({
        ...prev,
        agent3: {
          ...(prev.agent3 || {}),
          inventory,
          // Keep volunteers from the last REST poll (or empty during replay)
          volunteers: prev.agent3?.volunteers || [],
          total_volunteers: prev.agent3?.total_volunteers || 0,
          available_volunteers: prev.agent3?.available_volunteers || 0,
          deployed_volunteers: prev.agent3?.deployed_volunteers || 0,
        },
      }))
    }
  }, [events])

  return { kpi, agents, refreshDashboard: fetchAll }
}