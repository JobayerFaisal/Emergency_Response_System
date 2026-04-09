import React, { useEffect, useMemo, useState, useCallback } from 'react'
import KPIBar from './components/layout/KPIBar.jsx'
import Sidebar from './components/layout/Sidebar.jsx'
import MainMap from './components/map/MainMap.jsx'
import LayerControls from './components/map/LayerControls.jsx'
import useWebSocket from './hooks/useWebSocket.js'
import useDashboard from './hooks/useDashboard.js'
import useZones from './hooks/useZones.js'

function formatScenarioLabel(mode) {
  if (!mode || mode === 'LIVE') return 'LIVE'
  return mode.replaceAll('_', ' ')
}

function formatScenarioTimestamp(value) {
  if (!value) return ''
  const dt = new Date(value)
  if (Number.isNaN(dt.getTime())) return value
  return dt.toLocaleString('en-BD', {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function App() {
  const [activeLayers, setActiveLayers] = useState({
    flood: true,
    rain: true,
    teams: true,
    pins: true,
    river: true,
    routes: true,
  })
  const [activeAgent, setActiveAgent] = useState(null)
  const [sidebarTab, setSidebarTab] = useState('feed')
  const [replayState, setReplayState] = useState({
    mode: 'LIVE',
    enabled: false,
    running: false,
    paused: false,
    tick: 0,
    scenario_date: '',
  })
  const [replayBusy, setReplayBusy] = useState(false)
  const [eventResetToken, setEventResetToken] = useState(0)

  const isReplay = replayState.mode !== 'LIVE'

  const { events, connected } = useWebSocket('ws://localhost:8005/ws', {
    paused: isReplay,
    resetToken: eventResetToken,
  })
  const { kpi, agents, refreshDashboard } = useDashboard({ paused: isReplay, events })
  const { zones } = useZones()

  const scenarioLabel = useMemo(
    () => formatScenarioLabel(replayState.mode),
    [replayState.mode]
  )
  const scenarioTimeLabel = useMemo(
    () => formatScenarioTimestamp(replayState.scenario_date),
    [replayState.scenario_date]
  )

  const toggleLayer = (layer) => {
    setActiveLayers((prev) => ({ ...prev, [layer]: !prev[layer] }))
  }

  const handleAgentClick = (agentId) => {
    setActiveAgent(agentId)
    setSidebarTab('agent')
  }

  const loadReplayStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/replay/status')
      if (!res.ok) throw new Error(`Replay status failed: ${res.status}`)
      const json = await res.json()
      setReplayState(json)
    } catch (err) {
      console.error(err)
    }
  }, [])

  const sendReplayAction = useCallback(async (path) => {
    try {
      setReplayBusy(true)
      const res = await fetch(path, { method: 'POST' })
      if (!res.ok) throw new Error(`Replay action failed: ${res.status}`)
      const json = await res.json()
      setReplayState(json)

      if (path === '/api/replay/stop') {
        setEventResetToken((prev) => prev + 1)
        await refreshDashboard()
      }
    } catch (err) {
      console.error(err)
    } finally {
      setReplayBusy(false)
    }
  }, [refreshDashboard])

  useEffect(() => {
    loadReplayStatus()
    const timer = setInterval(loadReplayStatus, 30000)  // Adjusted to 30 seconds
    return () => clearInterval(timer)
  }, [loadReplayStatus])

  useEffect(() => {
    if (!isReplay) {
      refreshDashboard()
    }
  }, [isReplay, refreshDashboard])

  // Manual refresh function
  const handleManualRefresh = () => {
    refreshDashboard()
  }

  return (
    <div className="app-shell" style={{ position: 'relative' }}>
      <KPIBar kpi={kpi} connected={connected} />

      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 12,
          padding: '10px 18px',
          background: isReplay
            ? 'linear-gradient(90deg, rgba(124,58,237,0.20), rgba(59,130,246,0.12))'
            : 'linear-gradient(90deg, rgba(16,185,129,0.14), rgba(59,130,246,0.10))',
          borderBottom: isReplay
            ? '1px solid rgba(167,139,250,0.35)'
            : '1px solid rgba(52,211,153,0.28)',
          color: '#e5eefc',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              padding: '6px 12px',
              borderRadius: 999,
              fontSize: 12,
              fontWeight: 800,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: isReplay ? '#f5f3ff' : '#ecfdf5',
              background: isReplay ? 'rgba(109,40,217,0.85)' : 'rgba(5,150,105,0.85)',
            }}
          >
            <span>{isReplay ? '⏪' : '🟢'}</span>
            <span>{scenarioLabel}</span>
          </span>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: '#f8fbff' }}>
              {isReplay ? 'Historical replay scenario active' : 'Live operational data mode'}
            </span>
            <span style={{ fontSize: 12, color: 'rgba(226,232,240,0.82)' }}>
              {isReplay
                ? `Scenario source: historical replay${scenarioTimeLabel ? ` · ${scenarioTimeLabel}` : ''}`
                : 'Scenario source: real-time weather, river, and agent streams'}
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          {!isReplay && (
            <button
              onClick={() => sendReplayAction('/api/replay/start')}
              disabled={replayBusy}
              style={buttonStyle('#6d28d9')}
            >
              ⏪ Historical Replay
            </button>
          )}

          {isReplay && (
            <>
              <button
                onClick={() => sendReplayAction('/api/replay/start')}
                disabled={replayBusy}
                style={buttonStyle('#2563eb')}
              >
                ▶ Start
              </button>
              <button
                onClick={() => sendReplayAction('/api/replay/pause')}
                disabled={replayBusy}
                style={buttonStyle('#b45309')}
              >
                ⏸ Pause
              </button>
              <button
                onClick={() => sendReplayAction('/api/replay/reset')}
                disabled={replayBusy}
                style={buttonStyle('#6d28d9')}
              >
                ↺ Reset
              </button>
              <button
                onClick={() => sendReplayAction('/api/replay/stop')}
                disabled={replayBusy}
                style={buttonStyle('#059669')}
              >
                🟢 Return to Live
              </button>
              <span style={{ fontSize: 12, color: '#dbeafe' }}>
                {replayState.paused ? 'Paused' : replayState.running ? 'Running' : 'Stopped'} · Tick {replayState.tick}
              </span>
            </>
          )}
        </div>
      </div>

      <div className="main-layout">
        <div className="map-area" style={{ position: 'relative' }}>
          <MainMap
            zones={zones}
            activeLayers={activeLayers}
            agents={agents}
            events={events}
          />
          <LayerControls activeLayers={activeLayers} onToggle={toggleLayer} />

          {isReplay && (
            <div style={{ position: 'absolute', top: 14, left: 14, zIndex: 30, pointerEvents: 'none' }}>
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '8px 12px',
                  borderRadius: 12,
                  background: 'rgba(45, 27, 105, 0.78)',
                  border: '1px solid rgba(167,139,250,0.35)',
                  color: '#f5f3ff',
                  fontSize: 12,
                  fontWeight: 700,
                  boxShadow: '0 10px 28px rgba(0,0,0,0.22)',
                  backdropFilter: 'blur(8px)',
                }}
              >
                <span>⏪</span>
                <span>Replay Map View</span>
              </div>
            </div>
          )}
        </div>

        <Sidebar
          tab={sidebarTab}
          onTabChange={setSidebarTab}
          events={events}
          activeAgent={activeAgent}
          onAgentClick={handleAgentClick}
          agents={agents}
        />
      </div>

      {/* Manual Refresh Button */}
      <button onClick={handleManualRefresh} style={buttonStyle('#059669')}>
        Refresh Now
      </button>
    </div>
  )
}

function buttonStyle(bg) {
  return {
    border: 'none',
    borderRadius: 10,
    padding: '8px 12px',
    background: bg,
    color: '#fff',
    fontWeight: 700,
    cursor: 'pointer',
  }
}