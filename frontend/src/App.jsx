import React, { useEffect, useMemo, useState, useCallback } from 'react'
import KPIBar from './components/layout/KPIBar.jsx'
import Sidebar from './components/layout/Sidebar.jsx'
import MainMap from './components/map/MainMap.jsx'
import LayerControls from './components/map/LayerControls.jsx'
import AgentFeed from './components/layout/AgentFeed.jsx'
import Agent1Panel from './components/panels/Agent1Panel.jsx'
import Agent2Panel from './components/panels/Agent2Panel.jsx'
import Agent3Panel from './components/panels/Agent3Panel.jsx'
import Agent4Panel from './components/panels/Agent4Panel.jsx'
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
  return dt.toLocaleString('en-BD', { year:'numeric', month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit' })
}

// ── Page 1: Map Dashboard ─────────────────────────────────────────────────────
function PageMap({ zones, activeLayers, agents, events, connected, toggleLayer,
                   replayState, replayBusy, isReplay, scenarioLabel, scenarioTimeLabel,
                   sendReplayAction }) {
  return (
    <div className="page-map">
      {/* Scenario Banner */}
      <div className={`scenario-banner ${isReplay ? 'scenario-banner--replay' : 'scenario-banner--live'}`}>
        <div className="scenario-banner-left">
          <span className={`scenario-pill ${isReplay ? 'scenario-pill--replay' : 'scenario-pill--live'}`}>
            <span>{isReplay ? '⏪' : '🟢'}</span>
            <span>{scenarioLabel}</span>
          </span>
          <div className="scenario-desc">
            <span className="scenario-title">{isReplay ? 'Historical replay scenario active' : 'Live operational data mode'}</span>
            <span className="scenario-sub">
              {isReplay
                ? `Historical replay${scenarioTimeLabel ? ` · ${scenarioTimeLabel}` : ''}`
                : 'Real-time weather, river & agent streams'}
            </span>
          </div>
        </div>
        <div className="scenario-banner-actions">
          {!isReplay && (
            <button className="btn btn--purple" onClick={() => sendReplayAction('/api/replay/start')} disabled={replayBusy}>
              ⏪ Historical Replay
            </button>
          )}
          {isReplay && (<>
            <button className="btn btn--blue" onClick={() => sendReplayAction('/api/replay/start')} disabled={replayBusy}>▶ Start</button>
            <button className="btn btn--amber" onClick={() => sendReplayAction('/api/replay/pause')} disabled={replayBusy}>⏸ Pause</button>
            <button className="btn btn--purple" onClick={() => sendReplayAction('/api/replay/reset')} disabled={replayBusy}>↺ Reset</button>
            <button className="btn btn--green" onClick={() => sendReplayAction('/api/replay/stop')} disabled={replayBusy}>🟢 Live</button>
            <span className="scenario-tick">
              {replayState.paused ? 'Paused' : replayState.running ? 'Running' : 'Stopped'} · Tick {replayState.tick}
            </span>
          </>)}
        </div>
      </div>

      {/* Map + Sidebar */}
      <div className="map-page-body">
        <div className="map-area" style={{ position: 'relative' }}>
          {isReplay && (
            <div className="map-replay-badge">⏪ Replay Map View</div>
          )}
          <MainMap zones={zones} activeLayers={activeLayers} agents={agents} events={events} />
          <LayerControls activeLayers={activeLayers} onToggle={toggleLayer} />
        </div>
        <Sidebar events={events} agents={agents} />
      </div>
    </div>
  )
}

// ── Page 2: Agent Panels ──────────────────────────────────────────────────────
function PageAgents({ agents, events }) {
  const [activeTab, setActiveTab] = useState('agent1')

  const TABS = [
    { id: 'agent1', label: 'Agent 1', sub: 'Environmental', icon: '🛰', color: '#ef4444' },
    { id: 'agent2', label: 'Agent 2', sub: 'Distress',     icon: '📱', color: '#f59e0b' },
    { id: 'agent3', label: 'Agent 3', sub: 'Resource',     icon: '📦', color: '#3b9eff' },
    { id: 'agent4', label: 'Agent 4', sub: 'Dispatch',     icon: '🚤', color: '#22c55e' },
  ]

  return (
    <div className="page-agents">
      <div className="page-agents-header">
        <div>
          <h2 className="page-title">Agent Detail Panels</h2>
          <p className="page-subtitle">Tabbed deep-dive into each AI agent's reasoning and outputs</p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="agent-tab-bar">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`agent-tab-btn ${activeTab === t.id ? 'agent-tab-btn--active' : ''}`}
            style={{ '--tab-color': t.color }}
            onClick={() => setActiveTab(t.id)}
          >
            <span className="atb-icon">{t.icon}</span>
            <div className="atb-text">
              <span className="atb-label">{t.label}</span>
              <span className="atb-sub">{t.sub}</span>
            </div>
          </button>
        ))}
      </div>

      {/* Panel content — two-column layout */}
      <div className="agents-content">
        {activeTab === 'agent1' && (
          <div className="agents-two-col">
            <Agent1Panel data={agents?.agent1} />
            <div className="agent-info-card">
              <h3 className="aic-title">🛰 Environmental Intelligence</h3>
              <p className="aic-body">Satellite SAR + depth estimation + weather data + 8-factor predictor. Agent 1 ingests Sentinel-1 SAR imagery, compares reference (dry) vs current (flooded) scenes, applies a CNN flood mask, estimates water depth, and cross-references open weather and river APIs.</p>
              <div className="aic-pipeline">
                <div className="aic-step">SAR Imagery</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step">CNN Flood Mask</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step">8-Factor Score</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step active-step">Redis Pub/Sub</div>
              </div>
              <div className="aic-feed">
                <p className="aic-feed-title">Live Agent Events</p>
                {(events||[]).filter(e=>e.channel==='flood_alert').slice(-5).reverse().map((e,i)=>(
                  <div key={i} className="aic-event">
                    <span className="aic-event-dot" style={{background:'#ef4444'}}/>
                    <span>{e.payload?.zone_name || 'Zone'} — risk {((e.payload?.risk_score||0)*100).toFixed(0)}%</span>
                    <span className="aic-event-time">{new Date(e.timestamp).toLocaleTimeString('en-BD',{hour:'2-digit',minute:'2-digit'})}</span>
                  </div>
                ))}
                {!(events||[]).filter(e=>e.channel==='flood_alert').length && (
                  <p className="aic-empty">Waiting for flood alerts…</p>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'agent2' && (
          <div className="agents-two-col">
            <Agent2Panel data={agents?.agent2} />
            <div className="agent-info-card">
              <h3 className="aic-title">📱 Distress Intelligence</h3>
              <p className="aic-body">Trilingual NLP (Bengali/English/Banglish) + cross-reference with Agent 1 flood data. Agent 2 processes incoming distress reports, classifies urgency, and asks Agent 1 via Redis: "Is this zone actually flooded?" — CONFIRMED = satellite agrees.</p>
              <div className="aic-pipeline">
                <div className="aic-step">Social Reports</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step">NLP Classify</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step">Agent 1 Cross-ref</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step active-step">Urgency Queue</div>
              </div>
              <div className="aic-feed">
                <p className="aic-feed-title">Live Distress Events</p>
                {(events||[]).filter(e=>e.channel==='distress_queue').slice(-5).reverse().map((e,i)=>(
                  <div key={i} className="aic-event">
                    <span className="aic-event-dot" style={{background:'#f59e0b'}}/>
                    <span>{e.payload?.district||e.payload?.zone_name||'Unknown'} — sev {e.payload?.severity||e.payload?.urgency||'?'}/5</span>
                    <span className="aic-event-time">{new Date(e.timestamp).toLocaleTimeString('en-BD',{hour:'2-digit',minute:'2-digit'})}</span>
                  </div>
                ))}
                {!(events||[]).filter(e=>e.channel==='distress_queue').length && (
                  <p className="aic-empty">Waiting for distress reports…</p>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'agent3' && (
          <div className="agents-two-col">
            <Agent3Panel data={agents?.agent3} />
            <div className="agent-info-card">
              <h3 className="aic-title">📦 Resource Management</h3>
              <p className="aic-body">Inventory tracking + auto-deduct on dispatch + manual restock. Agent 3 receives allocation requests from Agent 2, checks availability, deducts from inventory, and publishes dispatch orders to Agent 4 over Redis.</p>
              <div className="aic-pipeline">
                <div className="aic-step">Allocation Request</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step">Inventory Check</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step">Deduct Units</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step active-step">Dispatch Order</div>
              </div>
              <div className="aic-feed">
                <p className="aic-feed-title">Live Inventory Events</p>
                {(events||[]).filter(e=>e.channel==='inventory_update').slice(-5).reverse().map((e,i)=>(
                  <div key={i} className="aic-event">
                    <span className="aic-event-dot" style={{background:'#3b9eff'}}/>
                    <span>Boats: {e.payload?.boats_available} · Kits: {e.payload?.medical_kits} · Food: {e.payload?.food_packs}</span>
                    <span className="aic-event-time">{new Date(e.timestamp).toLocaleTimeString('en-BD',{hour:'2-digit',minute:'2-digit'})}</span>
                  </div>
                ))}
                {!(events||[]).filter(e=>e.channel==='inventory_update').length && (
                  <p className="aic-empty">Waiting for inventory updates…</p>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'agent4' && (
          <div className="agents-two-col">
            <Agent4Panel data={agents?.agent4} />
            <div className="agent-info-card">
              <h3 className="aic-title">🚤 Dispatch Optimization</h3>
              <p className="aic-body">Team assignment + route computation via OSRM. Agent 4 receives dispatch orders, queries OSRM for optimal routes, assigns teams, and publishes route_assignment events back to the dashboard in real-time.</p>
              <div className="aic-pipeline">
                <div className="aic-step">Dispatch Order</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step">OSRM Routing</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step">Team Assign</div>
                <div className="aic-arrow">→</div>
                <div className="aic-step active-step">Route to Map</div>
              </div>
              <div className="aic-feed">
                <p className="aic-feed-title">Live Route Events</p>
                {(events||[]).filter(e=>e.channel==='route_assignment'||e.channel==='dispatch_order').slice(-5).reverse().map((e,i)=>(
                  <div key={i} className="aic-event">
                    <span className="aic-event-dot" style={{background:'#22c55e'}}/>
                    <span>{e.payload?.team_id||e.payload?.volunteer_id||'Team'} → {e.payload?.zone_name||'Zone'} ETA {e.payload?.eta_minutes||'?'}m</span>
                    <span className="aic-event-time">{new Date(e.timestamp).toLocaleTimeString('en-BD',{hour:'2-digit',minute:'2-digit'})}</span>
                  </div>
                ))}
                {!(events||[]).filter(e=>e.channel==='route_assignment'||e.channel==='dispatch_order').length && (
                  <p className="aic-empty">Waiting for dispatch events…</p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Page 3: System Overview ───────────────────────────────────────────────────
function PageSystem({ kpi, agents, events, connected, zones }) {
  const recentEvents = useMemo(() => [...(events||[])].reverse().slice(0,12), [events])

  const AGENT_COLORS = {
    flood_alert:      { color: '#ef4444', label: 'Agent 1 (Environmental)', icon: '🛰' },
    distress_queue:   { color: '#f59e0b', label: 'Agent 2 (Distress)',       icon: '📱' },
    inventory_update: { color: '#3b9eff', label: 'Agent 3 (Resource)',       icon: '📦' },
    route_assignment: { color: '#22c55e', label: 'Agent 4 (Dispatch)',       icon: '🚤' },
    dispatch_order:   { color: '#22c55e', label: 'Agent 4 (Dispatch)',       icon: '🚤' },
    agent_status:     { color: '#a78bfa', label: 'Team Status',              icon: '👥' },
  }

  const eventText = (ev) => {
    const p = ev.payload || {}
    switch(ev.channel) {
      case 'flood_alert':      return `${p.zone_name||'Zone'}: risk ${((p.risk_score||0)*100).toFixed(0)}% — ${p.severity_level||'unknown'} severity. Published FLOOD_RISK alert.`
      case 'distress_queue':   return `Received distress at ${p.district||p.zone_name||'unknown'}. Severity ${p.severity||p.urgency||'?'}/5, credibility ${((p.credibility||0)*100).toFixed(0)}%.`
      case 'inventory_update': return `Allocated resources. Boats: ${p.boats_available}, Kits: ${p.medical_kits}, Food: ${p.food_packs}. Status: ${p.status||'—'}.`
      case 'route_assignment':
      case 'dispatch_order':   return `${p.team_id||p.volunteer_id||'Team'} → ${p.zone_name||'Zone'}. ETA ${p.eta_minutes||'?'} min, ${p.distance_km||'?'} km.`
      case 'agent_status':     return `${p.team_id||p.volunteer_id||'Team'} — ${p.status||'unknown'} at ${p.zone_name||'base'}.`
      default:                 return JSON.stringify(p).slice(0,80)
    }
  }

  // Zone table from flood features
  const zones_list = useMemo(() => {
    const features = zones?.flood?.features || []
    return features.map(f => ({
      name: f.properties?.zone_name || f.properties?.name || 'Unknown',
      risk: f.properties?.risk || f.properties?.risk_score || 0,
      severity: f.properties?.severity_level || f.properties?.risk_level || 'minimal',
    })).sort((a,b) => b.risk - a.risk).slice(0, 7)
  }, [zones])

  const severityColor = (s) => ({critical:'#ef4444',high:'#f59e0b',moderate:'#3b9eff',low:'#22c55e',minimal:'#3d5266'}[s]||'#3d5266')

  return (
    <div className="page-system">
      <div className="page-system-header">
        <div>
          <h2 className="page-title">System Overview</h2>
          <p className="page-subtitle">All agents communicate via Redis pub/sub — every message logged to PostgreSQL</p>
        </div>
        <div className={`sys-status-pill ${connected ? 'sys-status-pill--live' : 'sys-status-pill--off'}`}>
          <span className="sys-status-dot"/>
          {connected ? 'SYSTEM LIVE' : 'OFFLINE'}
        </div>
      </div>

      <div className="sys-grid">
        {/* KPI cards — big version */}
        <div className="sys-kpi-row">
          {[
            { label: 'Zones Monitored',  value: (zones?.flood?.features||[]).length || '—',       color: '#3b9eff', icon: '📍' },
            { label: 'Active Alerts',     value: kpi?.criticalAlerts ?? '—',                        color: '#ef4444', icon: '🚨' },
            { label: 'Distress Reports',  value: (agents?.agent2?.total_reports ?? '—'),            color: '#f59e0b', icon: '📱' },
            { label: 'Teams Deployed',    value: kpi?.deployedTeams != null ? `${kpi.deployedTeams}` : '—', color: '#22c55e', icon: '🚤' },
            { label: 'People Affected',   value: kpi?.affectedPeople ? kpi.affectedPeople.toLocaleString() : '—', color: '#a78bfa', icon: '👥' },
            { label: 'System Status',     value: kpi?.incidentStatus || 'IDLE',                    color: kpi?.incidentStatus==='FLOOD_RESPONSE_ACTIVE'?'#ef4444':'#22c55e', icon: '⚡' },
          ].map((c,i) => (
            <div key={i} className="sys-kpi-card">
              <span className="sys-kpi-icon">{c.icon}</span>
              <span className="sys-kpi-val" style={{color: c.color}}>{c.value}</span>
              <span className="sys-kpi-label">{c.label}</span>
            </div>
          ))}
        </div>

        {/* Live coordination feed */}
        <div className="sys-feed-card">
          <div className="sys-card-header">
            <span>📡 Live Agent Coordination Feed</span>
            <span className="sys-card-sub">Proof that agents communicate via Redis pub/sub</span>
          </div>
          <div className="sys-feed-list">
            {recentEvents.length === 0 && (
              <div className="sys-feed-empty">Waiting for agent events…</div>
            )}
            {recentEvents.map((ev, i) => {
              const meta = AGENT_COLORS[ev.channel] || { color:'#3d5266', label: ev.channel, icon:'•' }
              return (
                <div key={i} className="sys-feed-row" style={{'--feed-color': meta.color}}>
                  <div className="sfr-dot"/>
                  <div className="sfr-body">
                    <div className="sfr-agent">
                      {meta.icon} {meta.label}
                    </div>
                    <div className="sfr-text">{eventText(ev)}</div>
                  </div>
                  <div className="sfr-time">{new Date(ev.timestamp).toLocaleTimeString('en-BD',{hour:'2-digit',minute:'2-digit'})}</div>
                </div>
              )
            })}
          </div>
          <div className="sys-feed-footer">Every message = one Redis pub/sub event logged to PostgreSQL</div>
        </div>

        {/* Zone status table */}
        <div className="sys-zones-card">
          <div className="sys-card-header">
            <span>🗺 Zone Status Table</span>
            <span className="sys-card-sub">Latest flood risk per monitored zone</span>
          </div>
          <table className="sys-zone-table">
            <thead>
              <tr><th>Zone</th><th>Risk Score</th><th>Severity</th><th>Status Bar</th></tr>
            </thead>
            <tbody>
              {zones_list.length === 0 && (
                <tr><td colSpan={4} className="sys-zone-empty">Loading zone data…</td></tr>
              )}
              {zones_list.map((z,i) => (
                <tr key={i}>
                  <td className="szt-name">{z.name}</td>
                  <td className="szt-risk" style={{color: severityColor(z.severity)}}>{(z.risk*100).toFixed(1)}%</td>
                  <td><span className="szt-badge" style={{background: severityColor(z.severity)+'22', color: severityColor(z.severity), border: `1px solid ${severityColor(z.severity)}44`}}>{z.severity}</span></td>
                  <td className="szt-bar-cell">
                    <div className="szt-bar-track"><div className="szt-bar-fill" style={{width:`${Math.round(z.risk*100)}%`, background: severityColor(z.severity)}}/></div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Agent pipeline diagram */}
        <div className="sys-pipeline-card">
          <div className="sys-card-header">
            <span>⚙ Multi-Agent Pipeline</span>
            <span className="sys-card-sub">Data flow: Satellite → People saved</span>
          </div>
          <div className="sys-pipeline">
            {[
              { label:'Satellite SAR\n+ Weather API',  icon:'🛰',  color:'#ef4444', sub:'Agent 1' },
              { label:'Distress Reports\nNLP Analysis', icon:'📱',  color:'#f59e0b', sub:'Agent 2' },
              { label:'Resource\nAllocation',          icon:'📦',  color:'#3b9eff', sub:'Agent 3' },
              { label:'Route\nOptimization',           icon:'🚤',  color:'#22c55e', sub:'Agent 4' },
              { label:'Teams\nDispatched',             icon:'✅',  color:'#a78bfa', sub:'Outcome' },
            ].map((s, i, arr) => (
              <React.Fragment key={i}>
                <div className="sp-node" style={{'--node-color': s.color}}>
                  <div className="sp-icon">{s.icon}</div>
                  <div className="sp-label">{s.label}</div>
                  <div className="sp-sub">{s.sub}</div>
                </div>
                {i < arr.length - 1 && (
                  <div className="sp-arrow">
                    <span>Redis</span>
                    <svg viewBox="0 0 40 12" fill="none"><path d="M0 6 L35 6" stroke="currentColor" strokeWidth="1.5"/><path d="M28 1 L36 6 L28 11" stroke="currentColor" strokeWidth="1.5"/></svg>
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Page 4: Citizen Reporter ──────────────────────────────────────────────────
function PageCitizen() {
  const [form, setForm] = useState({ name: '', phone: '', category: 'Flooding', message: '' })
  const [pin, setPin]   = useState(null)   // { lat, lon }
  const [status, setStatus] = useState(null) // null | 'submitting' | 'ok' | 'error'
  const [errorMsg, setErrorMsg] = useState('')
  const mapRef = React.useRef(null)
  const leafletRef = React.useRef(null)

  // Initialise Leaflet map once
  useEffect(() => {
    if (leafletRef.current) return
    if (!window.L) return
    const map = window.L.map(mapRef.current, {
      center: [24.0, 91.5], zoom: 8,
      zoomControl: true,
    })
    window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(map)
    let marker = null
    map.on('click', (e) => {
      const { lat, lng } = e.latlng
      setPin({ lat: +lat.toFixed(6), lon: +lng.toFixed(6) })
      if (marker) marker.remove()
      marker = window.L.marker([lat, lng]).addTo(map)
        .bindPopup(`📍 ${lat.toFixed(5)}, ${lng.toFixed(5)}`).openPopup()
    })
    leafletRef.current = map
  }, [])

  const field = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const submit = async () => {
    if (!pin)              { setErrorMsg('Click the map to pick a location first.'); setStatus('error'); return }
    if (!form.name.trim()) { setErrorMsg('Name is required.'); setStatus('error'); return }
    if (!form.message.trim()) { setErrorMsg('Please describe the situation.'); setStatus('error'); return }
    setStatus('submitting')
    setErrorMsg('')
    try {
      const res = await fetch('/api/citizen-reports', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, latitude: pin.lat, longitude: pin.lon }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setStatus('ok')
      setForm({ name: '', phone: '', category: 'Flooding', message: '' })
      setPin(null)
    } catch (e) {
      setErrorMsg(e.message)
      setStatus('error')
    }
  }

  const CATEGORIES = ['Flooding', 'Medical Emergency', 'Rescue Needed', 'Road Blockage', 'Building Collapse', 'Fire', 'Other']

  return (
    <div className="page-citizen">
      <div className="page-citizen-header">
        <div>
          <h2 className="page-title">📣 Citizen Reporter</h2>
          <p className="page-subtitle">Click the map to pin your location, fill in the details, and submit a ground-level report.</p>
        </div>
      </div>

      <div className="citizen-layout">
        {/* Map picker */}
        <div className="citizen-map-wrap">
          {!window.L && (
            <div className="citizen-map-loading">Loading map…</div>
          )}
          <div ref={mapRef} className="citizen-map" />
          {pin && (
            <div className="citizen-pin-badge">
              📍 {pin.lat}, {pin.lon}
            </div>
          )}
          {!pin && (
            <div className="citizen-pin-hint">Click anywhere on the map to set your location</div>
          )}
        </div>

        {/* Form */}
        <div className="citizen-form-card">
          {status === 'ok' && (
            <div className="citizen-alert citizen-alert--ok">
              ✅ Report submitted! Our agents will process it shortly.
            </div>
          )}
          {status === 'error' && (
            <div className="citizen-alert citizen-alert--err">
              ⚠ {errorMsg}
            </div>
          )}

          <div className="citizen-field">
            <label className="citizen-label">Your name *</label>
            <input className="citizen-input" value={form.name} onChange={e => field('name', e.target.value)} placeholder="Full name" />
          </div>

          <div className="citizen-field">
            <label className="citizen-label">Phone number</label>
            <input className="citizen-input" value={form.phone} onChange={e => field('phone', e.target.value)} placeholder="+880 …" />
          </div>

          <div className="citizen-field">
            <label className="citizen-label">Emergency type</label>
            <select className="citizen-input citizen-select" value={form.category} onChange={e => field('category', e.target.value)}>
              {CATEGORIES.map(c => <option key={c}>{c}</option>)}
            </select>
          </div>

          <div className="citizen-field">
            <label className="citizen-label">Describe the situation *</label>
            <textarea className="citizen-input citizen-textarea" rows={4} value={form.message}
              onChange={e => field('message', e.target.value)}
              placeholder="How many people? Water level? Access routes blocked?" />
          </div>

          <button
            className="citizen-submit"
            onClick={submit}
            disabled={status === 'submitting'}
          >
            {status === 'submitting' ? 'Submitting…' : '🚨 Submit Report'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Page 5: Risk Validation Results ───────────────────────────────────────────
function PageReports() {
  const [reports, setReports] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')  // all | pending | validated
  const [search, setSearch] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/citizen-reports?limit=50')
      if (res.ok) {
        const data = await res.json()
        setReports(data.reports || [])
      }
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { load() }, [load])

  const filtered = useMemo(() => {
    let r = reports
    if (filter === 'pending')   r = r.filter(x => x.status === 'pending')
    if (filter === 'validated') r = r.filter(x => x.status !== 'pending')
    if (search.trim()) {
      const q = search.toLowerCase()
      r = r.filter(x => (x.name||'').toLowerCase().includes(q) || (x.category||'').toLowerCase().includes(q) || (x.message||'').toLowerCase().includes(q))
    }
    return r
  }, [reports, filter, search])

  const riskColor = (lvl) => ({ low:'#22c55e', moderate:'#f59e0b', high:'#ef4444', critical:'#ef4444', extreme:'#ef4444' }[lvl] || '#3d5266')
  const statusBg  = (s) => s === 'pending' ? '#3d5266' : '#22c55e22'
  const statusCol = (s) => s === 'pending' ? '#8aa0bc' : '#22c55e'

  return (
    <div className="page-reports">
      <div className="page-reports-header">
        <div>
          <h2 className="page-title">🛡 Citizen Reports &amp; Risk Validation</h2>
          <p className="page-subtitle">Ground-level reports submitted by citizens, cross-validated by Agent 1's flood risk model.</p>
        </div>
        <button className="btn btn--blue" onClick={load} disabled={loading}>
          {loading ? '…' : '↻ Refresh'}
        </button>
      </div>

      {/* Toolbar */}
      <div className="reports-toolbar">
        <div className="reports-filters">
          {['all','pending','validated'].map(f => (
            <button key={f}
              className={`filter-tab ${filter === f ? 'filter-tab--active' : ''}`}
              onClick={() => setFilter(f)}>
              {f === 'all' ? 'All' : f === 'pending' ? '⏳ Pending' : '✅ Validated'}
            </button>
          ))}
        </div>
        <input
          className="reports-search"
          placeholder="Search name, category, message…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {/* Cards */}
      {loading && <div className="reports-loading">Loading reports…</div>}
      {!loading && filtered.length === 0 && (
        <div className="reports-empty">
          {reports.length === 0
            ? 'No citizen reports yet. Use the Citizen Reporter page to submit one.'
            : 'No reports match the current filter.'}
        </div>
      )}

      <div className="reports-grid">
        {filtered.map((r, i) => (
          <div key={i} className="report-card-v2">
            <div className="rcv2-header">
              <span className="rcv2-category">{r.category}</span>
              <span className="rcv2-status" style={{ background: statusBg(r.status), color: statusCol(r.status) }}>
                {r.status || 'pending'}
              </span>
              <span className="rcv2-time">{r.created_at ? new Date(r.created_at).toLocaleString('en-BD',{month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'}) : '—'}</span>
            </div>

            <div className="rcv2-body">
              <div className="rcv2-meta">
                <span className="rcv2-name">👤 {r.name || '—'}</span>
                {r.phone && <span className="rcv2-phone">📞 {r.phone}</span>}
                {r.latitude != null && <span className="rcv2-coords">📍 {(+r.latitude).toFixed(4)}, {(+r.longitude).toFixed(4)}</span>}
              </div>
              <p className="rcv2-message">"{r.message}"</p>
            </div>

            {/* Validation block — shown when Agent 1 has processed it */}
            {r.flood_risk_level && (
              <div className="rcv2-validation">
                <div className="rcv2-val-title">Agent 1 Risk Assessment</div>
                <div className="rcv2-val-row">
                  <span className="rcv2-val-label">Flood risk</span>
                  <span className="rcv2-val-badge" style={{ color: riskColor(r.flood_risk_level), borderColor: riskColor(r.flood_risk_level)+'44', background: riskColor(r.flood_risk_level)+'15' }}>
                    {(r.flood_risk_level||'').toUpperCase()}
                  </span>
                </div>
                {r.risk_score != null && (
                  <div className="rcv2-val-row">
                    <span className="rcv2-val-label">Risk score</span>
                    <div className="rcv2-score-bar">
                      <div className="rcv2-score-fill" style={{ width: `${r.risk_score}%`, background: riskColor(r.flood_risk_level) }}/>
                      <span className="rcv2-score-num">{r.risk_score}/100</span>
                    </div>
                  </div>
                )}
                {r.validation_notes && (
                  <p className="rcv2-notes">{r.validation_notes}</p>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Root App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [activePage, setActivePage] = useState('map')
  const [activeLayers, setActiveLayers] = useState({ flood:true, rain:false, teams:true, pins:true, river:true, routes:true })
  const [replayState, setReplayState] = useState({ mode:'LIVE', enabled:false, running:false, paused:false, tick:0, scenario_date:'' })
  const [replayBusy, setReplayBusy] = useState(false)
  const [eventResetToken, setEventResetToken] = useState(0)

  const isReplay = replayState.mode !== 'LIVE'

  const { events, connected } = useWebSocket('ws://localhost:8005/ws', { paused: isReplay, resetToken: eventResetToken })
  const { kpi, agents, refreshDashboard } = useDashboard({ paused: isReplay, events })
  const { zones } = useZones()

  const scenarioLabel = useMemo(() => formatScenarioLabel(replayState.mode), [replayState.mode])
  const scenarioTimeLabel = useMemo(() => formatScenarioTimestamp(replayState.scenario_date), [replayState.scenario_date])

  const toggleLayer = (layer) => setActiveLayers(prev => ({ ...prev, [layer]: !prev[layer] }))

  const loadReplayStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/replay/status')
      if (!res.ok) return
      setReplayState(await res.json())
    } catch {}
  }, [])

  const sendReplayAction = useCallback(async (path) => {
    try {
      setReplayBusy(true)
      const res = await fetch(path, { method: 'POST' })
      if (!res.ok) return
      const json = await res.json()
      setReplayState(json)
      if (path === '/api/replay/stop') {
        setEventResetToken(p => p + 1)
        await refreshDashboard()
      }
    } catch {} finally { setReplayBusy(false) }
  }, [refreshDashboard])

  useEffect(() => {
    loadReplayStatus()
    const t = setInterval(loadReplayStatus, 30000)
    return () => clearInterval(t)
  }, [loadReplayStatus])

  useEffect(() => { if (!isReplay) refreshDashboard() }, [isReplay, refreshDashboard])

  const NAV_PAGES = [
    { id: 'map',     icon: '🗺',  label: 'Live Map' },
    { id: 'agents',  icon: '🤖', label: 'Agent Panels' },
    { id: 'system',  icon: '⚙',  label: 'System Overview' },
    { id: 'citizen', icon: '📣', label: 'Citizen Reporter' },
    { id: 'reports', icon: '🛡',  label: 'Reports' },
  ]

  return (
    <div className="app-shell">
      <KPIBar kpi={kpi} connected={connected} />

      {/* Page nav */}
      <nav className="page-nav">
        {NAV_PAGES.map(p => (
          <button
            key={p.id}
            className={`page-nav-btn ${activePage === p.id ? 'page-nav-btn--active' : ''}`}
            onClick={() => setActivePage(p.id)}
          >
            <span>{p.icon}</span>
            <span>{p.label}</span>
          </button>
        ))}
        <div className="page-nav-spacer" />
        <button className="page-nav-refresh" onClick={refreshDashboard} title="Refresh data">
          ↻ Refresh
        </button>
      </nav>

      {/* Pages */}
      <div className="page-content">
        {activePage === 'map' && (
          <PageMap
            zones={zones} activeLayers={activeLayers} agents={agents} events={events}
            connected={connected} toggleLayer={toggleLayer}
            replayState={replayState} replayBusy={replayBusy} isReplay={isReplay}
            scenarioLabel={scenarioLabel} scenarioTimeLabel={scenarioTimeLabel}
            sendReplayAction={sendReplayAction}
          />
        )}
        {activePage === 'agents' && <PageAgents agents={agents} events={events} />}
        {activePage === 'system' && <PageSystem kpi={kpi} agents={agents} events={events} connected={connected} zones={zones} />}
        {activePage === 'citizen' && <PageCitizen />}
        {activePage === 'reports' && <PageReports />}
      </div>
    </div>
  )
}