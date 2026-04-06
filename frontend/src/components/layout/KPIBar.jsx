import React from 'react'

const STATUS_COLOR = {
  FLOOD_RESPONSE_ACTIVE: 'var(--danger)',
  ACTIVE: 'var(--warning)',
  IDLE: 'var(--muted)',
}

export default function KPIBar({ kpi, connected }) {
  const {
    incidentStatus = 'IDLE',
    activeZones = 0,
    criticalAlerts = 0,
    deployedTeams = 0,
    affectedPeople = 0,
    severity = 0,
    confidence = 0,
  } = kpi || {}

  const statusColor = STATUS_COLOR[incidentStatus] || 'var(--muted)'

  return (
    <header className="kpi-bar">
      <div className="kpi-brand">
        <span className="kpi-brand-icon">🌊</span>
        <span className="kpi-brand-name">FLOOD BRAIN</span>
        <span className="kpi-brand-sub">Sylhet Division Response</span>
      </div>

      <div className="kpi-metrics">
        <KPIChip
          label="Status"
          value={incidentStatus}
          color={statusColor}
          blink={incidentStatus === 'FLOOD_RESPONSE_ACTIVE'}
        />
        <KPIChip label="Active Zones" value={activeZones} color="var(--accent)" />
        <KPIChip label="Critical Alerts" value={criticalAlerts} color="var(--danger)" />
        <KPIChip label="Teams Deployed" value={deployedTeams} color="var(--info)" />
        <KPIChip label="Affected People" value={affectedPeople.toLocaleString()} color="var(--warning)" />
        <KPIChip label="Severity" value={`${severity}/5`} color={severity >= 4 ? 'var(--danger)' : 'var(--warning)'} />
        <KPIChip label="Confidence" value={`${Math.round(confidence * 100)}%`} color="var(--success)" />
      </div>

      <div className="kpi-status-dot-wrap">
        <span className={`kpi-ws-dot ${connected ? 'connected' : 'disconnected'}`} />
        <span className="kpi-ws-label">{connected ? 'LIVE' : 'OFFLINE'}</span>
      </div>
    </header>
  )
}

function KPIChip({ label, value, color, blink }) {
  return (
    <div className={`kpi-chip ${blink ? 'kpi-chip--blink' : ''}`}>
      <span className="kpi-chip-value" style={{ color }}>{value}</span>
      <span className="kpi-chip-label">{label}</span>
    </div>
  )
}
