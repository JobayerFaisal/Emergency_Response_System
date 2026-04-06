import React from 'react'

const STATUS_COLOR = {
  CREATED:   'var(--muted)',
  ASSIGNED:  'var(--accent)',
  EN_ROUTE:  'var(--info)',
  ACTIVE:    'var(--warning)',
  COMPLETED: 'var(--success)',
  FAILED:    'var(--danger)',
}

const STATUS_ICON = {
  CREATED:   '📋',
  ASSIGNED:  '👤',
  EN_ROUTE:  '🚤',
  ACTIVE:    '⚡',
  COMPLETED: '✅',
  FAILED:    '❌',
}

const PRIORITY_COLOR = {
  CRITICAL: 'var(--danger)',
  HIGH:     'var(--warning)',
  MEDIUM:   'var(--info)',
  LOW:      'var(--muted)',
}

function MissionRow({ mission }) {
  const statusColor = STATUS_COLOR[mission.status] || 'var(--muted)'
  const priorityColor = PRIORITY_COLOR[mission.priority] || 'var(--muted)'

  return (
    <div className="mission-row">
      <div className="mission-row-left">
        <span className="mission-status-icon">{STATUS_ICON[mission.status] || '•'}</span>
        <div className="mission-info">
          <span className="mission-id">{mission.mission_id?.slice(0, 8)}</span>
          <span className="mission-type">{mission.type}</span>
        </div>
      </div>
      <div className="mission-row-center">
        <span className="mission-district">{mission.district || '—'}</span>
        <span className="mission-volunteer">
          {mission.assigned_volunteer ? `👤 ${mission.assigned_volunteer}` : 'Unassigned'}
        </span>
      </div>
      <div className="mission-row-right">
        <span className="mission-priority" style={{ color: priorityColor }}>
          {mission.priority}
        </span>
        <span className="mission-status" style={{ color: statusColor }}>
          {mission.status}
        </span>
        {mission.eta_minutes != null && (
          <span className="mission-eta">ETA {mission.eta_minutes}m</span>
        )}
      </div>
    </div>
  )
}

function MissionPipeline({ missions }) {
  const statuses = ['CREATED', 'ASSIGNED', 'EN_ROUTE', 'ACTIVE', 'COMPLETED']
  const counts = {}
  statuses.forEach(s => {
    counts[s] = missions.filter(m => m.status === s).length
  })

  return (
    <div className="pipeline">
      {statuses.map((s, i) => (
        <React.Fragment key={s}>
          <div className="pipeline-stage">
            <span className="pipeline-icon">{STATUS_ICON[s]}</span>
            <span className="pipeline-count" style={{ color: STATUS_COLOR[s] }}>{counts[s]}</span>
            <span className="pipeline-label">{s.replace('_', ' ')}</span>
          </div>
          {i < statuses.length - 1 && <span className="pipeline-arrow">›</span>}
        </React.Fragment>
      ))}
    </div>
  )
}

export default function Agent4Panel({ data }) {
  const {
    missions = [],
    total_missions = 0,
    active_missions = 0,
    completed_missions = 0,
    failed_missions = 0,
  } = data || {}

  return (
    <div className="panel agent4-panel">
      <div className="panel-header">
        <span className="panel-icon">🚤</span>
        <span className="panel-title">Dispatch Optimization</span>
        <span className="panel-badge panel-badge--info">{total_missions} Missions</span>
      </div>

      {/* Summary stats */}
      <div className="agent4-summary">
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: 'var(--accent)' }}>{total_missions}</span>
          <span className="summary-stat-label">Total</span>
        </div>
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: 'var(--warning)' }}>{active_missions}</span>
          <span className="summary-stat-label">Active</span>
        </div>
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: 'var(--success)' }}>{completed_missions}</span>
          <span className="summary-stat-label">Done</span>
        </div>
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: 'var(--danger)' }}>{failed_missions}</span>
          <span className="summary-stat-label">Failed</span>
        </div>
      </div>

      {/* Pipeline visualization */}
      {missions.length > 0 && (
        <>
          <h4 className="panel-section-title">Mission Pipeline</h4>
          <MissionPipeline missions={missions} />
        </>
      )}

      {/* Mission table */}
      <h4 className="panel-section-title">Assignment Table</h4>
      {missions.length === 0 && (
        <div className="panel-empty">No missions dispatched yet.</div>
      )}
      <div className="mission-list">
        {missions.map((m, i) => (
          <MissionRow key={i} mission={m} />
        ))}
      </div>
    </div>
  )
}
