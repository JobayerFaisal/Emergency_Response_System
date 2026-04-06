import React from 'react'

const RESOURCE_ICONS = {
  boat:      '🚤',
  food:      '🍱',
  medicine:  '💊',
  shelter:   '⛺',
  water:     '💧',
  rescuer:   '🧑‍🚒',
  vehicle:   '🚐',
}

function InventoryBar({ type, available, total, unit = '' }) {
  const pct = total > 0 ? Math.round((available / total) * 100) : 0
  const color =
    pct <= 20 ? 'var(--danger)' :
    pct <= 50 ? 'var(--warning)' :
    'var(--success)'
  const icon = RESOURCE_ICONS[type.toLowerCase()] || '📦'

  return (
    <div className="inventory-row">
      <span className="inventory-icon">{icon}</span>
      <span className="inventory-label">{type}</span>
      <div className="inventory-track">
        <div className="inventory-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="inventory-count" style={{ color }}>
        {available}{unit}<span className="inventory-total">/{total}{unit}</span>
      </span>
    </div>
  )
}

function VolunteerCard({ v }) {
  return (
    <div className={`volunteer-card ${v.available ? 'volunteer-card--available' : 'volunteer-card--deployed'}`}>
      <div className="volunteer-card-top">
        <span className="volunteer-id">{v.volunteer_id}</span>
        <span className={`volunteer-status ${v.available ? 'vs-available' : 'vs-deployed'}`}>
          {v.available ? '● Available' : '● Deployed'}
        </span>
      </div>
      <div className="volunteer-card-body">
        <span className="volunteer-district">{v.district}</span>
        <span className="volunteer-skills">{v.skills?.join(', ')}</span>
        {v.equipment?.length > 0 && (
          <span className="volunteer-equip">🎒 {v.equipment.join(', ')}</span>
        )}
      </div>
    </div>
  )
}

export default function Agent3Panel({ data }) {
  const {
    inventory = [],
    volunteers = [],
    total_volunteers = 0,
    available_volunteers = 0,
    deployed_volunteers = 0,
  } = data || {}

  return (
    <div className="panel agent3-panel">
      <div className="panel-header">
        <span className="panel-icon">📦</span>
        <span className="panel-title">Resource Management</span>
        <span className="panel-badge panel-badge--info">{available_volunteers} Available</span>
      </div>

      {/* Volunteer summary */}
      <div className="agent3-summary">
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: 'var(--accent)' }}>{total_volunteers}</span>
          <span className="summary-stat-label">Total Teams</span>
        </div>
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: 'var(--success)' }}>{available_volunteers}</span>
          <span className="summary-stat-label">Available</span>
        </div>
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: 'var(--warning)' }}>{deployed_volunteers}</span>
          <span className="summary-stat-label">Deployed</span>
        </div>
      </div>

      {/* Resource inventory bars */}
      <h4 className="panel-section-title">Inventory</h4>
      {inventory.length === 0 && (
        <div className="panel-empty">No inventory data received yet.</div>
      )}
      <div className="inventory-list">
        {inventory.map((item, i) => (
          <InventoryBar
            key={i}
            type={item.type}
            available={item.available}
            total={item.total}
            unit={item.unit || ''}
          />
        ))}
      </div>

      {/* Volunteer cards */}
      <h4 className="panel-section-title">Volunteer Roster</h4>
      {volunteers.length === 0 && (
        <div className="panel-empty">No volunteer data received yet.</div>
      )}
      <div className="volunteer-list">
        {volunteers.map((v, i) => (
          <VolunteerCard key={i} v={v} />
        ))}
      </div>
    </div>
  )
}
