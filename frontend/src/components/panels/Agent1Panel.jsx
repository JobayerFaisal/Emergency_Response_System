import React from 'react'

const RISK_FACTORS = [
  { key: 'rainfall_score',      label: 'Rainfall Intensity' },
  { key: 'river_level_score',   label: 'River Level' },
  { key: 'water_extent_score',  label: 'Water Extent (SAR)' },
  { key: 'social_score',        label: 'Social Signal Density' },
  { key: 'depth_score',         label: 'Flood Depth Est.' },
  { key: 'trend_score',         label: 'Trend (Rising/Stable)' },
  { key: 'satellite_conf',      label: 'Satellite Confidence' },
  { key: 'credibility_score',   label: 'Report Credibility' },
  { key: 'cluster_score',       label: 'Cluster Density' },
]

const SEVERITY_COLOR = {
  critical: 'var(--danger)',
  high:     'var(--warning)',
  moderate: 'var(--info)',
  low:      'var(--success)',
  minimal:  'var(--muted)',
}

function RiskBar({ label, value = 0 }) {
  const pct   = Math.min(100, Math.round(value * 100))
  const color = pct >= 80 ? 'var(--danger)' : pct >= 50 ? 'var(--warning)' : 'var(--success)'
  return (
    <div className="risk-bar-row">
      <span className="risk-bar-label">{label}</span>
      <div className="risk-bar-track">
        <div className="risk-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="risk-bar-pct" style={{ color }}>{pct}%</span>
    </div>
  )
}

function WeatherCard({ label, value, unit, danger }) {
  return (
    <div className={`weather-card ${danger ? 'weather-card--danger' : ''}`}>
      <span className="weather-value">{value ?? '—'}<span className="weather-unit">{unit}</span></span>
      <span className="weather-label">{label}</span>
    </div>
  )
}

// Small "data freshness" badge shown next to the panel title
function FreshnessBadge({ ageMinutes }) {
  if (ageMinutes == null) return null
  const fresh = ageMinutes < 10
  const stale = ageMinutes >= 30
  const color = fresh ? 'var(--success)' : stale ? 'var(--danger)' : 'var(--warning)'
  const label = ageMinutes < 1
    ? '< 1 min ago'
    : ageMinutes < 60
    ? `${Math.round(ageMinutes)} min ago`
    : `${(ageMinutes / 60).toFixed(1)} h ago`
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, color,
      background: color + '22',
      border: `1px solid ${color}44`,
      borderRadius: 999, padding: '2px 8px', marginLeft: 8,
    }}>
      ⏱ {label}
    </span>
  )
}

// Zone risk table — shown when all_zones is populated
function ZoneTable({ zones }) {
  if (!zones || zones.length === 0) return null
  const sorted = [...zones].sort((a, b) => b.risk_score - a.risk_score)
  return (
    <div style={{ marginTop: 8 }}>
      <h4 className="panel-section-title">All Monitored Zones</h4>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {sorted.map((z, i) => {
          const pct   = Math.round((z.risk_score || 0) * 100)
          const color = SEVERITY_COLOR[z.severity] || 'var(--muted)'
          return (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '5px 8px', borderRadius: 8,
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.07)',
            }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)', minWidth: 90 }}>{z.zone_name}</span>
              <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.08)' }}>
                <div style={{ width: `${pct}%`, height: '100%', borderRadius: 3, background: color, transition: 'width 0.4s' }} />
              </div>
              <span style={{ fontSize: 11, fontWeight: 700, color, minWidth: 36, textAlign: 'right' }}>{pct}%</span>
              <span style={{
                fontSize: 10, fontWeight: 600, color,
                background: color + '22', border: `1px solid ${color}44`,
                borderRadius: 999, padding: '1px 6px',
              }}>{z.severity}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function Agent1Panel({ data }) {
  const {
    detected         = false,
    severity         = 0,
    confidence       = 0,
    trend            = 'stable',
    risk_factors     = [],
    evidence_refs    = [],
    weather          = {},
    scores           = {},
    data_age_minutes = null,
    all_zones        = [],
    no_recent_data   = false,
  } = data || {}

  // FIX — NO RECENT DATA STATE:
  // When Agent 1 has not run in the last 6 hours (e.g. on first boot, or after
  // a replay session where the old DB rows are filtered out), the backend now
  // returns no_recent_data=true instead of surfacing stale 2022 values.
  // Show a clear "waiting" state rather than misleading numbers.
  if (no_recent_data || !data) {
    return (
      <div className="panel agent1-panel">
        <div className="panel-header">
          <span className="panel-icon">🛰</span>
          <span className="panel-title">Environmental Intelligence</span>
          <span className="panel-badge panel-badge--ok">MONITORING</span>
        </div>
        <div className="panel-empty" style={{ padding: '32px 16px', textAlign: 'center' }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>🛰</div>
          <div style={{ fontWeight: 700, color: 'var(--text)', marginBottom: 6 }}>
            Waiting for Agent 1 data
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 }}>
            No predictions in the last 6 hours.<br />
            Agent 1 runs on its configured interval (default 5 min).<br />
            Data will appear here once the first cycle completes.
          </div>
          {all_zones.length > 0 && <ZoneTable zones={all_zones} />}
        </div>
      </div>
    )
  }

  return (
    <div className="panel agent1-panel">
      <div className="panel-header">
        <span className="panel-icon">🛰</span>
        <span className="panel-title">Environmental Intelligence</span>
        <span className={`panel-badge ${detected ? 'panel-badge--danger' : 'panel-badge--ok'}`}>
          {detected ? 'FLOOD DETECTED' : 'MONITORING'}
        </span>
        {/* FIX: Show data freshness so operators know how stale the reading is */}
        <FreshnessBadge ageMinutes={data_age_minutes} />
      </div>

      {/* Detection summary */}
      <div className="agent1-summary">
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: severity >= 4 ? 'var(--danger)' : 'var(--warning)' }}>
            {severity}<span className="summary-stat-max">/5</span>
          </span>
          <span className="summary-stat-label">Severity</span>
        </div>
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: 'var(--accent)' }}>
            {Math.round(confidence * 100)}<span className="summary-stat-max">%</span>
          </span>
          <span className="summary-stat-label">Confidence</span>
        </div>
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: trend === 'rising' ? 'var(--danger)' : 'var(--success)' }}>
            {trend === 'rising' ? '↑' : trend === 'falling' ? '↓' : '→'}
          </span>
          <span className="summary-stat-label">Trend</span>
        </div>
      </div>

      {/* Weather signals */}
      <h4 className="panel-section-title">Weather Signals</h4>
      <div className="weather-cards">
        <WeatherCard label="Rainfall"     value={weather.rainfall_mm}    unit=" mm" danger={(weather.rainfall_mm || 0) > 100} />
        <WeatherCard label="River Level"  value={weather.river_level_m}  unit=" m"  danger={(weather.river_level_m || 0) > (weather.danger_level_m || 99)} />
        <WeatherCard label="Danger Level" value={weather.danger_level_m} unit=" m" />
      </div>

      {/* 9-factor risk bars */}
      <h4 className="panel-section-title">9-Factor Risk Analysis</h4>
      <div className="risk-bars">
        {RISK_FACTORS.map(rf => (
          <RiskBar key={rf.key} label={rf.label} value={scores[rf.key] ?? 0} />
        ))}
      </div>

      {/* Recommended actions (from predictor) */}
      {risk_factors.length > 0 && (
        <>
          <h4 className="panel-section-title">Active Risk Factors</h4>
          <ul className="risk-factor-list">
            {risk_factors.map((rf, i) => (
              <li key={i} className="risk-factor-item">⚠ {rf}</li>
            ))}
          </ul>
        </>
      )}

      {/* Evidence */}
      {evidence_refs.length > 0 && (
        <>
          <h4 className="panel-section-title">Evidence Sources</h4>
          <ul className="evidence-list">
            {evidence_refs.map((e, i) => (
              <li key={i} className="evidence-item">📎 {e}</li>
            ))}
          </ul>
        </>
      )}

      {/* All-zones table */}
      <ZoneTable zones={all_zones} />
    </div>
  )
}
