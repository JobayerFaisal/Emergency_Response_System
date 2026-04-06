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

function RiskBar({ label, value = 0 }) {
  const pct = Math.min(100, Math.round(value * 100))
  const color =
    pct >= 80 ? 'var(--danger)' :
    pct >= 50 ? 'var(--warning)' :
    'var(--success)'

  return (
    <div className="risk-bar-row">
      <span className="risk-bar-label">{label}</span>
      <div className="risk-bar-track">
        <div
          className="risk-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
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

export default function Agent1Panel({ data }) {
  const {
    detected = false,
    severity = 0,
    confidence = 0,
    trend = 'stable',
    risk_factors = [],
    evidence_refs = [],
    weather = {},
    scores = {},
  } = data || {}

  return (
    <div className="panel agent1-panel">
      <div className="panel-header">
        <span className="panel-icon">🛰</span>
        <span className="panel-title">Environmental Intelligence</span>
        <span className={`panel-badge ${detected ? 'panel-badge--danger' : 'panel-badge--ok'}`}>
          {detected ? 'FLOOD DETECTED' : 'MONITORING'}
        </span>
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
        <WeatherCard
          label="Rainfall"
          value={weather.rainfall_mm}
          unit=" mm"
          danger={(weather.rainfall_mm || 0) > 100}
        />
        <WeatherCard
          label="River Level"
          value={weather.river_level_m}
          unit=" m"
          danger={(weather.river_level_m || 0) > (weather.danger_level_m || 99)}
        />
        <WeatherCard
          label="Danger Level"
          value={weather.danger_level_m}
          unit=" m"
        />
      </div>

      {/* 9-factor risk bars */}
      <h4 className="panel-section-title">9-Factor Risk Analysis</h4>
      <div className="risk-bars">
        {RISK_FACTORS.map(rf => (
          <RiskBar key={rf.key} label={rf.label} value={scores[rf.key] ?? 0} />
        ))}
      </div>

      {/* Risk factors */}
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
    </div>
  )
}
