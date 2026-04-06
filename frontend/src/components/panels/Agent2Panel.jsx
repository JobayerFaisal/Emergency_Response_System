import React, { useState } from 'react'

const URGENCY_LABEL = ['', 'Low', 'Low-Med', 'Medium', 'High', 'Critical']
const URGENCY_COLOR = ['', 'var(--success)', 'var(--success)', 'var(--warning)', 'var(--danger)', 'var(--danger)']

function UrgencyBadge({ urgency }) {
  return (
    <span
      className="urgency-badge"
      style={{ background: URGENCY_COLOR[urgency] || 'var(--muted)' }}
    >
      {urgency}/5 {URGENCY_LABEL[urgency] || ''}
    </span>
  )
}

function CredibilityBar({ value }) {
  const pct = Math.round(value * 100)
  return (
    <div className="cred-bar-wrap">
      <div className="cred-bar" style={{ width: `${pct}%` }} />
      <span className="cred-pct">{pct}%</span>
    </div>
  )
}

export default function Agent2Panel({ data }) {
  const {
    reports = [],
    clusters = {},
    total_reports = 0,
    critical_reports = 0,
  } = data || {}

  const [filter, setFilter] = useState('all')

  const filtered = filter === 'critical'
    ? reports.filter(r => r.urgency >= 5)
    : filter === 'high'
    ? reports.filter(r => r.urgency >= 4)
    : reports

  return (
    <div className="panel agent2-panel">
      <div className="panel-header">
        <span className="panel-icon">📱</span>
        <span className="panel-title">Distress Intelligence</span>
        <span className="panel-badge panel-badge--info">{total_reports} Reports</span>
      </div>

      {/* Summary stats */}
      <div className="agent2-summary">
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: 'var(--danger)' }}>{critical_reports}</span>
          <span className="summary-stat-label">Critical</span>
        </div>
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: 'var(--accent)' }}>
            {Object.keys(clusters).length}
          </span>
          <span className="summary-stat-label">Clusters</span>
        </div>
        <div className="summary-stat">
          <span className="summary-stat-val" style={{ color: 'var(--warning)' }}>{total_reports}</span>
          <span className="summary-stat-label">Total</span>
        </div>
      </div>

      {/* Filter tabs */}
      <div className="filter-tabs">
        {['all', 'high', 'critical'].map(f => (
          <button
            key={f}
            className={`filter-tab ${filter === f ? 'filter-tab--active' : ''}`}
            onClick={() => setFilter(f)}
          >
            {f === 'all' ? 'All' : f === 'high' ? 'High (4+)' : '🆘 Critical (5)'}
          </button>
        ))}
      </div>

      {/* Reports table */}
      <div className="reports-list">
        {filtered.length === 0 && (
          <div className="panel-empty">No reports match this filter.</div>
        )}
        {filtered.map((r, i) => (
          <div key={i} className={`report-card ${r.urgency >= 5 ? 'report-card--critical' : ''}`}>
            <div className="report-card-top">
              <UrgencyBadge urgency={r.urgency} />
              <span className="report-district">{r.district || r.division || '—'}</span>
              <span className="report-id">{r.report_id?.slice(0, 12)}</span>
            </div>
            <p className="report-text">"{r.text}"</p>
            <div className="report-card-bottom">
              <span className="report-credibility-label">Credibility</span>
              <CredibilityBar value={r.credibility || 0} />
              <span className="report-coords">
                {r.lat?.toFixed(4)}, {r.lon?.toFixed(4)}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Cluster summary */}
      {Object.keys(clusters).length > 0 && (
        <>
          <h4 className="panel-section-title">Incident Clusters</h4>
          {Object.entries(clusters).map(([cid, reports_in_cluster]) => {
            const count = Array.isArray(reports_in_cluster) ? reports_in_cluster.length : reports_in_cluster
            return (
              <div key={cid} className="cluster-row">
                <span className="cluster-id">Cluster {cid}</span>
                <span className="cluster-count">{count} reports</span>
              </div>
            )
          })}
        </>
      )}
    </div>
  )
}
