import React, { useRef, useEffect } from 'react'

const EVENT_ICON = {
  FLOOD_DETECTED: '🔴',
  FLOOD_CONFIRMED: '🌊',
  FLOOD_INTELLIGENCE_UPDATED: '📡',
  MISSION_REQUESTED: '📋',
  MISSION_CREATED: '✅',
  MISSION_ASSIGNED: '👤',
  MISSION_COMPLETED: '🏁',
  MISSION_FAILED: '❌',
  MISSION_EN_ROUTE: '🚤',
  MISSION_ACTIVE: '⚡',
  SOCIAL_REPORT_RECEIVED: '📱',
  WEATHER_UPDATE: '🌧',
  SATELLITE_UPDATE: '🛰',
  CLUSTERS_UPDATED: '📍',
  INCIDENT_STARTED: '🚨',
  VOLUNTEER_UPDATE: '🙋',
}

const SEVERITY_CLASS = {
  FLOOD_DETECTED: 'event--danger',
  FLOOD_CONFIRMED: 'event--danger',
  MISSION_FAILED: 'event--warn',
  MISSION_COMPLETED: 'event--success',
  INCIDENT_STARTED: 'event--danger',
  WEATHER_UPDATE: 'event--info',
  SATELLITE_UPDATE: 'event--info',
}

function formatTime(ts) {
  if (!ts) return ''
  const d = new Date(ts)
  return d.toLocaleTimeString('en-BD', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function truncate(str, n = 60) {
  if (!str) return ''
  return str.length > n ? str.slice(0, n) + '…' : str
}

export default function AgentFeed({ events = [] }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  if (!events.length) {
    return (
      <div className="agent-feed agent-feed--empty">
        <span className="feed-empty-icon">📡</span>
        <p>Waiting for live events…</p>
      </div>
    )
  }

  return (
    <div className="agent-feed">
      {[...events].reverse().map((ev, i) => {
        const icon = EVENT_ICON[ev.type] || '•'
        const cls = SEVERITY_CLASS[ev.type] || ''
        return (
          <div key={i} className={`feed-event ${cls}`}>
            <span className="feed-icon">{icon}</span>
            <div className="feed-body">
              <span className="feed-type">{ev.type}</span>
              {ev.payload?.severity && (
                <span className="feed-badge feed-badge--severity">
                  SEV {ev.payload.severity}
                </span>
              )}
              {ev.payload?.mission_id && (
                <span className="feed-badge feed-badge--mission">
                  {String(ev.payload.mission_id).slice(0, 8)}
                </span>
              )}
              {ev.payload?.volunteer_id && (
                <span className="feed-badge feed-badge--volunteer">
                  {ev.payload.volunteer_id}
                </span>
              )}
              {ev.payload?.text && (
                <p className="feed-text">{truncate(ev.payload.text)}</p>
              )}
              {ev.payload?.risk_factors?.length > 0 && (
                <p className="feed-text feed-text--risk">
                  ⚠ {ev.payload.risk_factors.slice(0, 2).join(' · ')}
                </p>
              )}
            </div>
            <span className="feed-time">{formatTime(ev.timestamp)}</span>
          </div>
        )
      })}
      <div ref={bottomRef} />
    </div>
  )
}
