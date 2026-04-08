import { useMemo } from 'react'

function getEventType(event) {
  if (event.channel === 'distress_queue') return 'distress'
  if (event.channel === 'agent_status') return 'team'
  if (event.channel === 'route_assignment') return 'route'
  if (event.channel === 'inventory_update') return 'resource'
  return 'system'
}

function getColor(type) {
  switch (type) {
    case 'distress': return '#ff4d4f'
    case 'team': return '#40c4ff'
    case 'route': return '#ffd166'
    case 'resource': return '#73d13d'
    default: return '#8b9bb4'
  }
}

function formatMessage(event) {
  const p = event.payload || {}

  switch (getEventType(event)) {
    case 'distress':
      return `🆘 Distress (${p.zone_name}) — ${p.text}`

    case 'team':
      return `🚑 Team ${p.team_id} → ${p.status}`

    case 'route':
      return `🧭 Route assigned → ${p.zone_name} (ETA ${p.eta_minutes}m)`

    case 'resource':
      return `📦 Inventory update (${p.zone_name})`

    default:
      return `${event.channel}`
  }
}

function formatTime(ts) {
  try {
    return new Date(ts).toLocaleTimeString()
  } catch {
    return ''
  }
}

/**
 * AgentFeed
 * Live event stream from WebSocket
 */
export default function AgentFeed({ events = [] }) {

  // sort latest first
  const sorted = useMemo(() => {
    return [...events].slice(-50).reverse()
  }, [events])

  return (
    <div style={{
      height: '100%',
      overflowY: 'auto',
      padding: '12px',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px'
    }}>
      {sorted.map((event, idx) => {
        const type = getEventType(event)
        const color = getColor(type)

        return (
          <div
            key={idx}
            style={{
              padding: '10px 12px',
              borderRadius: '10px',
              background: 'rgba(12, 18, 28, 0.85)',
              borderLeft: `4px solid ${color}`,
              boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
              fontSize: '12px',
              lineHeight: '1.4',
            }}
          >
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginBottom: '4px',
              color: '#9fb3c8',
              fontSize: '10px'
            }}>
              <span style={{ color }}>{type.toUpperCase()}</span>
              <span>{formatTime(event.payload?.timestamp)}</span>
            </div>

            <div style={{ color: '#e6f0ff' }}>
              {formatMessage(event)}
            </div>
          </div>
        )
      })}
    </div>
  )
}