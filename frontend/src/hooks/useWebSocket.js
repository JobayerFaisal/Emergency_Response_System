import { useState, useEffect, useRef, useCallback } from 'react'

const MAX_EVENTS = 200

/**
 * useWebSocket
 * Connects to FastAPI WebSocket endpoint, parses JSON events,
 * and maintains a rolling buffer of the last MAX_EVENTS events.
 *
 * Expected message format from server:
 * { type: string, payload: object, timestamp: string }
 */
export default function useWebSocket(url) {
  const [events, setEvents] = useState([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
    }

    ws.onmessage = (e) => {
      try {
        const raw = JSON.parse(e.data)

        // 🔥 Normalize backend → frontend format
        const normalized = {
          type: raw.type || raw.event || raw.channel || 'UNKNOWN',
          payload: raw.payload || raw.data || {},
          timestamp: raw.timestamp || new Date().toISOString(),
          channel: raw.channel || 'system'
        }

        setEvents(prev => {
          const next = [...prev, normalized]
          if (normalized.channel === 'system') return prev
          return next.length > MAX_EVENTS ? next.slice(-MAX_EVENTS) : next
        })

      } catch {
        // ignore malformed messages
      }
    }

    ws.onclose = () => {
      setConnected(false)
      // Reconnect after 3 seconds
      reconnectTimer.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [url])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { events, connected }
}
