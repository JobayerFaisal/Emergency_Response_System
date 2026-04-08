// frontend/src/hooks/useWebSocket.js

import { useState, useEffect, useRef, useCallback } from 'react'

const MAX_EVENTS = 200

export default function useWebSocket(url, { paused = false, resetToken = 0 } = {}) {
  const [events, setEvents] = useState([])
  const [connected, setConnected] = useState(false)

  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)
  const pausedRef = useRef(paused)

  useEffect(() => {
    pausedRef.current = paused
  }, [paused])

  useEffect(() => {
    setEvents([])
  }, [resetToken])

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
      if (pausedRef.current) return

      try {
        const raw = JSON.parse(e.data)

        const normalized = {
          type: raw.type || raw.event || raw.channel || 'UNKNOWN',
          payload: raw.payload || raw.data || {},
          timestamp: raw.timestamp || new Date().toISOString(),
          channel: raw.channel || 'system',
        }

        if (normalized.channel === 'system') return

        setEvents((prev) => {
          const next = [...prev, normalized]
          return next.length > MAX_EVENTS ? next.slice(-MAX_EVENTS) : next
        })
      } catch {
        // ignore malformed messages
      }
    }

    ws.onclose = () => {
      setConnected(false)
      reconnectTimer.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [url])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { events, connected }
}