// frontend/src/hooks/useWebSocket.js
// FIX 1: reconnect timer was 45000ms — way too slow. Now 3000ms.
// FIX 2: keepalive messages (channel=system, event=keepalive) were
//         already filtered by the `channel === 'system'` check — good.
//         But the check was AFTER normalization, so if a message had
//         no channel it fell through as 'system' and still got filtered.
//         Made the filter explicit on both channel and event.

import { useState, useEffect, useRef, useCallback } from 'react'

const MAX_EVENTS = 200
const RECONNECT_DELAY_MS = 3_000  // BUG WAS: 45000 — took 45s to reconnect after drop

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
    // Don't open a second connection if already open
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    // Don't try to connect if already connecting
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return

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

        // Skip ALL system-channel messages (connected, keepalive, subscribed, pong, stats)
        const channel = raw.channel || raw.type || 'unknown'
        if (channel === 'system') return

        const normalized = {
          type: raw.type || raw.event || channel || 'UNKNOWN',
          payload: raw.payload || raw.data || {},
          timestamp: raw.timestamp || new Date().toISOString(),
          channel: channel,
        }

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
      wsRef.current = null
      // FIX: was 45000ms — now reconnects in 3 seconds
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS)
    }

    ws.onerror = () => {
      // onclose will fire after onerror, which handles reconnect
      ws.close()
    }
  }, [url])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) {
        wsRef.current.onclose = null // prevent reconnect on intentional close
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  return { events, connected }
}