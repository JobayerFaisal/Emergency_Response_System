import { useState, useEffect } from 'react'

const ZONES_URL = '/api/zones'
const POLL_INTERVAL_MS = 15_000

/**
 * useZones
 * Fetches flood zone GeoJSON from the dashboard API on mount
 * and refreshes every POLL_INTERVAL_MS milliseconds.
 *
 * Returns:
 * {
 *   zones: {
 *     flood: GeoJSON FeatureCollection   // risk polygons (feature.properties.risk: 0–1)
 *     rivers: GeoJSON FeatureCollection  // river lines (feature.properties.discharge: m³/s)
 *   },
 *   loading: boolean,
 *   error: string | null
 * }
 */
export default function useZones() {
  const [zones, setZones] = useState({ flood: null, rivers: null })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchZones = async () => {
    try {
      const res = await fetch(ZONES_URL)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setZones({
        flood: data.flood ?? { type: 'FeatureCollection', features: [] },
        rivers: data.rivers ?? { type: 'FeatureCollection', features: [] },
      })
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchZones()
    const interval = setInterval(fetchZones, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [])

  return { zones, loading, error }
}
