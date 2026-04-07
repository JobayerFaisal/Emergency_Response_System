import { useState, useEffect, useCallback } from 'react'

const ZONES_URL = '/api/zones'
const POLL_INTERVAL_MS = 15_000
const EMPTY_FC = { type: 'FeatureCollection', features: [] }

function createCircle([lon, lat], radiusKm = 5, steps = 32) {
  const coords = []
  const R = 6371 // Earth radius km

  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * (2 * Math.PI)

    const dx = radiusKm * Math.cos(angle)
    const dy = radiusKm * Math.sin(angle)

    const newLat = lat + (dy / R) * (180 / Math.PI)
    const newLon =
      lon +
      (dx / (R * Math.cos((lat * Math.PI) / 180))) * (180 / Math.PI)

    coords.push([newLon, newLat])
  }

  return coords
}

function normalizeFlood(fc) {
  if (!fc?.features) return { type: 'FeatureCollection', features: [] }

  return {
    type: 'FeatureCollection',
    features: fc.features.map((f) => {
      const p = f.properties || {}

      let risk =
        typeof p.risk === 'number'
          ? p.risk
          : typeof p.risk_score === 'number'
            ? p.risk_score
            : 0

      if (risk < 0) risk = 0
      if (risk > 1) risk = 1

      // 🔥 Create smooth circular flood area
      const center =
        p.center?.coordinates ||
        p.center_coordinates ||
        p.centroid_coordinates

      if (!center) return null

      return {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [createCircle(center, 5)],
        },
        properties: {
          ...p,
          risk,
          zone_name: p.zone_name || p.name,
        },
      }
    }).filter(Boolean),
  }
}

function normalizeRivers(fc) {
  if (!fc || !Array.isArray(fc.features)) return EMPTY_FC

  return {
    type: 'FeatureCollection',
    features: fc.features.map((feature) => {
      const props = feature?.properties || {}

      let discharge =
        typeof props.discharge === 'number'
          ? props.discharge
          : typeof props.flow_rate === 'number'
            ? props.flow_rate
            : 0

      if (!Number.isFinite(discharge)) discharge = 0
      if (discharge < 0) discharge = 0

      return {
        ...feature,
        properties: {
          ...props,
          discharge,
          river_name: props.river_name || props.name || 'River segment',
        },
      }
    }),
  }
}

/**
 * useZones
 * Fetches flood zone GeoJSON from the dashboard API on mount
 * and refreshes every POLL_INTERVAL_MS milliseconds.
 *
 * Returns:
 * {
 *   zones: {
 *     flood: GeoJSON FeatureCollection
 *     rivers: GeoJSON FeatureCollection
 *   },
 *   loading: boolean,
 *   error: string | null,
 *   reload: () => Promise<void>
 * }
 */
export default function useZones() {
  const [zones, setZones] = useState({
    flood: EMPTY_FC,
    rivers: EMPTY_FC,
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchZones = useCallback(async () => {
    try {
      const res = await fetch(ZONES_URL, {
        headers: { Accept: 'application/json' },
      })

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }

      const data = await res.json()

      setZones({
        flood: normalizeFlood(data?.flood),
        rivers: normalizeRivers(data?.rivers),
      })

      setError(null)
    } catch (err) {
      setError(err?.message || 'Failed to fetch zones')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchZones()

    const interval = setInterval(() => {
      fetchZones()
    }, POLL_INTERVAL_MS)

    return () => clearInterval(interval)
  }, [fetchZones])

  return { zones, loading, error, reload: fetchZones }
}
