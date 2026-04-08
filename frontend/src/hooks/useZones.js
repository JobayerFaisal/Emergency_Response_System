import { useState, useEffect, useCallback, useRef } from 'react'

const ZONES_URL = '/api/zones'
const POLL_INTERVAL_MS = 15_000
const EMPTY_FC = { type: 'FeatureCollection', features: [] }

function createCircle([lon, lat], radiusKm = 5, steps = 32) {
  const coords = []
  const R = 6371

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

function applyBreathingRisk(risk, phase) {
  const delta = Math.sin(phase) * 0.06
  const next = risk + delta
  return Math.max(0, Math.min(1, next))
}

function normalizeFlood(fc, phase = 0) {
  if (!fc?.features) return EMPTY_FC

  return {
    type: 'FeatureCollection',
    features: fc.features
      .map((f) => {
        const p = f.properties || {}

        let risk =
          typeof p.risk === 'number'
            ? p.risk
            : typeof p.risk_score === 'number'
              ? p.risk_score
              : 0

        if (!Number.isFinite(risk)) risk = 0
        if (risk < 0) risk = 0
        if (risk > 1) risk = 1

        const animatedRisk = applyBreathingRisk(risk, phase)

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
            risk: animatedRisk,
            base_risk: risk,
            zone_name: p.zone_name || p.name || 'Unknown zone',
          },
        }
      })
      .filter(Boolean),
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

export default function useZones() {
  const [zones, setZones] = useState({
    flood: EMPTY_FC,
    rivers: EMPTY_FC,
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const rawFloodRef = useRef(EMPTY_FC)
  const phaseRef = useRef(0)

  const fetchZones = useCallback(async () => {
    try {
      const res = await fetch(ZONES_URL, {
        headers: { Accept: 'application/json' },
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const data = await res.json()
      rawFloodRef.current = data?.flood || EMPTY_FC

      setZones({
        flood: normalizeFlood(rawFloodRef.current, phaseRef.current),
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

    const poll = setInterval(() => {
      fetchZones()
    }, POLL_INTERVAL_MS)

    const animate = setInterval(() => {
      phaseRef.current += 0.25
      setZones((prev) => ({
        ...prev,
        flood: normalizeFlood(rawFloodRef.current, phaseRef.current),
      }))
    }, 1200)

    return () => {
      clearInterval(poll)
      clearInterval(animate)
    }
  }, [fetchZones])

  return { zones, loading, error, reload: fetchZones }
}