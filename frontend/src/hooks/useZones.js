import { useState, useEffect, useCallback, useRef } from 'react'

const ZONES_URL = '/api/zones'
const POLL_INTERVAL_MS = 15_000
const EMPTY_FC = { type: 'FeatureCollection', features: [] }

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
        const geom = f.geometry || {}

        // BUG WAS: code looked for p.center?.coordinates / p.center_coordinates
        // but the backend puts coordinates in geometry, not properties.
        // The backend already builds Polygon geometry via _make_circle(),
        // so we use the geometry directly. If it's a Point, build a circle.
        let geometry = null

        if (geom.type === 'Polygon' && geom.coordinates?.length) {
          // Already a polygon — use it as-is
          geometry = geom
        } else if (geom.type === 'Point' && geom.coordinates?.length === 2) {
          // Fallback: build an approximate circle from a Point
          const [lon, lat] = geom.coordinates
          const radiusKm = typeof p.radius_km === 'number' ? p.radius_km : 5
          geometry = {
            type: 'Polygon',
            coordinates: [buildCircle([lon, lat], radiusKm)],
          }
        } else {
          return null
        }

        let risk =
          typeof p.risk === 'number'
            ? p.risk
            : typeof p.risk_score === 'number'
              ? p.risk_score
              : 0

        if (!Number.isFinite(risk)) risk = 0
        risk = Math.max(0, Math.min(1, risk))

        const animatedRisk = applyBreathingRisk(risk, phase)

        return {
          type: 'Feature',
          geometry,
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

function buildCircle([lon, lat], radiusKm = 5, steps = 32) {
  const coords = []
  const R = 6371
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * (2 * Math.PI)
    const dx = radiusKm * Math.cos(angle)
    const dy = radiusKm * Math.sin(angle)
    const newLat = lat + (dy / R) * (180 / Math.PI)
    const newLon = lon + (dx / (R * Math.cos((lat * Math.PI) / 180))) * (180 / Math.PI)
    coords.push([newLon, newLat])
  }
  return coords
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

// Build synthetic river LineString features from flood zone polygons.
// Each zone gets a river line through its center, sized by risk score.
// This is used when the backend returns an empty rivers FeatureCollection.
function buildSyntheticRivers(floodFc) {
  if (!floodFc?.features?.length) return EMPTY_FC

  const features = floodFc.features
    .map((f, i) => {
      const geom = f.geometry
      const p = f.properties || {}

      // Compute centroid of polygon ring
      if (geom?.type !== 'Polygon' || !geom.coordinates?.[0]?.length) return null
      const ring = geom.coordinates[0]
      const sumLon = ring.reduce((s, c) => s + c[0], 0) / ring.length
      const sumLat = ring.reduce((s, c) => s + c[1], 0) / ring.length

      const risk = typeof p.base_risk === 'number' ? p.base_risk : (p.risk_score || 0)
      const discharge = Math.round(risk * 1200)

      // Draw a short river segment through the centroid
      return {
        type: 'Feature',
        geometry: {
          type: 'LineString',
          coordinates: [
            [sumLon - 0.10, sumLat - 0.03],
            [sumLon - 0.04, sumLat - 0.01],
            [sumLon, sumLat],
            [sumLon + 0.04, sumLat + 0.01],
            [sumLon + 0.10, sumLat + 0.03],
          ],
        },
        properties: {
          id: `river-${i}`,
          discharge,
          river_name: p.zone_name || p.name || 'River',
          zone_name: p.zone_name || p.name || 'Unknown',
        },
      }
    })
    .filter(Boolean)

  return { type: 'FeatureCollection', features }
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
      const floodRaw = data?.flood || EMPTY_FC

      // Normalize flood first so we have polygons with base_risk set
      const normalizedFlood = normalizeFlood(floodRaw, phaseRef.current)
      rawFloodRef.current = normalizedFlood

      // Use backend rivers if available, otherwise synthesize from flood zones
      const backendRivers = data?.rivers
      const hasRealRivers = backendRivers?.features?.length > 0
      const rivers = hasRealRivers
        ? normalizeRivers(backendRivers)
        : buildSyntheticRivers(normalizedFlood)

      setZones({ flood: normalizedFlood, rivers })
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
      // Only animate flood — rivers don't need breathing
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