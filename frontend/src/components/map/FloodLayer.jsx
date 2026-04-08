// frontend/src/components/map/FloodLayer.jsx

import { useEffect, useRef } from 'react'
import maplibregl from 'maplibre-gl'

const SOURCE_ID = 'flood-zones'
const FILL_LAYER = 'flood-fill'
const GLOW_LAYER = 'flood-glow'
const OUTLINE_LAYER = 'flood-outline'
const HEAT_LAYER = 'flood-heat'
const HEAT_POINTS_SOURCE_ID = 'flood-heat-points'
const EMPTY_FC = { type: 'FeatureCollection', features: [] }

function hasStyle(map) {
  return !!map && !!map.style
}

function safeGetLayer(map, id) {
  if (!hasStyle(map)) return null
  try {
    return map.getLayer(id)
  } catch {
    return null
  }
}

function safeGetSource(map, id) {
  if (!hasStyle(map)) return null
  try {
    return map.getSource(id)
  } catch {
    return null
  }
}

function toHeatPoints(fc) {
  if (!fc?.features?.length) return EMPTY_FC

  return {
    type: 'FeatureCollection',
    features: fc.features
      .map((feature, idx) => {
        const props = feature.properties || {}
        const geom = feature.geometry || {}
        const coords = geom.coordinates
        let center = null

        if (geom.type === 'Polygon' && Array.isArray(coords?.[0]) && coords[0].length) {
          const ring = coords[0]
          const sum = ring.reduce(
            (acc, [lng, lat]) => ({ lng: acc.lng + lng, lat: acc.lat + lat }),
            { lng: 0, lat: 0 }
          )
          center = [sum.lng / ring.length, sum.lat / ring.length]
        }

        if (!center) return null

        return {
          type: 'Feature',
          geometry: { type: 'Point', coordinates: center },
          properties: {
            id: props.id || `heat-${idx}`,
            risk: Number(props.risk || 0),
            zone_name: props.zone_name || props.name || 'Flood zone',
          },
        }
      })
      .filter(Boolean),
  }
}

export default function FloodLayer({ map, geojson, visible }) {
  const pulseRef = useRef(0)
  const animRef = useRef(null)

  useEffect(() => {
    if (!map) return

    const onLoad = () => {
      if (!safeGetSource(map, SOURCE_ID)) {
        map.addSource(SOURCE_ID, {
          type: 'geojson',
          data: geojson || EMPTY_FC,
        })
      }

      if (!safeGetSource(map, HEAT_POINTS_SOURCE_ID)) {
        map.addSource(HEAT_POINTS_SOURCE_ID, {
          type: 'geojson',
          data: toHeatPoints(geojson || EMPTY_FC),
        })
      }

      if (!safeGetLayer(map, HEAT_LAYER)) {
        map.addLayer({
          id: HEAT_LAYER,
          type: 'heatmap',
          source: HEAT_POINTS_SOURCE_ID,
          paint: {
            'heatmap-weight': [
              'interpolate',
              ['linear'],
              ['coalesce', ['to-number', ['get', 'risk']], 0],
              0, 0,
              0.2, 0.35,
              0.5, 0.7,
              1, 1.2,
            ],
            'heatmap-intensity': 1.35,
            'heatmap-radius': 42,
            'heatmap-opacity': 0.62,
            'heatmap-color': [
              'interpolate',
              ['linear'],
              ['heatmap-density'],
              0.00, 'rgba(0,0,255,0)',
              0.20, '#4dabf7',
              0.40, '#339af0',
              0.60, '#1c7ed6',
              0.80, '#1864ab',
              1.00, '#0b3d91',
            ],
          },
        })
      }

      if (!safeGetLayer(map, FILL_LAYER)) {
        map.addLayer({
          id: FILL_LAYER,
          type: 'fill',
          source: SOURCE_ID,
          paint: {
            'fill-color': [
              'interpolate',
              ['linear'],
              ['coalesce', ['to-number', ['*', ['get', 'risk'], 2.5]], 0],
              0.00, 'rgba(0,0,0,0)',
              0.15, '#0f3b82',
              0.35, '#155eef',
              0.55, '#1d9bf0',
              0.75, '#60a5fa',
              1.00, '#bae6fd',
            ],
            'fill-opacity': [
              'interpolate',
              ['linear'],
              ['coalesce', ['to-number', ['*', ['get', 'risk'], 2.5]], 0],
              0.00, 0.00,
              0.20, 0.08,
              0.40, 0.13,
              0.60, 0.18,
              0.80, 0.24,
              1.00, 0.30,
            ],
          },
        })
      }

      if (!safeGetLayer(map, GLOW_LAYER)) {
        map.addLayer({
          id: GLOW_LAYER,
          type: 'fill',
          source: SOURCE_ID,
          paint: {
            'fill-color': '#7dd3fc',
            'fill-opacity': [
              'interpolate',
              ['linear'],
              ['coalesce', ['to-number', ['*', ['get', 'risk'], 2.5]], 0],
              0.00, 0.00,
              0.60, 0.00,
              0.80, 0.08,
              1.00, 0.16,
            ],
          },
        })
      }

      if (!safeGetLayer(map, OUTLINE_LAYER)) {
        map.addLayer({
          id: OUTLINE_LAYER,
          type: 'line',
          source: SOURCE_ID,
          paint: {
            'line-color': '#93c5fd',
            'line-width': 1.0,
            'line-opacity': 0.24,
          },
        })
      }

      const animateFlood = () => {
        if (!hasStyle(map)) return

        pulseRef.current += 0.015
        const t = (Math.sin(pulseRef.current) + 1) / 2

        try {
          if (safeGetLayer(map, HEAT_LAYER)) {
            map.setPaintProperty(HEAT_LAYER, 'heatmap-intensity', 1.2 + t * 0.45)
            map.setPaintProperty(HEAT_LAYER, 'heatmap-opacity', 0.52 + t * 0.14)
          }

          if (safeGetLayer(map, GLOW_LAYER)) {
            map.setPaintProperty(GLOW_LAYER, 'fill-opacity', [
              'interpolate',
              ['linear'],
              ['coalesce', ['to-number', ['*', ['get', 'risk'], 2.5]], 0],
              0.00, 0.00,
              0.60, 0.00,
              0.80, 0.08 + t * 0.03,
              1.00, 0.14 + t * 0.05,
            ])
          }
        } catch {
          return
        }

        animRef.current = requestAnimationFrame(animateFlood)
      }

      animateFlood()
    }

    if (hasStyle(map) && map.isStyleLoaded()) {
      onLoad()
    } else {
      map.once('load', onLoad)
    }

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current)
      if (!hasStyle(map)) return

      try {
        if (safeGetLayer(map, OUTLINE_LAYER)) map.removeLayer(OUTLINE_LAYER)
        if (safeGetLayer(map, GLOW_LAYER)) map.removeLayer(GLOW_LAYER)
        if (safeGetLayer(map, FILL_LAYER)) map.removeLayer(FILL_LAYER)
        if (safeGetLayer(map, HEAT_LAYER)) map.removeLayer(HEAT_LAYER)
        if (safeGetSource(map, HEAT_POINTS_SOURCE_ID)) map.removeSource(HEAT_POINTS_SOURCE_ID)
        if (safeGetSource(map, SOURCE_ID)) map.removeSource(SOURCE_ID)
      } catch {
        // ignore during teardown
      }
    }
  }, [map])

  useEffect(() => {
    const floodSource = safeGetSource(map, SOURCE_ID)
    if (floodSource) {
      try {
        floodSource.setData(geojson || EMPTY_FC)
      } catch {}
    }

    const heatSource = safeGetSource(map, HEAT_POINTS_SOURCE_ID)
    if (heatSource) {
      try {
        heatSource.setData(toHeatPoints(geojson || EMPTY_FC))
      } catch {}
    }
  }, [map, geojson])

  useEffect(() => {
    if (!hasStyle(map)) return
    const vis = visible ? 'visible' : 'none'

    ;[HEAT_LAYER, FILL_LAYER, GLOW_LAYER, OUTLINE_LAYER].forEach((id) => {
      if (safeGetLayer(map, id)) {
        try {
          map.setLayoutProperty(id, 'visibility', vis)
        } catch {}
      }
    })
  }, [map, visible])

  return null
}