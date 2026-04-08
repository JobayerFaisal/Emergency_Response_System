import { useEffect, useMemo, useRef } from 'react'

const SOURCE_ID = 'osrm-routes'
const LAYER_BG = 'route-bg'
const LAYER_LINE = 'route-line'
const LAYER_DASH = 'route-dash'
const LAYER_ARROW = 'route-arrow'

const STATUS_COLOR = {
  ASSIGNED: '#38bdf8',
  EN_ROUTE: '#22d3ee',
  ACTIVE: '#f59e0b',
  COMPLETED: '#22c55e',
  FAILED: '#ef4444',
}

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

function buildFeatureCollection(routes) {
  return {
    type: 'FeatureCollection',
    features: routes
      .filter((r) => r.geometry?.coordinates?.length)
      .map((r) => ({
        type: 'Feature',
        geometry: r.geometry,
        properties: {
          mission_id: r.mission_id,
          volunteer_id: r.volunteer_id,
          status: r.status || 'ASSIGNED',
          color: STATUS_COLOR[r.status] || '#38bdf8',
          eta: r.eta_minutes,
          distance: r.distance_km,
        },
      })),
  }
}

export default function RouteMap({ map, routes = [], visible }) {
  const animFrameRef = useRef(null)

  const featureCollection = useMemo(
    () => buildFeatureCollection(routes),
    [routes]
  )

  useEffect(() => {
    if (!map) return

    const onLoad = () => {
      if (!safeGetSource(map, SOURCE_ID)) {
        map.addSource(SOURCE_ID, {
          type: 'geojson',
          data: featureCollection,
        })
      }

      if (!safeGetLayer(map, LAYER_BG)) {
        map.addLayer({
          id: LAYER_BG,
          type: 'line',
          source: SOURCE_ID,
          layout: {
            'line-join': 'round',
            'line-cap': 'round',
          },
          paint: {
            'line-color': ['get', 'color'],
            'line-width': 12,
            'line-opacity': 0.18,
            'line-blur': 6,
          },
        })
      }

      if (!safeGetLayer(map, LAYER_LINE)) {
        map.addLayer({
          id: LAYER_LINE,
          type: 'line',
          source: SOURCE_ID,
          layout: {
            'line-join': 'round',
            'line-cap': 'round',
          },
          paint: {
            'line-color': ['get', 'color'],
            'line-width': [
              'case',
              ['==', ['get', 'status'], 'ACTIVE'], 4.5,
              ['==', ['get', 'status'], 'EN_ROUTE'], 4,
              3.2,
            ],
            'line-opacity': [
              'case',
              ['==', ['get', 'status'], 'ACTIVE'], 1.0,
              ['==', ['get', 'status'], 'EN_ROUTE'], 0.92,
              ['==', ['get', 'status'], 'ASSIGNED'], 0.78,
              ['==', ['get', 'status'], 'COMPLETED'], 0.55,
              0.5,
            ],
          },
        })
      }

      if (!safeGetLayer(map, LAYER_DASH)) {
        map.addLayer({
          id: LAYER_DASH,
          type: 'line',
          source: SOURCE_ID,
          filter: ['in', ['get', 'status'], ['literal', ['ASSIGNED', 'EN_ROUTE', 'ACTIVE']]],
          layout: {
            'line-join': 'round',
            'line-cap': 'round',
          },
          paint: {
            'line-color': '#ffffff',
            'line-width': [
              'case',
              ['==', ['get', 'status'], 'ACTIVE'], 2.6,
              2.1,
            ],
            'line-opacity': [
              'case',
              ['==', ['get', 'status'], 'ACTIVE'], 0.98,
              0.88,
            ],
            'line-dasharray': [0, 2, 3],
          },
        })
      }

      if (!safeGetLayer(map, LAYER_ARROW)) {
        map.addLayer({
          id: LAYER_ARROW,
          type: 'symbol',
          source: SOURCE_ID,
          filter: ['in', ['get', 'status'], ['literal', ['ASSIGNED', 'EN_ROUTE', 'ACTIVE']]],
          layout: {
            'symbol-placement': 'line',
            'symbol-spacing': 60,
            'text-field': '▶',
            'text-size': 12,
            'text-keep-upright': false,
            'text-rotation-alignment': 'map',
            'text-font': ['Open Sans Regular'],
          },
          paint: {
            'text-color': '#dff6ff',
            'text-opacity': 0.88,
            'text-halo-color': 'rgba(0,0,0,0.35)',
            'text-halo-width': 1,
          },
        })
      }

      let dashPhase = 0
      const animateDash = () => {
        if (!hasStyle(map) || !safeGetLayer(map, LAYER_DASH)) return

        dashPhase += 0.08
        const phase = dashPhase % 4

        try {
          map.setPaintProperty(LAYER_DASH, 'line-dasharray', [phase, 2, 3])
        } catch {
          return
        }

        animFrameRef.current = requestAnimationFrame(animateDash)
      }

      animateDash()
    }

    if (hasStyle(map) && map.isStyleLoaded()) {
      onLoad()
    } else {
      map.once('load', onLoad)
    }

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
      if (!hasStyle(map)) return

      try {
        if (safeGetLayer(map, LAYER_ARROW)) map.removeLayer(LAYER_ARROW)
        if (safeGetLayer(map, LAYER_DASH)) map.removeLayer(LAYER_DASH)
        if (safeGetLayer(map, LAYER_LINE)) map.removeLayer(LAYER_LINE)
        if (safeGetLayer(map, LAYER_BG)) map.removeLayer(LAYER_BG)
        if (safeGetSource(map, SOURCE_ID)) map.removeSource(SOURCE_ID)
      } catch {
        // map/style already torn down
      }
    }
  }, [map])

  useEffect(() => {
    const source = safeGetSource(map, SOURCE_ID)
    if (!source) return
    try {
      source.setData(featureCollection)
    } catch {
      // ignore during style teardown
    }
  }, [map, featureCollection])

  useEffect(() => {
    if (!hasStyle(map)) return
    const vis = visible ? 'visible' : 'none'

    ;[LAYER_BG, LAYER_LINE, LAYER_DASH, LAYER_ARROW].forEach((id) => {
      if (safeGetLayer(map, id)) {
        try {
          map.setLayoutProperty(id, 'visibility', vis)
        } catch {
          // ignore during style transitions
        }
      }
    })
  }, [map, visible])

  return null
}