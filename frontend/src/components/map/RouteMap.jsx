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
  const dashPhaseRef = useRef(0)

  const featureCollection = useMemo(
    () => buildFeatureCollection(routes),
    [routes]
  )

  useEffect(() => {
    if (!map) return

    const onLoad = () => {
      if (!map.getSource(SOURCE_ID)) {
        map.addSource(SOURCE_ID, {
          type: 'geojson',
          data: featureCollection,
        })
      }

      if (!map.getLayer(LAYER_BG)) {
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

      if (!map.getLayer(LAYER_LINE)) {
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

      if (!map.getLayer(LAYER_DASH)) {
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

      if (!map.getLayer(LAYER_ARROW)) {
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

      const animateDash = () => {
        dashPhaseRef.current += 0.08
        const phase = dashPhaseRef.current % 4

        if (map.getLayer(LAYER_DASH)) {
          map.setPaintProperty(LAYER_DASH, 'line-dasharray', [
            phase,
            2,
            3,
          ])
        }

        animFrameRef.current = requestAnimationFrame(animateDash)
      }

      animateDash()
    }

    if (map.isStyleLoaded()) {
      onLoad()
    } else {
      map.once('load', onLoad)
    }

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
      if (map.getLayer(LAYER_ARROW)) map.removeLayer(LAYER_ARROW)
      if (map.getLayer(LAYER_DASH)) map.removeLayer(LAYER_DASH)
      if (map.getLayer(LAYER_LINE)) map.removeLayer(LAYER_LINE)
      if (map.getLayer(LAYER_BG)) map.removeLayer(LAYER_BG)
      if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID)
    }
  }, [map])

  useEffect(() => {
    if (!map || !map.getSource(SOURCE_ID)) return
    map.getSource(SOURCE_ID).setData(featureCollection)
  }, [map, featureCollection])

  useEffect(() => {
    if (!map || !map.isStyleLoaded()) return
    const vis = visible ? 'visible' : 'none'
    ;[LAYER_BG, LAYER_LINE, LAYER_DASH, LAYER_ARROW].forEach((id) => {
      if (map.getLayer(id)) {
        map.setLayoutProperty(id, 'visibility', vis)
      }
    })
  }, [map, visible])

  return null
}