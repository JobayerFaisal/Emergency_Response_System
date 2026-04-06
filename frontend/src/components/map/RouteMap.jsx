import { useEffect, useRef } from 'react'

const SOURCE_ID    = 'osrm-routes'
const LAYER_BG     = 'route-bg'
const LAYER_LINE   = 'route-line'
const LAYER_DASH   = 'route-dash'

/**
 * RouteMap
 * Renders OSRM dispatch routes on the MapLibre map with an animated dashed overlay.
 *
 * Props:
 *   map      - maplibregl.Map instance
 *   routes   - array from Agent4 data:
 *              [{
 *                mission_id, volunteer_id, status,
 *                geometry: GeoJSON LineString,   ← from OSRM
 *                distance_km, eta_minutes
 *              }]
 *   visible  - boolean, controlled by LayerControls (activeLayers.routes)
 *
 * Route colors by mission status:
 *   ASSIGNED / EN_ROUTE → cyan animated dash
 *   ACTIVE              → orange
 *   COMPLETED           → green (static)
 *   FAILED              → red   (static)
 */

const STATUS_COLOR = {
  ASSIGNED:  '#00c8ff',
  EN_ROUTE:  '#00c8ff',
  ACTIVE:    '#ffaa00',
  COMPLETED: '#00e676',
  FAILED:    '#ff3d00',
}

function buildFeatureCollection(routes) {
  return {
    type: 'FeatureCollection',
    features: routes
      .filter(r => r.geometry)
      .map(r => ({
        type: 'Feature',
        geometry: r.geometry,
        properties: {
          mission_id:   r.mission_id,
          volunteer_id: r.volunteer_id,
          status:       r.status || 'ASSIGNED',
          color:        STATUS_COLOR[r.status] || '#00c8ff',
          eta:          r.eta_minutes,
          distance:     r.distance_km,
        },
      })),
  }
}

export default function RouteMap({ map, routes = [], visible }) {
  const animFrameRef = useRef(null)
  const dashOffsetRef = useRef(0)

  // Add source + layers once
  useEffect(() => {
    if (!map) return

    const onLoad = () => {
      if (!map.getSource(SOURCE_ID)) {
        map.addSource(SOURCE_ID, {
          type: 'geojson',
          data: buildFeatureCollection(routes),
        })
      }

      // Background glow line
      if (!map.getLayer(LAYER_BG)) {
        map.addLayer({
          id: LAYER_BG,
          type: 'line',
          source: SOURCE_ID,
          layout: { 'line-join': 'round', 'line-cap': 'round' },
          paint: {
            'line-color': ['get', 'color'],
            'line-width': 8,
            'line-opacity': 0.18,
            'line-blur': 4,
          },
        })
      }

      // Solid base route line
      if (!map.getLayer(LAYER_LINE)) {
        map.addLayer({
          id: LAYER_LINE,
          type: 'line',
          source: SOURCE_ID,
          layout: { 'line-join': 'round', 'line-cap': 'round' },
          paint: {
            'line-color': ['get', 'color'],
            'line-width': 2.5,
            'line-opacity': 0.75,
          },
        })
      }

      // Animated dashed overlay (for active routes)
      if (!map.getLayer(LAYER_DASH)) {
        map.addLayer({
          id: LAYER_DASH,
          type: 'line',
          source: SOURCE_ID,
          filter: ['in', ['get', 'status'], ['literal', ['ASSIGNED', 'EN_ROUTE', 'ACTIVE']]],
          layout: { 'line-join': 'round', 'line-cap': 'round' },
          paint: {
            'line-color': ['get', 'color'],
            'line-width': 3,
            'line-dasharray': [0, 4, 3],
            'line-opacity': 0.9,
          },
        })
      }

      // Animate the dash
      const animateDash = () => {
        dashOffsetRef.current = (dashOffsetRef.current - 0.5) % 10
        if (map.getLayer(LAYER_DASH)) {
          map.setPaintProperty(LAYER_DASH, 'line-dasharray', [
            dashOffsetRef.current, 4, 3,
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
      cancelAnimationFrame(animFrameRef.current)
      if (map.getLayer(LAYER_DASH))  map.removeLayer(LAYER_DASH)
      if (map.getLayer(LAYER_LINE))  map.removeLayer(LAYER_LINE)
      if (map.getLayer(LAYER_BG))    map.removeLayer(LAYER_BG)
      if (map.getSource(SOURCE_ID))  map.removeSource(SOURCE_ID)
    }
  }, [map])

  // Update route data
  useEffect(() => {
    if (!map || !map.getSource(SOURCE_ID)) return
    map.getSource(SOURCE_ID).setData(buildFeatureCollection(routes))
  }, [map, routes])

  // Toggle visibility
  useEffect(() => {
    if (!map || !map.isStyleLoaded()) return
    const vis = visible ? 'visible' : 'none'
    ;[LAYER_BG, LAYER_LINE, LAYER_DASH].forEach(id => {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', vis)
    })
  }, [map, visible])

  return null
}
