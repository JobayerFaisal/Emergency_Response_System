// frontend/src/components/map/FloodLayer.jsx

import { useEffect } from 'react'

const SOURCE_ID = 'flood-zones'
const FILL_LAYER = 'flood-fill'
const OUTLINE_LAYER = 'flood-outline'

/**
 * FloodLayer
 * Adds / updates the flood risk polygon fill layer on the MapLibre map.
 *
 * Props:
 *   map      - maplibregl.Map instance
 *   geojson  - GeoJSON FeatureCollection; each feature needs properties.risk (0.0–1.0)
 *   visible  - boolean, controlled by LayerControls (activeLayers.flood)
 *
 * Color scale:
 *   0.0  → semi-transparent blue   (low risk)
 *   0.5  → mid blue                (moderate)
 *   1.0  → red-orange              (critical)
 */
export default function FloodLayer({ map, geojson, visible }) {

  // Add source + layers once on mount
  useEffect(() => {
    if (!map) return

    const onLoad = () => {
      if (!map.getSource(SOURCE_ID)) {
        map.addSource(SOURCE_ID, {
          type: 'geojson',
          data: geojson || { type: 'FeatureCollection', features: [] },
        })
      }

      if (!map.getLayer(FILL_LAYER)) {
        map.addLayer({
          id: FILL_LAYER,
          type: 'fill',
          source: SOURCE_ID,
          paint: {
            'fill-color': [
              'interpolate', ['linear'], ['get', 'risk'],
              0,   '#0033cc',
              0.3, '#0077ff',
              0.6, '#ff8800',
              1,   '#ff2200',
            ],
            'fill-opacity': [
              'interpolate', ['linear'], ['get', 'risk'],
              0, 0.25,
              1, 0.70,
            ],
          },
        })
      }

      if (!map.getLayer(OUTLINE_LAYER)) {
        map.addLayer({
          id: OUTLINE_LAYER,
          type: 'line',
          source: SOURCE_ID,
          paint: {
            'line-color': '#00aaff',
            'line-width': 1.5,
            'line-opacity': 0.8,
          },
        })
      }
    }

    if (map.isStyleLoaded()) {
      onLoad()
    } else {
      map.once('load', onLoad)
    }

    return () => {
      if (map.getLayer(OUTLINE_LAYER)) map.removeLayer(OUTLINE_LAYER)
      if (map.getLayer(FILL_LAYER))    map.removeLayer(FILL_LAYER)
      if (map.getSource(SOURCE_ID))    map.removeSource(SOURCE_ID)
    }
  }, [map])

  // Update GeoJSON data whenever it changes
  useEffect(() => {
    if (!map || !map.getSource(SOURCE_ID)) return
    map.getSource(SOURCE_ID).setData(
      geojson || { type: 'FeatureCollection', features: [] }
    )
  }, [map, geojson])

  // Toggle visibility
  useEffect(() => {
    if (!map || !map.isStyleLoaded()) return
    const vis = visible ? 'visible' : 'none'
    if (map.getLayer(FILL_LAYER))    map.setLayoutProperty(FILL_LAYER,    'visibility', vis)
    if (map.getLayer(OUTLINE_LAYER)) map.setLayoutProperty(OUTLINE_LAYER, 'visibility', vis)
  }, [map, visible])

  return null
}
