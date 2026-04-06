import { useEffect } from 'react'

const SOURCE_ID = 'rivers'
const LAYER_ID  = 'river-lines'

/**
 * RiverLayer
 * Renders river discharge lines on the MapLibre map.
 *
 * Props:
 *   map     - maplibregl.Map instance
 *   geojson - GeoJSON FeatureCollection; each feature needs properties.discharge (m³/s)
 *   visible - boolean, controlled by LayerControls (activeLayers.river)
 *
 * Color scale (discharge m³/s):
 *   0    → blue   (normal flow)
 *   500  → orange (elevated)
 *   1000 → red    (danger / flood stage)
 */
export default function RiverLayer({ map, geojson, visible }) {

  useEffect(() => {
    if (!map) return

    const onLoad = () => {
      if (!map.getSource(SOURCE_ID)) {
        map.addSource(SOURCE_ID, {
          type: 'geojson',
          data: geojson || { type: 'FeatureCollection', features: [] },
        })
      }

      if (!map.getLayer(LAYER_ID)) {
        map.addLayer({
          id: LAYER_ID,
          type: 'line',
          source: SOURCE_ID,
          layout: {
            'line-join': 'round',
            'line-cap':  'round',
          },
          paint: {
            'line-color': [
              'interpolate', ['linear'], ['get', 'discharge'],
              0,    '#2288cc',
              300,  '#00aaff',
              600,  '#ffaa00',
              1000, '#ff3300',
            ],
            'line-width': [
              'interpolate', ['linear'], ['get', 'discharge'],
              0,    1.5,
              500,  3,
              1000, 5,
            ],
            'line-opacity': 0.85,
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
      if (map.getLayer(LAYER_ID))  map.removeLayer(LAYER_ID)
      if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID)
    }
  }, [map])

  useEffect(() => {
    if (!map || !map.getSource(SOURCE_ID)) return
    map.getSource(SOURCE_ID).setData(
      geojson || { type: 'FeatureCollection', features: [] }
    )
  }, [map, geojson])

  useEffect(() => {
    if (!map || !map.isStyleLoaded()) return
    if (map.getLayer(LAYER_ID)) {
      map.setLayoutProperty(LAYER_ID, 'visibility', visible ? 'visible' : 'none')
    }
  }, [map, visible])

  return null
}
