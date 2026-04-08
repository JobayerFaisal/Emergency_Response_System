import { useEffect } from 'react'

const SOURCE_ID = 'rivers'
const LAYER_ID = 'river-lines'
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

export default function RiverLayer({ map, geojson, visible }) {
  useEffect(() => {
    if (!map) return

    const onLoad = () => {
      if (!safeGetSource(map, SOURCE_ID)) {
        map.addSource(SOURCE_ID, {
          type: 'geojson',
          data: geojson || EMPTY_FC,
        })
      }

      if (!safeGetLayer(map, LAYER_ID)) {
        map.addLayer({
          id: LAYER_ID,
          type: 'line',
          source: SOURCE_ID,
          layout: {
            'line-join': 'round',
            'line-cap': 'round',
          },
          paint: {
            'line-color': [
              'interpolate', ['linear'], ['get', 'discharge'],
              0, '#2288cc',
              300, '#00aaff',
              600, '#ffaa00',
              1000, '#ff3300',
            ],
            'line-width': [
              'interpolate', ['linear'], ['get', 'discharge'],
              0, 1.5,
              500, 3,
              1000, 5,
            ],
            'line-opacity': 0.85,
          },
        })
      }
    }

    if (hasStyle(map) && map.isStyleLoaded()) {
      onLoad()
    } else {
      map.once('load', onLoad)
    }

    return () => {
      if (!hasStyle(map)) return
      try {
        if (safeGetLayer(map, LAYER_ID)) map.removeLayer(LAYER_ID)
        if (safeGetSource(map, SOURCE_ID)) map.removeSource(SOURCE_ID)
      } catch {
        // ignore during teardown
      }
    }
  }, [map])

  useEffect(() => {
    const source = safeGetSource(map, SOURCE_ID)
    if (!source) return
    try {
      source.setData(geojson || EMPTY_FC)
    } catch {
      // ignore during teardown
    }
  }, [map, geojson])

  useEffect(() => {
    if (!hasStyle(map)) return
    if (safeGetLayer(map, LAYER_ID)) {
      try {
        map.setLayoutProperty(LAYER_ID, 'visibility', visible ? 'visible' : 'none')
      } catch {
        // ignore during teardown
      }
    }
  }, [map, visible])

  return null
}