import { useEffect } from 'react'

const SOURCE_ID = 'flood-zones'
const FILL_LAYER = 'flood-fill'
const GLOW_LAYER = 'flood-glow'
const OUTLINE_LAYER = 'flood-outline'

const EMPTY_FC = { type: 'FeatureCollection', features: [] }

export default function FloodLayer({ map, geojson, visible }) {
  useEffect(() => {
    if (!map) return

    const onLoad = () => {
      if (!map.getSource(SOURCE_ID)) {
        map.addSource(SOURCE_ID, {
          type: 'geojson',
          data: geojson || EMPTY_FC,
        })
      }

      if (!map.getLayer(FILL_LAYER)) {
        map.addLayer({
          id: FILL_LAYER,
          type: 'fill',
          source: SOURCE_ID,
          paint: {
            'fill-color': [
              'interpolate', ['linear'],
              ['coalesce', ['to-number', ['get', 'risk']], 0],
              0.00, 'rgba(0,0,0,0)',
              0.15, '#0f3b82',
              0.35, '#155eef',
              0.55, '#1d9bf0',
              0.75, '#60a5fa',
              1.00, '#bae6fd',
            ],
            'fill-opacity': [
              'interpolate', ['linear'],
              ['coalesce', ['to-number', ['get', 'risk']], 0],
              0.00, 0.00,
              0.20, 0.10,
              0.40, 0.16,
              0.60, 0.22,
              0.80, 0.30,
              1.00, 0.40,
            ],
          },
        })
      }

      if (!map.getLayer(GLOW_LAYER)) {
        map.addLayer({
          id: GLOW_LAYER,
          type: 'fill',
          source: SOURCE_ID,
          paint: {
            'fill-color': '#7dd3fc',
            'fill-opacity': [
              'interpolate', ['linear'],
              ['coalesce', ['to-number', ['get', 'risk']], 0],
              0.00, 0.00,
              0.60, 0.00,
              0.80, 0.10,
              1.00, 0.18,
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
            'line-color': '#93c5fd',
            'line-width': 1.0,
            'line-opacity': 0.28,
          },
        })
      }

      map.on('click', FILL_LAYER, (e) => {
        const feature = e.features?.[0]
        if (!feature) return

        const p = feature.properties || {}
        const risk = Number(p.risk || 0)
        const pct = Math.round(risk * 100)

        new window.maplibregl.Popup({ offset: 12 })
          .setLngLat(e.lngLat)
          .setHTML(`
            <div style="min-width:180px">
              <strong>${p.zone_name || p.name || 'Flood zone'}</strong><br/>
              <span>Flood risk: ${pct}%</span><br/>
              ${p.risk_level ? `<span>Level: ${p.risk_level}</span><br/>` : ''}
              ${p.population_density ? `<span>Population density: ${p.population_density}</span>` : ''}
            </div>
          `)
          .addTo(map)
      })
    }

    if (map.isStyleLoaded()) {
      onLoad()
    } else {
      map.once('load', onLoad)
    }

    return () => {
      if (map.getLayer(OUTLINE_LAYER)) map.removeLayer(OUTLINE_LAYER)
      if (map.getLayer(GLOW_LAYER)) map.removeLayer(GLOW_LAYER)
      if (map.getLayer(FILL_LAYER)) map.removeLayer(FILL_LAYER)
      if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID)
    }
  }, [map])

  useEffect(() => {
    if (!map || !map.getSource(SOURCE_ID)) return
    map.getSource(SOURCE_ID).setData(geojson || EMPTY_FC)
  }, [map, geojson])

  useEffect(() => {
    if (!map || !map.isStyleLoaded()) return
    const vis = visible ? 'visible' : 'none'
    ;[FILL_LAYER, GLOW_LAYER, OUTLINE_LAYER].forEach((id) => {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', vis)
    })
  }, [map, visible])

  return null
}