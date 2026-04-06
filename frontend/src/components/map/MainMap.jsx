import React, { useRef, useEffect } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

// Sylhet division center
const DEFAULT_CENTER = [91.8687, 24.8949]
const DEFAULT_ZOOM = 9

export default function MainMap({ zones, activeLayers, agents, events }) {
  const mapContainer = useRef(null)
  const mapRef = useRef(null)
  const markersRef = useRef([])

  // Init map
  useEffect(() => {
    if (mapRef.current) return

    mapRef.current = new maplibregl.Map({
      container: mapContainer.current,
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
    })

    const map = mapRef.current

    map.addControl(new maplibregl.NavigationControl(), 'top-left')
    map.addControl(new maplibregl.ScaleControl(), 'bottom-left')

    map.on('load', () => {
      // ── Flood fill layer ──────────────────────────────
      map.addSource('flood-zones', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      })
      map.addLayer({
        id: 'flood-fill',
        type: 'fill',
        source: 'flood-zones',
        paint: {
          'fill-color': [
            'interpolate', ['linear'],
            ['get', 'risk'],
            0,   '#0033cc22',
            0.5, '#0055ff66',
            1,   '#ff2200cc',
          ],
          'fill-opacity': 0.65,
        },
      })
      map.addLayer({
        id: 'flood-outline',
        type: 'line',
        source: 'flood-zones',
        paint: {
          'line-color': '#00aaff',
          'line-width': 1.5,
        },
      })

      // ── River layer ───────────────────────────────────
      map.addSource('rivers', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      })
      map.addLayer({
        id: 'river-lines',
        type: 'line',
        source: 'rivers',
        paint: {
          'line-color': [
            'interpolate', ['linear'],
            ['get', 'discharge'],
            0,   '#2288cc',
            500, '#ffaa00',
            1000,'#ff3300',
          ],
          'line-width': 2.5,
        },
      })
    })

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  // Update flood zones
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.isStyleLoaded()) return
    if (!zones?.flood) return
    map.getSource('flood-zones')?.setData(zones.flood)
  }, [zones?.flood])

  // Toggle layers
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.isStyleLoaded()) return
    const vis = (on) => (on ? 'visible' : 'none')
    if (map.getLayer('flood-fill'))   map.setLayoutProperty('flood-fill',   'visibility', vis(activeLayers.flood))
    if (map.getLayer('flood-outline')) map.setLayoutProperty('flood-outline','visibility', vis(activeLayers.flood))
    if (map.getLayer('river-lines'))  map.setLayoutProperty('river-lines',  'visibility', vis(activeLayers.river))
  }, [activeLayers])

  // Team markers
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    markersRef.current.forEach(m => m.remove())
    markersRef.current = []

    if (!activeLayers.teams || !agents?.agent4?.teams) return

    agents.agent4.teams.forEach(team => {
      const el = document.createElement('div')
      el.className = 'team-marker'
      el.innerHTML = team.available ? '🟢' : '🔴'
      el.title = `${team.id} — ${team.district}`

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([team.lon, team.lat])
        .setPopup(
          new maplibregl.Popup({ offset: 25 }).setHTML(
            `<strong>${team.id}</strong><br/>District: ${team.district}<br/>Status: ${team.available ? 'Available' : 'Deployed'}<br/>Skills: ${team.skills?.join(', ')}`
          )
        )
        .addTo(map)

      markersRef.current.push(marker)
    })
  }, [agents?.agent4?.teams, activeLayers.teams])

  // Distress pins
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    if (!activeLayers.pins || !agents?.agent2?.reports) return

    agents.agent2.reports.forEach(report => {
      const el = document.createElement('div')
      el.className = 'distress-pin'
      el.innerHTML = report.urgency >= 5 ? '🆘' : '📍'
      el.title = report.text

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([report.lon, report.lat])
        .setPopup(
          new maplibregl.Popup({ offset: 20 }).setHTML(
            `<strong>Urgency: ${report.urgency}/5</strong><br/>${report.text}<br/>Credibility: ${Math.round(report.credibility * 100)}%`
          )
        )
        .addTo(map)

      markersRef.current.push(marker)
    })
  }, [agents?.agent2?.reports, activeLayers.pins])

  return (
    <div className="main-map-wrapper">
      <div ref={mapContainer} className="map-container" />
    </div>
  )
}
