import { useEffect, useRef } from 'react'
import maplibregl from 'maplibre-gl'

/**
 * TeamMarkers
 * Renders volunteer/team position markers on the MapLibre map.
 *
 * Props:
 *   map      - maplibregl.Map instance
 *   teams    - array from Agent3/Agent4 data:
 *              [{ volunteer_id, lat, lon, available, district, skills[], equipment[] }]
 *   visible  - boolean, controlled by LayerControls (activeLayers.teams)
 *
 * Marker colors:
 *   🟢 green pulse = available
 *   🔴 red         = deployed / unavailable
 */
export default function TeamMarkers({ map, teams = [], visible }) {
  const markersRef = useRef([])

  useEffect(() => {
    markersRef.current.forEach(m => m.remove())
    markersRef.current = []

    if (!map || !visible || !teams.length) return

    teams.forEach(team => {
      if (team.lat == null || team.lon == null) return

      const isAvailable = team.available !== false

      const el = document.createElement('div')
      el.className = 'team-marker-el'
      el.style.cssText = `
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: ${isAvailable ? '#00e676' : '#ff3d00'};
        border: 2px solid ${isAvailable ? '#00ff88' : '#ff6633'};
        box-shadow: 0 0 ${isAvailable ? '8px #00e676' : '6px #ff3d00'};
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        transition: transform 0.12s;
        animation: ${isAvailable ? 'teamPulse 2s ease-in-out infinite' : 'none'};
      `
      el.textContent = isAvailable ? '🟢' : '🔴'
      el.title = `${team.volunteer_id} — ${team.district}`

      el.addEventListener('mouseenter', () => { el.style.transform = 'scale(1.3)' })
      el.addEventListener('mouseleave', () => { el.style.transform = 'scale(1)' })

      const popup = new maplibregl.Popup({ offset: 18, closeButton: true })
        .setHTML(`
          <strong>${isAvailable ? '🟢' : '🔴'} ${team.volunteer_id}</strong>
          <div style="font-size:12px;margin-top:5px;color:#a0c4e8;">
            <div>District: <b style="color:#d4e8ff">${team.district || '—'}</b></div>
            <div>Status: <b style="color:${isAvailable ? '#00e676' : '#ff3d00'}">${isAvailable ? 'Available' : 'Deployed'}</b></div>
            <div>Skills: ${team.skills?.join(', ') || '—'}</div>
            ${team.equipment?.length ? `<div>Equipment: ${team.equipment.join(', ')}</div>` : ''}
            <div style="font-family:monospace;font-size:10px;margin-top:4px;color:#5a85a8">
              ${team.lat?.toFixed(5)}, ${team.lon?.toFixed(5)}
            </div>
          </div>
        `)

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([team.lon, team.lat])
        .setPopup(popup)
        .addTo(map)

      markersRef.current.push(marker)
    })

    return () => {
      markersRef.current.forEach(m => m.remove())
      markersRef.current = []
    }
  }, [map, teams, visible])

  return null
}
