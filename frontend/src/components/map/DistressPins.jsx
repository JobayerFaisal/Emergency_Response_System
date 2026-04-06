import { useEffect, useRef } from 'react'
import maplibregl from 'maplibre-gl'

/**
 * DistressPins
 * Renders distress report pins on the MapLibre map as custom DOM markers.
 *
 * Props:
 *   map        - maplibregl.Map instance (passed down from MainMap)
 *   reports    - array from Agent2 data: [{ report_id, text, lat, lon, urgency, credibility, district }]
 *   visible    - boolean, controlled by LayerControls (activeLayers.pins)
 */
export default function DistressPins({ map, reports = [], visible }) {
  const markersRef = useRef([])

  useEffect(() => {
    // Clear existing pins
    markersRef.current.forEach(m => m.remove())
    markersRef.current = []

    if (!map || !visible || !reports.length) return

    reports.forEach(report => {
      if (report.lat == null || report.lon == null) return

      // Choose icon by urgency level
      const emoji =
        report.urgency >= 5 ? '🆘' :
        report.urgency >= 4 ? '🚨' :
        '📍'

      const el = document.createElement('div')
      el.className = 'distress-pin'
      el.textContent = emoji
      el.title = report.text || ''
      el.style.cssText = `
        font-size: 20px;
        cursor: pointer;
        filter: drop-shadow(0 2px 6px rgba(0,0,0,0.9));
        transition: transform 0.12s;
        user-select: none;
      `
      el.addEventListener('mouseenter', () => { el.style.transform = 'scale(1.35)' })
      el.addEventListener('mouseleave', () => { el.style.transform = 'scale(1)' })

      const credPct = Math.round((report.credibility || 0) * 100)
      const urgencyLabel = ['', 'Low', 'Low-Med', 'Medium', 'High', 'Critical'][report.urgency] || '—'

      const popup = new maplibregl.Popup({ offset: 20, closeButton: true })
        .setHTML(`
          <strong>${emoji} Urgency ${report.urgency}/5 — ${urgencyLabel}</strong>
          <p style="margin:6px 0 4px;font-size:12px;color:#a0c4e8;">"${report.text || 'No description'}"</p>
          <div style="font-size:11px;color:#5a85a8;">
            <span>District: ${report.district || '—'}</span><br/>
            <span>Credibility: ${credPct}%</span><br/>
            <span style="font-family:monospace">${report.lat?.toFixed(5)}, ${report.lon?.toFixed(5)}</span>
          </div>
        `)

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([report.lon, report.lat])
        .setPopup(popup)
        .addTo(map)

      markersRef.current.push(marker)
    })

    return () => {
      markersRef.current.forEach(m => m.remove())
      markersRef.current = []
    }
  }, [map, reports, visible])

  return null
}
