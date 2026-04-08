import { useEffect, useRef } from 'react'
import maplibregl from 'maplibre-gl'

function getUrgencyEmoji(urgency) {
  if (urgency >= 5) return '🆘'
  if (urgency >= 4) return '🚨'
  if (urgency >= 3) return '⚠️'
  return '📍'
}

function getPulseColor(urgency) {
  if (urgency >= 5) return '#ff3b30'
  if (urgency >= 4) return '#ff6b35'
  if (urgency >= 3) return '#ffb020'
  return '#4dabf7'
}

function getSeverityLabel(urgency) {
  if (urgency >= 5) return 'Critical'
  if (urgency >= 4) return 'High'
  if (urgency >= 3) return 'Urgent'
  return 'Moderate'
}

function buildPopupHtml(report) {
  const credPct = Math.round((report.credibility || 0) * 100)
  const urgency = Number(report.urgency || 1)

  return `
    <div style="min-width:220px">
      <strong style="font-size:13px">${getUrgencyEmoji(urgency)} ${getSeverityLabel(urgency)} distress</strong>
      <p style="margin:8px 0 6px;font-size:12px;color:#d7e7f5;line-height:1.45;">
        ${report.text || 'No description'}
      </p>
      <div style="font-size:11px;color:#9fc0dc;line-height:1.5;">
        <span><strong>District:</strong> ${report.district || '—'}</span><br/>
        <span><strong>Urgency:</strong> ${urgency}/5</span><br/>
        <span><strong>Credibility:</strong> ${credPct}%</span><br/>
        <span><strong>Coords:</strong> ${report.lat?.toFixed(5)}, ${report.lon?.toFixed(5)}</span>
      </div>
    </div>
  `
}

function createPinElement(report) {
  const urgency = Number(report.urgency || 1)
  const pulseColor = getPulseColor(urgency)
  const emoji = getUrgencyEmoji(urgency)

  const wrapper = document.createElement('div')
  wrapper.className = 'distress-pin-wrap'
  wrapper.style.cssText = `
    position: relative;
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
    cursor: pointer;
  `

  const pulseOuter = document.createElement('div')
  pulseOuter.style.cssText = `
    position: absolute;
    inset: 0;
    border-radius: 999px;
    background: ${pulseColor};
    opacity: 0.14;
    transform-origin: center;
    animation: distress-pulse-outer ${urgency >= 4 ? '1.3s' : '1.9s'} ease-out infinite;
    box-shadow: 0 0 28px ${pulseColor};
  `

  const pulseInner = document.createElement('div')
  pulseInner.style.cssText = `
    position: absolute;
    inset: 5px;
    border-radius: 999px;
    background: ${pulseColor};
    opacity: 0.26;
    transform-origin: center;
    animation: distress-pulse-inner ${urgency >= 4 ? '1.1s' : '1.6s'} ease-out infinite;
  `

  const core = document.createElement('div')
  core.className = 'distress-pin-core'
  core.textContent = emoji
  core.title = report.text || ''
  core.style.cssText = `
    position: relative;
    z-index: 2;
    width: 24px;
    height: 24px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: ${urgency >= 5 ? '17px' : '15px'};
    background: rgba(8, 15, 26, 0.92);
    border: 1px solid rgba(255,255,255,0.16);
    box-shadow: 0 0 0 2px ${pulseColor}, 0 4px 14px rgba(0,0,0,0.45);
    transition: transform 120ms ease, box-shadow 120ms ease;
    backdrop-filter: blur(4px);
  `

  wrapper.addEventListener('mouseenter', () => {
    core.style.transform = 'scale(1.18)'
    core.style.boxShadow = `0 0 0 2px ${pulseColor}, 0 8px 22px ${pulseColor}`
  })

  wrapper.addEventListener('mouseleave', () => {
    core.style.transform = 'scale(1)'
    core.style.boxShadow = `0 0 0 2px ${pulseColor}, 0 4px 14px rgba(0,0,0,0.45)`
  })

  wrapper.appendChild(pulseOuter)
  wrapper.appendChild(pulseInner)
  wrapper.appendChild(core)
  return wrapper
}

function ensurePulseStyles() {
  const id = 'distress-pin-pulse-style'
  if (document.getElementById(id)) return

  const style = document.createElement('style')
  style.id = id
  style.textContent = `
    @keyframes distress-pulse-outer {
      0%   { transform: scale(0.65); opacity: 0.30; }
      70%  { transform: scale(2.10); opacity: 0.06; }
      100% { transform: scale(2.45); opacity: 0; }
    }
    @keyframes distress-pulse-inner {
      0%   { transform: scale(0.75); opacity: 0.36; }
      70%  { transform: scale(1.55); opacity: 0.10; }
      100% { transform: scale(1.85); opacity: 0; }
    }
  `
  document.head.appendChild(style)
}

export default function DistressPins({ map, reports = [], visible }) {
  const markersRef = useRef([])

  useEffect(() => {
    ensurePulseStyles()
  }, [])

  useEffect(() => {
    markersRef.current.forEach((m) => m.remove())
    markersRef.current = []

    if (!map || !visible || !reports.length) return

    reports.forEach((report) => {
      if (report.lat == null || report.lon == null) return

      const el = createPinElement(report)

      const popup = new maplibregl.Popup({
        offset: 22,
        closeButton: true,
        closeOnClick: false,
      }).setHTML(buildPopupHtml(report))

      const marker = new maplibregl.Marker({
        element: el,
        anchor: 'center',
      })
        .setLngLat([report.lon, report.lat])
        .setPopup(popup)
        .addTo(map)

      markersRef.current.push(marker)
    })

    return () => {
      markersRef.current.forEach((m) => m.remove())
      markersRef.current = []
    }
  }, [map, reports, visible])

  return null
}