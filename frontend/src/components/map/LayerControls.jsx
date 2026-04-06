import React from 'react'

const LAYERS = [
  { key: 'flood',  label: 'Flood Zones', icon: '🌊' },
  { key: 'rain',   label: 'Rain Overlay', icon: '🌧' },
  { key: 'river',  label: 'River Levels', icon: '〰' },
  { key: 'teams',  label: 'Teams', icon: '👥' },
  { key: 'pins',   label: 'Distress Pins', icon: '📍' },
  { key: 'routes', label: 'Routes', icon: '🛣' },
]

export default function LayerControls({ activeLayers, onToggle }) {
  return (
    <div className="layer-controls">
      {LAYERS.map(({ key, label, icon }) => (
        <button
          key={key}
          className={`layer-btn ${activeLayers[key] ? 'layer-btn--on' : 'layer-btn--off'}`}
          onClick={() => onToggle(key)}
          title={label}
        >
          <span className="layer-btn-icon">{icon}</span>
          <span className="layer-btn-label">{label}</span>
        </button>
      ))}
    </div>
  )
}
