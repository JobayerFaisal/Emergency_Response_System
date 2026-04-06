import React, { useState, useEffect } from 'react'
import KPIBar from './components/layout/KPIBar.jsx'
import Sidebar from './components/layout/Sidebar.jsx'
import MainMap from './components/map/MainMap.jsx'
import LayerControls from './components/map/LayerControls.jsx'
import useWebSocket from './hooks/useWebSocket.js'
import useDashboard from './hooks/useDashboard.js'
import useZones from './hooks/useZones.js'

export default function App() {
  const [activeLayers, setActiveLayers] = useState({
    flood: true,
    rain: true,
    teams: true,
    pins: true,
    river: true,
    routes: false,
  })
  const [activeAgent, setActiveAgent] = useState(null)
  const [sidebarTab, setSidebarTab] = useState('feed') // 'feed' | 'agent'

  const { events, connected } = useWebSocket('ws://localhost:8005/ws')
  const { kpi, agents } = useDashboard()
  const { zones } = useZones()

  const toggleLayer = (layer) => {
    setActiveLayers(prev => ({ ...prev, [layer]: !prev[layer] }))
  }

  const handleAgentClick = (agentId) => {
    setActiveAgent(agentId)
    setSidebarTab('agent')
  }

  return (
    <div className="app-shell">
      <KPIBar kpi={kpi} connected={connected} />

      <div className="main-layout">
        <div className="map-area">
          <MainMap
            zones={zones}
            activeLayers={activeLayers}
            agents={agents}
            events={events}
          />
          <LayerControls activeLayers={activeLayers} onToggle={toggleLayer} />
        </div>

        <Sidebar
          tab={sidebarTab}
          onTabChange={setSidebarTab}
          events={events}
          activeAgent={activeAgent}
          onAgentClick={handleAgentClick}
          agents={agents}
        />
      </div>
    </div>
  )
}
