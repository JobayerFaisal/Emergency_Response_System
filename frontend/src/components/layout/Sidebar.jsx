import React from 'react'
import AgentFeed from './AgentFeed.jsx'
import Agent1Panel from '../panels/Agent1Panel.jsx'
import Agent2Panel from '../panels/Agent2Panel.jsx'
import Agent3Panel from '../panels/Agent3Panel.jsx'
import Agent4Panel from '../panels/Agent4Panel.jsx'

const AGENT_TABS = [
  { id: 'feed',   label: '📡 Live Feed' },
  { id: 'agent1', label: '🛰 Env' },
  { id: 'agent2', label: '📱 Distress' },
  { id: 'agent3', label: '📦 Resources' },
  { id: 'agent4', label: '🚤 Dispatch' },
]

export default function Sidebar({ tab, onTabChange, events, agents }) {
  return (
    <aside className="sidebar">
      <nav className="sidebar-tabs">
        {AGENT_TABS.map(t => (
          <button
            key={t.id}
            className={`sidebar-tab ${tab === t.id ? 'sidebar-tab--active' : ''}`}
            onClick={() => onTabChange(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-content">
        {tab === 'feed'   && <AgentFeed events={events} />}
        {tab === 'agent1' && <Agent1Panel data={agents?.agent1} />}
        {tab === 'agent2' && <Agent2Panel data={agents?.agent2} />}
        {tab === 'agent3' && <Agent3Panel data={agents?.agent3} />}
        {tab === 'agent4' && <Agent4Panel data={agents?.agent4} />}
      </div>
    </aside>
  )
}
