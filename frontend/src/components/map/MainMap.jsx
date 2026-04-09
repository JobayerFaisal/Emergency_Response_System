import React, { useMemo, useRef, useEffect, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

import FloodLayer from './FloodLayer.jsx'
import RiverLayer from './RiverLayer.jsx'
import DistressPins from './DistressPins.jsx'
import TeamMarkers from './TeamMarkers.jsx'
import RouteMap from './RouteMap.jsx'
import RainLayer from './RainLayer.jsx'

const DEFAULT_CENTER = [91.8687, 24.8949]
const DEFAULT_ZOOM = 9
const EMPTY_FC = { type: 'FeatureCollection', features: [] }

function normalizeFloodZones(zones) {
  // useZones already builds proper Polygon features — just pass through
  return zones?.flood || EMPTY_FC
}

function buildReports(agent2, events) {
  const fromApi = Array.isArray(agent2?.reports) ? agent2.reports : []

  // Deduplicate WS events by report_id — later events update the same pin
  const wsMap = new Map()
  ;(events || [])
    .filter((ev) => ev.channel === 'distress_queue' || ev.type?.toLowerCase().includes('distress'))
    .forEach((ev, i) => {
      const p = ev.payload || {}
      const id = p.report_id || p.id || `ws-report-${i}`
      const lat = p.lat ?? p.latitude ?? p.location?.lat
      const lon = p.lon ?? p.longitude ?? p.lng ?? p.location?.lng
      if (lat == null || lon == null) return
      wsMap.set(id, {
        report_id: id,
        text: p.text || p.message || p.description || 'Distress report',
        lat,
        lon,
        urgency: Number(p.urgency || p.severity || 3),
        credibility: Number(p.credibility || 0.7),
        district: p.district || p.zone_name || 'Unknown',
      })
    })

  // Merge: API reports + deduped WS reports (WS wins on same id)
  const merged = new Map(fromApi.map((r) => [r.report_id || r.id, r]))
  wsMap.forEach((v, k) => merged.set(k, v))
  return Array.from(merged.values()).filter((r) => r.lat != null && r.lon != null)
}

function buildTeams(agent4, events) {
  const fromApi = Array.isArray(agent4?.teams) ? agent4.teams : []

  const wsMap = new Map()
  ;(events || [])
    .filter((ev) => ev.channel === 'agent_status')
    .forEach((ev, i) => {
      const p = ev.payload || {}
      const id = p.volunteer_id || p.team_id || `team-${i}`
      const lat = p.lat ?? p.latitude ?? p.location?.lat
      const lon = p.lon ?? p.longitude ?? p.lng ?? p.location?.lng
      if (lat == null || lon == null) return
      wsMap.set(id, {
        volunteer_id: id,
        lat, lon,
        available: p.available ?? (p.status !== 'deployed' && p.status !== 'busy'),
        district: p.district || p.zone_name || 'Unknown',
        skills: p.skills || [],
        equipment: p.equipment || [],
      })
    })

  const merged = new Map(fromApi.map((t) => [t.volunteer_id || t.id, t]))
  wsMap.forEach((v, k) => merged.set(k, v))
  return Array.from(merged.values()).filter((t) => t.lat != null && t.lon != null)
}

function buildRoutes(agent4, events) {
  const fromApi = Array.isArray(agent4?.routes) ? agent4.routes : []

  const wsMap = new Map()
  ;(events || [])
    .filter((ev) => ev.channel === 'route_assignment' || ev.channel === 'dispatch_order')
    .forEach((ev, i) => {
      const p = ev.payload || {}
      const geometry = p.geometry || p.route_geometry || p.route || null
      if (!geometry?.coordinates) return
      const id = p.mission_id || p.dispatch_id || `route-${i}`
      wsMap.set(id, {
        mission_id: id,
        volunteer_id: p.volunteer_id || p.team_id || 'unknown',
        status: p.status || 'ASSIGNED',
        geometry,
        distance_km: p.distance_km || p.distance || null,
        eta_minutes: p.eta_minutes || p.eta || null,
      })
    })

  const merged = new Map(fromApi.map((r) => [r.mission_id || r.id, r]))
  wsMap.forEach((v, k) => merged.set(k, v))
  return Array.from(merged.values())
}

export default function MainMap({ zones, activeLayers, agents, events }) {
  const mapContainer = useRef(null)
  const mapRef = useRef(null)
  const [mapReady, setMapReady] = useState(false)
  const fittedRef = useRef(false)

  // BUG FIX: useZones already has proper Polygon geometry — use it directly.
  // Old buildRiverGeoJSON() looked for p.center which doesn't exist in props.
  // Now we use zones.rivers which useZones synthesizes from the zone centroids.
  const floodGeoJSON = useMemo(() => normalizeFloodZones(zones), [zones])
  const riverGeoJSON = useMemo(() => zones?.rivers || EMPTY_FC, [zones])
  const reports = useMemo(() => buildReports(agents?.agent2, events), [agents?.agent2, events])
  const teams   = useMemo(() => buildTeams(agents?.agent4, events),   [agents?.agent4, events])
  const routes  = useMemo(() => buildRoutes(agents?.agent4, events),  [agents?.agent4, events])

  const avgRisk = useMemo(() => {
    const features = floodGeoJSON?.features || []
    if (!features.length) return 0
    const total = features.reduce((sum, f) => sum + (f.properties?.risk || 0), 0)
    return total / features.length
  }, [floodGeoJSON])

  // ── Map init — empty deps so map is created ONCE ───────────────────────────
  useEffect(() => {
    if (mapRef.current) return

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      attributionControl: false,
    })

    map.addControl(new maplibregl.NavigationControl(), 'top-left')
    map.addControl(new maplibregl.ScaleControl(), 'bottom-left')
    map.on('load', () => setMapReady(true))

    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
      fittedRef.current = false
      setMapReady(false)
    }
  }, [])

  // ── Fit bounds once when flood data first arrives ─────────────────────────
  useEffect(() => {
    if (!mapReady || fittedRef.current) return
    const features = floodGeoJSON?.features || []
    if (!features.length) return

    const bounds = new maplibregl.LngLatBounds()
    features.forEach((f) => {
      const polygonCoords = f?.geometry?.coordinates || []
      polygonCoords.forEach((ring) => {
        ring.forEach(([lng, lat]) => bounds.extend([lng, lat]))
      })
    })

    if (!bounds.isEmpty()) {
      mapRef.current.fitBounds(bounds, { padding: 40, duration: 800, maxZoom: 10 })
      fittedRef.current = true
    }
  }, [mapReady, floodGeoJSON])

  return (
    <div className="main-map-wrapper" style={{ position: 'relative' }}>
      <div ref={mapContainer} className="map-container" />

      {mapReady && (
        <>
          <FloodLayer map={mapRef.current} geojson={floodGeoJSON} visible={activeLayers.flood} />
          <RiverLayer map={mapRef.current} geojson={riverGeoJSON} visible={activeLayers.river} />
          <DistressPins map={mapRef.current} reports={reports} visible={activeLayers.pins} />
          <TeamMarkers map={mapRef.current} teams={teams} visible={activeLayers.teams} />
          <RouteMap map={mapRef.current} routes={routes} visible={activeLayers.routes} />
        </>
      )}

      <RainLayer active={activeLayers.rain} intensity={avgRisk} />
    </div>
  )
}