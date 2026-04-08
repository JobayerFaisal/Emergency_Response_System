import React, { useMemo, useRef, useEffect, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

import FloodLayer from './FloodLayer.jsx'
import RiverLayer from './RiverLayer.jsx'
import DistressPins from './DistressPins.jsx'
import TeamMarkers from './TeamMarkers.jsx'
import RouteMap from './RouteMap.jsx'
import RainLayer from './RainLayer.jsx'

// Sylhet-centered default view
const DEFAULT_CENTER = [91.8687, 24.8949]
const DEFAULT_ZOOM = 9

function normalizeFloodZones(zones) {
  const fc = zones?.flood || { type: 'FeatureCollection', features: [] }

  return {
    type: 'FeatureCollection',
    features: (fc.features || []).map((feature) => {
      const props = feature.properties || {}
      const risk =
        typeof props.risk === 'number'
          ? props.risk
          : typeof props.risk_score === 'number'
            ? props.risk_score
            : 0

      return {
        ...feature,
        properties: {
          ...props,
          risk,
          zone_name: props.zone_name || props.name || 'Unknown zone',
        },
      }
    }),
  }
}

function buildRiverGeoJSON(zones) {
  const flood = zones?.flood?.features || []

  return {
    type: 'FeatureCollection',
    features: flood
      .map((f, i) => {
        const p = f.properties || {}
        const center =
          p.center?.coordinates ||
          p.center_coordinates ||
          p.centroid_coordinates

        if (!Array.isArray(center) || center.length < 2) return null

        const [lon, lat] = center
        const discharge = Math.round((p.risk || p.risk_score || 0) * 1200)

        return {
          type: 'Feature',
          geometry: {
            type: 'LineString',
            coordinates: [
              [lon - 0.08, lat - 0.02],
              [lon, lat],
              [lon + 0.08, lat + 0.02],
            ],
          },
          properties: {
            id: `river-${i}`,
            discharge,
            zone_name: p.zone_name || p.name || 'Unknown',
          },
        }
      })
      .filter(Boolean),
  }
}

function buildReports(agent2, events) {
  const fromApi = Array.isArray(agent2?.reports) ? agent2.reports : []

  const fromWs = (events || [])
    .filter((ev) => ev.channel === 'distress_queue' || ev.type?.toLowerCase().includes('distress'))
    .map((ev, i) => {
      const p = ev.payload || {}
      return {
        report_id: p.report_id || p.id || `ws-report-${i}`,
        text: p.text || p.message || p.description || 'Distress report',
        lat: p.lat ?? p.latitude ?? p.location?.lat,
        lon: p.lon ?? p.longitude ?? p.lng ?? p.location?.lng,
        urgency: Number(p.urgency || p.severity || 3),
        credibility: Number(p.credibility || 0.7),
        district: p.district || p.zone_name || 'Unknown',
      }
    })
    .filter((r) => r.lat != null && r.lon != null)

  return [...fromApi, ...fromWs]
}

function buildTeams(agent4, events) {
  const fromApi = Array.isArray(agent4?.teams) ? agent4.teams : []

  const fromWs = (events || [])
    .filter((ev) => ev.channel === 'agent_status')
    .map((ev, i) => {
      const p = ev.payload || {}
      return {
        volunteer_id: p.volunteer_id || p.team_id || `team-${i}`,
        lat: p.lat ?? p.latitude ?? p.location?.lat,
        lon: p.lon ?? p.longitude ?? p.lng ?? p.location?.lng,
        available: p.available ?? (p.status !== 'deployed' && p.status !== 'busy'),
        district: p.district || p.zone_name || 'Unknown',
        skills: p.skills || [],
        equipment: p.equipment || [],
      }
    })
    .filter((t) => t.lat != null && t.lon != null)

  return [...fromApi, ...fromWs]
}

function buildRoutes(agent4, events) {
  const fromApi = Array.isArray(agent4?.routes) ? agent4.routes : []

  const fromWs = (events || [])
    .filter((ev) => ev.channel === 'route_assignment' || ev.channel === 'dispatch_order')
    .map((ev, i) => {
      const p = ev.payload || {}
      const geometry =
        p.geometry ||
        p.route_geometry ||
        p.route ||
        null

      if (!geometry?.coordinates) return null

      return {
        mission_id: p.mission_id || p.dispatch_id || `route-${i}`,
        volunteer_id: p.volunteer_id || p.team_id || 'unknown',
        status: p.status || 'ASSIGNED',
        geometry,
        distance_km: p.distance_km || p.distance || null,
        eta_minutes: p.eta_minutes || p.eta || null,
      }
    })
    .filter(Boolean)

  return [...fromApi, ...fromWs]
}

export default function MainMap({ zones, activeLayers, agents, events }) {
  const mapContainer = useRef(null)
  const mapRef = useRef(null)
  const [mapReady, setMapReady] = useState(false)

  const floodGeoJSON = useMemo(() => normalizeFloodZones(zones), [zones])
  const riverGeoJSON = useMemo(() => buildRiverGeoJSON(zones), [zones])
  const reports = useMemo(() => buildReports(agents?.agent2, events), [agents?.agent2, events])
  const teams = useMemo(() => buildTeams(agents?.agent4, events), [agents?.agent4, events])
  const routes = useMemo(() => buildRoutes(agents?.agent4, events), [agents?.agent4, events])

  const avgRisk = useMemo(() => {
    const features = floodGeoJSON?.features || []
    if (!features.length) return 0

    const total = features.reduce((sum, f) => {
      return sum + (f.properties?.risk || 0)
    }, 0)

    return total / features.length
  }, [floodGeoJSON])

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

    map.on('load', () => {
      setMapReady(true)

      if (floodGeoJSON.features.length) {
        const bounds = new maplibregl.LngLatBounds()

        floodGeoJSON.features.forEach((f) => {
          const polygonCoords = f?.geometry?.coordinates || []
          polygonCoords.forEach((ring) => {
            ring.forEach(([lng, lat]) => bounds.extend([lng, lat]))
          })
        })

        if (!bounds.isEmpty()) {
          map.fitBounds(bounds, { padding: 40, duration: 0, maxZoom: 10 })
        }
      }
    })

    mapRef.current = map

    return () => {
      map.remove()
      mapRef.current = null
      setMapReady(false)
    }
  }, [floodGeoJSON])

  return (
    <div className="main-map-wrapper" style={{ position: 'relative' }}>
      <div ref={mapContainer} className="map-container" />

      {mapReady && (
        <>
          <FloodLayer
            map={mapRef.current}
            geojson={floodGeoJSON}
            visible={activeLayers.flood}
          />

          <RiverLayer
            map={mapRef.current}
            geojson={riverGeoJSON}
            visible={activeLayers.river}
          />

          <DistressPins
            map={mapRef.current}
            reports={reports}
            visible={activeLayers.pins}
          />

          <TeamMarkers
            map={mapRef.current}
            teams={teams}
            visible={activeLayers.teams}
          />

          <RouteMap
            map={mapRef.current}
            routes={routes}
            visible={activeLayers.routes}
          />
        </>
      )}

      <RainLayer active={activeLayers.rain} intensity={avgRisk} />
    </div>
  )
}