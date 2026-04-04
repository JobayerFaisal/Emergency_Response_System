-- ============================================================================
-- 004_dispatch_schema.sql
-- Database Schema for Dispatch Optimization (Agent 4)
-- Run after: 003_resource_schema.sql
-- ============================================================================

CREATE TABLE IF NOT EXISTS dispatch_routes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    allocation_id       UUID,   -- References resource_allocations(id) — soft FK
    incident_id         VARCHAR(100) NOT NULL,
    zone_id             VARCHAR(100) NOT NULL,
    zone_name           VARCHAR(200),
    destination         GEOGRAPHY(POINT, 4326) NOT NULL,
    priority            INTEGER CHECK (priority >= 1 AND priority <= 5),
    total_eta_minutes   FLOAT,
    route_safety_score  FLOAT CHECK (route_safety_score >= 0 AND route_safety_score <= 1),
    status              VARCHAR(20) DEFAULT 'active',
    completed_at        TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS team_routes (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dispatch_id    UUID REFERENCES dispatch_routes(id) ON DELETE CASCADE,
    unit_id        UUID,   -- Soft ref to resource_units(id)
    unit_name      VARCHAR(100),
    resource_type  VARCHAR(30),
    transport_mode VARCHAR(20) CHECK (transport_mode IN ('road','waterway')),
    origin         GEOGRAPHY(POINT, 4326) NOT NULL,
    destination    GEOGRAPHY(POINT, 4326) NOT NULL,
    route_geometry JSONB,
    distance_km    FLOAT,
    eta_minutes    FLOAT,
    status         VARCHAR(20) DEFAULT 'dispatched',
    departed_at    TIMESTAMP WITH TIME ZONE,
    arrived_at     TIMESTAMP WITH TIME ZONE,
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_dispatch_routes_zone
    ON dispatch_routes(zone_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_dispatch_routes_status
    ON dispatch_routes(status);
CREATE INDEX IF NOT EXISTS idx_team_routes_dispatch
    ON team_routes(dispatch_id);
CREATE INDEX IF NOT EXISTS idx_team_routes_unit
    ON team_routes(unit_id);
CREATE INDEX IF NOT EXISTS idx_team_routes_status
    ON team_routes(status);

-- View: active dispatches with team summaries
CREATE OR REPLACE VIEW active_dispatches AS
SELECT
    dr.id            AS dispatch_id,
    dr.zone_name,
    dr.priority,
    dr.total_eta_minutes,
    dr.route_safety_score,
    COUNT(tr.id)           AS team_count,
    ARRAY_AGG(tr.unit_name) AS team_names,
    MIN(tr.eta_minutes)    AS fastest_eta,
    MAX(tr.eta_minutes)    AS slowest_eta
FROM dispatch_routes dr
JOIN team_routes tr ON tr.dispatch_id = dr.id
WHERE dr.status = 'active'
GROUP BY dr.id, dr.zone_name, dr.priority,
         dr.total_eta_minutes, dr.route_safety_score;
