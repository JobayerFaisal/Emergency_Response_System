-- ============================================================
-- 001_base_schema.sql
-- Base tables required by Agent 1 (Environmental Intelligence).
--
-- Agent 1's spatial_analyzer.initialize_schema() creates these
-- with CREATE TABLE IF NOT EXISTS, but PostgreSQL runs migrations
-- in alphabetical order. Having them here guarantees PostGIS is
-- enabled and the base schema exists before any agent starts,
-- even if Agent 1 is slow to come up.
-- ============================================================

-- PostGIS extension (required by all spatial queries)
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Sentinel zones (Agent 1 monitors these) ──────────────────
CREATE TABLE IF NOT EXISTS sentinel_zones (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name               TEXT NOT NULL,
    center             GEOGRAPHY(POINT, 4326) NOT NULL,
    radius_km          FLOAT NOT NULL DEFAULT 5.0,
    risk_level         TEXT NOT NULL DEFAULT 'minimal'
                           CHECK (risk_level IN ('minimal','low','moderate','high','critical')),
    population_density INTEGER,
    elevation          FLOAT,
    drainage_capacity  TEXT,
    created_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_monitored     TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_sentinel_zones_center
    ON sentinel_zones USING GIST (center);

-- ── Weather observations (Agent 1 stores per-cycle data) ──────
CREATE TABLE IF NOT EXISTS weather_data (
    id            SERIAL PRIMARY KEY,
    zone_id       UUID REFERENCES sentinel_zones(id) ON DELETE CASCADE,
    timestamp     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    temperature   FLOAT,
    humidity      FLOAT,
    pressure      FLOAT,
    wind_speed    FLOAT,
    rainfall_mm   FLOAT,
    cloud_cover   FLOAT,
    raw_data      JSONB
);

CREATE INDEX IF NOT EXISTS idx_weather_data_zone_time
    ON weather_data (zone_id, timestamp DESC);

-- ── Social media posts (Agent 1 NLP pipeline) ─────────────────
CREATE TABLE IF NOT EXISTS social_media_posts (
    id            SERIAL PRIMARY KEY,
    zone_id       UUID REFERENCES sentinel_zones(id) ON DELETE CASCADE,
    timestamp     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    platform      TEXT,
    content       TEXT,
    location      GEOGRAPHY(POINT, 4326),
    sentiment     TEXT,
    flood_related BOOLEAN DEFAULT FALSE,
    urgency_score FLOAT,
    raw_data      JSONB
);

CREATE INDEX IF NOT EXISTS idx_social_posts_zone_time
    ON social_media_posts (zone_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_social_posts_location
    ON social_media_posts USING GIST (location);

-- ── Flood predictions (Agent 1 outputs, consumed by Agent 2) ──
CREATE TABLE IF NOT EXISTS flood_predictions (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id               UUID REFERENCES sentinel_zones(id) ON DELETE CASCADE,
    timestamp             TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    risk_score            FLOAT NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    severity_level        TEXT NOT NULL DEFAULT 'minimal'
                              CHECK (severity_level IN ('minimal','low','moderate','high','critical')),
    confidence            FLOAT CHECK (confidence BETWEEN 0 AND 1),
    time_to_impact_hours  FLOAT,
    affected_area_km2     FLOAT,
    risk_factors          JSONB,
    recommended_actions   JSONB
);

CREATE INDEX IF NOT EXISTS idx_flood_predictions_zone_time
    ON flood_predictions (zone_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_flood_predictions_severity
    ON flood_predictions (severity_level);
