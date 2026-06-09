-- ============================================================
-- 001_base_schema.sql
-- Base tables for Agent 1 (Environmental Intelligence).
-- Column names/types MUST match spatial_analyzer.initialize_schema()
-- ============================================================

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Sentinel zones
CREATE TABLE IF NOT EXISTS sentinel_zones (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name               VARCHAR(255) NOT NULL,
    center             GEOMETRY(Point, 4326) NOT NULL,
    radius_km          FLOAT NOT NULL,
    risk_level         VARCHAR(50) NOT NULL,
    population_density INTEGER,
    elevation          FLOAT,
    drainage_capacity  VARCHAR(50),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_monitored     TIMESTAMPTZ
);

-- Weather data (location column required for GIST index)
CREATE TABLE IF NOT EXISTS weather_data (
    id                 UUID PRIMARY KEY,
    zone_id            UUID REFERENCES sentinel_zones(id),
    timestamp          TIMESTAMPTZ NOT NULL,
    location           GEOMETRY(Point, 4326) NOT NULL,
    temperature        FLOAT,
    humidity           FLOAT,
    pressure           FLOAT,
    wind_speed         FLOAT,
    precipitation_1h   FLOAT,
    precipitation_3h   FLOAT,
    precipitation_24h  FLOAT,
    condition          VARCHAR(50),
    raw_data           JSONB,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Social media posts
CREATE TABLE IF NOT EXISTS social_media_posts (
    id                    UUID PRIMARY KEY,
    platform_id           VARCHAR(255) UNIQUE NOT NULL,
    zone_id               UUID REFERENCES sentinel_zones(id),
    timestamp             TIMESTAMPTZ NOT NULL,
    content               TEXT NOT NULL,
    author                VARCHAR(255),
    location              GEOMETRY(Point, 4326),
    relevance_score       FLOAT,
    sentiment             VARCHAR(50),
    contains_flood_report BOOLEAN DEFAULT FALSE,
    enriched              BOOLEAN DEFAULT FALSE,
    raw_data              JSONB,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Flood predictions
CREATE TABLE IF NOT EXISTS flood_predictions (
    id                   UUID PRIMARY KEY,
    zone_id              UUID REFERENCES sentinel_zones(id),
    timestamp            TIMESTAMPTZ NOT NULL,
    risk_score           FLOAT NOT NULL,
    severity_level       VARCHAR(50) NOT NULL,
    confidence           FLOAT NOT NULL,
    time_to_impact_hours FLOAT,
    affected_area_km2    FLOAT,
    risk_factors         JSONB,
    recommended_actions  JSONB,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Spatial indexes
CREATE INDEX IF NOT EXISTS idx_zones_center    ON sentinel_zones     USING GIST(center);
CREATE INDEX IF NOT EXISTS idx_weather_location ON weather_data       USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_social_location  ON social_media_posts USING GIST(location);

-- Time indexes
CREATE INDEX IF NOT EXISTS idx_weather_timestamp ON weather_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_social_timestamp  ON social_media_posts(timestamp DESC);
