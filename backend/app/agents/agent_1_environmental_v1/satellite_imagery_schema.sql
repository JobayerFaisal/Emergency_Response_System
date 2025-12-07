-- ============================================================================
-- Database Schema for Satellite Imagery Data
-- For Agent 1: Environmental Intelligence
-- ============================================================================

-- Enable PostGIS extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================================
-- TABLE: satellite_imagery_detections
-- Stores individual flood detection results from satellite analysis
-- ============================================================================

CREATE TABLE IF NOT EXISTS satellite_imagery_detections (
    -- Primary key
    id SERIAL PRIMARY KEY,
    
    -- Temporal information
    detection_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    analysis_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    analysis_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    baseline_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    baseline_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Spatial information
    center_location GEOGRAPHY(POINT, 4326) NOT NULL,
    analysis_radius_km FLOAT NOT NULL,
    analysis_area GEOGRAPHY(POLYGON, 4326),
    
    -- Detection results
    flood_detected BOOLEAN NOT NULL DEFAULT FALSE,
    flood_area_km2 FLOAT NOT NULL DEFAULT 0.0,
    flood_pixels INTEGER NOT NULL DEFAULT 0,
    
    -- Threat assessment
    threat_level VARCHAR(20) NOT NULL CHECK (threat_level IN 
        ('none', 'low', 'moderate', 'high', 'critical')),
    confidence_score FLOAT NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    -- GEE metadata
    sentinel1_images_used INTEGER,
    polarization VARCHAR(10),
    orbit_pass VARCHAR(20),
    
    -- Processing parameters
    smoothing_radius_m FLOAT,
    detection_threshold_db FLOAT,
    
    -- Affected areas
    affected_regions JSONB,  -- Array of affected region objects
    
    -- Visualization
    geojson_data JSONB,  -- Complete GeoJSON of flood extent
    map_tile_urls JSONB,  -- URLs for map visualization
    
    -- Status and metadata
    processing_status VARCHAR(20) DEFAULT 'completed',
    error_message TEXT,
    raw_metadata JSONB,
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_satellite_detection_timestamp ON satellite_imagery_detections(detection_timestamp DESC);
CREATE INDEX idx_satellite_threat_level ON satellite_imagery_detections(threat_level);
CREATE INDEX idx_satellite_flood_detected ON satellite_imagery_detections(flood_detected);
CREATE INDEX idx_satellite_location ON satellite_imagery_detections USING GIST(center_location);
CREATE INDEX idx_satellite_area ON satellite_imagery_detections USING GIST(analysis_area);
CREATE INDEX idx_satellite_affected_regions ON satellite_imagery_detections USING GIN(affected_regions);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_satellite_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_satellite_updated_at
    BEFORE UPDATE ON satellite_imagery_detections
    FOR EACH ROW
    EXECUTE FUNCTION update_satellite_updated_at();


-- ============================================================================
-- TABLE: satellite_flood_zones
-- Stores specific flood zone geometries for detailed analysis
-- ============================================================================

CREATE TABLE IF NOT EXISTS satellite_flood_zones (
    -- Primary key
    id SERIAL PRIMARY KEY,
    
    -- Foreign key to detection
    detection_id INTEGER REFERENCES satellite_imagery_detections(id) ON DELETE CASCADE,
    
    -- Zone information
    zone_number INTEGER NOT NULL,
    zone_geometry GEOGRAPHY(POLYGON, 4326) NOT NULL,
    zone_area_km2 FLOAT NOT NULL,
    
    -- Zone characteristics
    severity VARCHAR(20),  -- local severity within the zone
    population_affected INTEGER,  -- estimated from census data
    infrastructure_affected JSONB,  -- roads, buildings, etc.
    
    -- Temporal tracking
    first_detected_at TIMESTAMP WITH TIME ZONE,
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_flood_zone_detection ON satellite_flood_zones(detection_id);
CREATE INDEX idx_flood_zone_geometry ON satellite_flood_zones USING GIST(zone_geometry);
CREATE INDEX idx_flood_zone_severity ON satellite_flood_zones(severity);


-- ============================================================================
-- TABLE: satellite_monitoring_schedule
-- Tracks monitoring schedule and automation
-- ============================================================================

CREATE TABLE IF NOT EXISTS satellite_monitoring_schedule (
    -- Primary key
    id SERIAL PRIMARY KEY,
    
    -- Location to monitor
    location_name VARCHAR(255) NOT NULL,
    center_point GEOGRAPHY(POINT, 4326) NOT NULL,
    radius_km FLOAT NOT NULL,
    
    -- Schedule
    monitoring_frequency_hours INTEGER NOT NULL DEFAULT 6,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Last run information
    last_check_at TIMESTAMP WITH TIME ZONE,
    last_detection_id INTEGER REFERENCES satellite_imagery_detections(id),
    consecutive_flood_detections INTEGER DEFAULT 0,
    
    -- Alert settings
    alert_threshold VARCHAR(20) DEFAULT 'moderate',
    alert_enabled BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_monitoring_schedule_active ON satellite_monitoring_schedule(active);
CREATE INDEX idx_monitoring_schedule_location ON satellite_monitoring_schedule USING GIST(center_point);


-- ============================================================================
-- TABLE: satellite_image_cache
-- Cache metadata for processed Sentinel-1 images to avoid reprocessing
-- ============================================================================

CREATE TABLE IF NOT EXISTS satellite_image_cache (
    -- Primary key
    id SERIAL PRIMARY KEY,
    
    -- Image identification
    sentinel1_id VARCHAR(255) NOT NULL UNIQUE,
    acquisition_date TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Spatial coverage
    footprint GEOGRAPHY(POLYGON, 4326),
    
    -- Image properties
    polarization VARCHAR(10),
    orbit_pass VARCHAR(20),
    instrument_mode VARCHAR(10),
    
    -- Processing status
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP WITH TIME ZONE,
    
    -- Storage
    gee_asset_id VARCHAR(500),
    local_cache_path VARCHAR(500),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_image_cache_sentinel_id ON satellite_image_cache(sentinel1_id);
CREATE INDEX idx_image_cache_date ON satellite_image_cache(acquisition_date);
CREATE INDEX idx_image_cache_footprint ON satellite_image_cache USING GIST(footprint);


-- ============================================================================
-- VIEW: recent_flood_detections
-- Easy access to recent flood detections with key information
-- ============================================================================

CREATE OR REPLACE VIEW recent_flood_detections AS
SELECT 
    id,
    detection_timestamp,
    ST_Y(center_location::geometry) as latitude,
    ST_X(center_location::geometry) as longitude,
    flood_detected,
    flood_area_km2,
    threat_level,
    confidence_score,
    affected_regions,
    processing_status
FROM satellite_imagery_detections
WHERE detection_timestamp > NOW() - INTERVAL '7 days'
ORDER BY detection_timestamp DESC;


-- ============================================================================
-- VIEW: active_flood_zones
-- Currently active flood zones from recent detections
-- ============================================================================

CREATE OR REPLACE VIEW active_flood_zones AS
SELECT 
    fz.id,
    fz.zone_number,
    fz.zone_area_km2,
    fz.severity,
    fz.population_affected,
    sid.detection_timestamp,
    sid.threat_level,
    sid.flood_area_km2 as total_flood_area_km2,
    ST_AsGeoJSON(fz.zone_geometry::geometry) as zone_geojson
FROM satellite_flood_zones fz
JOIN satellite_imagery_detections sid ON fz.detection_id = sid.id
WHERE sid.detection_timestamp > NOW() - INTERVAL '24 hours'
  AND sid.flood_detected = TRUE
ORDER BY sid.detection_timestamp DESC, fz.zone_number;


-- ============================================================================
-- FUNCTION: get_flood_trend
-- Get flood area trend over time for a location
-- ============================================================================

CREATE OR REPLACE FUNCTION get_flood_trend(
    p_latitude FLOAT,
    p_longitude FLOAT,
    p_radius_km FLOAT DEFAULT 50,
    p_days INTEGER DEFAULT 7
)
RETURNS TABLE (
    detection_date TIMESTAMP WITH TIME ZONE,
    flood_area_km2 FLOAT,
    threat_level VARCHAR(20),
    confidence_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        detection_timestamp as detection_date,
        satellite_imagery_detections.flood_area_km2,
        satellite_imagery_detections.threat_level,
        satellite_imagery_detections.confidence_score
    FROM satellite_imagery_detections
    WHERE 
        ST_DWithin(
            center_location,
            ST_SetSRID(ST_MakePoint(p_longitude, p_latitude), 4326)::geography,
            p_radius_km * 1000
        )
        AND detection_timestamp > NOW() - (p_days || ' days')::INTERVAL
        AND flood_detected = TRUE
    ORDER BY detection_timestamp ASC;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- FUNCTION: calculate_threat_score
-- Calculate numerical threat score based on multiple factors
-- ============================================================================

CREATE OR REPLACE FUNCTION calculate_threat_score(
    p_flood_area_km2 FLOAT,
    p_confidence FLOAT,
    p_affected_regions JSONB
)
RETURNS FLOAT AS $$
DECLARE
    v_score FLOAT;
    v_region_count INTEGER;
BEGIN
    -- Base score from flood area (0-100)
    v_score := LEAST(p_flood_area_km2, 100);
    
    -- Weight by confidence
    v_score := v_score * p_confidence;
    
    -- Add bonus for multiple affected regions
    v_region_count := jsonb_array_length(p_affected_regions);
    v_score := v_score + (v_region_count * 5);
    
    -- Cap at 100
    v_score := LEAST(v_score, 100);
    
    RETURN v_score;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- SAMPLE DATA (for testing)
-- ============================================================================

-- Insert a test monitoring location (Dhaka)
INSERT INTO satellite_monitoring_schedule (
    location_name,
    center_point,
    radius_km,
    monitoring_frequency_hours,
    active
) VALUES (
    'Dhaka Metropolitan Area',
    ST_SetSRID(ST_MakePoint(90.4125, 23.8103), 4326)::geography,
    50,
    6,
    TRUE
) ON CONFLICT DO NOTHING;


-- ============================================================================
-- GRANTS (adjust based on your user roles)
-- ============================================================================

-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO agent1_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO agent1_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO agent1_user;


-- ============================================================================
-- USEFUL QUERIES
-- ============================================================================

-- Query 1: Get latest flood detection for Dhaka
-- SELECT * FROM recent_flood_detections LIMIT 10;

-- Query 2: Get flood trend for Dhaka over last 7 days
-- SELECT * FROM get_flood_trend(23.8103, 90.4125, 50, 7);

-- Query 3: Get all active flood zones
-- SELECT * FROM active_flood_zones;

-- Query 4: Get detection statistics by threat level
-- SELECT 
--     threat_level,
--     COUNT(*) as detection_count,
--     AVG(flood_area_km2) as avg_flood_area,
--     AVG(confidence_score) as avg_confidence
-- FROM satellite_imagery_detections
-- WHERE detection_timestamp > NOW() - INTERVAL '30 days'
-- GROUP BY threat_level
-- ORDER BY 
--     CASE threat_level
--         WHEN 'critical' THEN 1
--         WHEN 'high' THEN 2
--         WHEN 'moderate' THEN 3
--         WHEN 'low' THEN 4
--         WHEN 'none' THEN 5
--     END;

-- Query 5: Get locations with consecutive flood detections
-- SELECT 
--     location_name,
--     consecutive_flood_detections,
--     last_check_at
-- FROM satellite_monitoring_schedule
-- WHERE consecutive_flood_detections > 2
-- ORDER BY consecutive_flood_detections DESC;
