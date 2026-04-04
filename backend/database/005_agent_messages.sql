-- ============================================================================
-- 005_agent_messages.sql
-- Shared inter-agent message log (powers Dashboard alert feed)
-- Run once — shared by all 4 agents
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_messages (
    id             SERIAL PRIMARY KEY,
    message_id     UUID NOT NULL,
    timestamp      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sender_agent   VARCHAR(50) NOT NULL,
    receiver_agent VARCHAR(50) NOT NULL,
    message_type   VARCHAR(50) NOT NULL,
    zone_id        VARCHAR(100),
    priority       INTEGER DEFAULT 3,
    payload        JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_messages_time
    ON agent_messages(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_messages_type
    ON agent_messages(message_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_messages_zone
    ON agent_messages(zone_id, timestamp DESC);

-- ── STUB for Agent 1's flood detection table ──────────────────────────────
-- This allows Agent 4's safety_checker to compile even before Agent 1 merges.
-- Agent 1 will CREATE OR REPLACE this with the real schema.
-- Agent 4 handles the case gracefully (see safety_checker.py).
CREATE TABLE IF NOT EXISTS satellite_imagery_detections (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id         VARCHAR(100),
    risk_level      VARCHAR(20),
    flood_depth_m   FLOAT,
    flood_geometry  GEOGRAPHY(GEOMETRY, 4326),
    is_active       BOOLEAN DEFAULT TRUE,
    detected_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sat_detections_active
    ON satellite_imagery_detections(is_active);
