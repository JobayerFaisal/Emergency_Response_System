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

-- ── satellite_imagery_detections compatibility ───────────────────────────
-- The real table is created by 002_satellite_schema.sql (id SERIAL, no is_active).
-- 005 must NOT redefine it — that causes the IF NOT EXISTS to skip creation
-- while still trying to index a column that doesn't exist → ERROR.
--
-- FIX: Just add the columns Agent 4 needs if they are missing, then index safely.

-- Add is_active if 002 didn't include it
ALTER TABLE satellite_imagery_detections
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

-- Add zone_id alias if missing (002 has no zone_id column)
ALTER TABLE satellite_imagery_detections
    ADD COLUMN IF NOT EXISTS zone_id_ref VARCHAR(100);

-- Add flood_depth_m if missing
ALTER TABLE satellite_imagery_detections
    ADD COLUMN IF NOT EXISTS flood_depth_m FLOAT;

-- Now the index is safe because the column is guaranteed to exist
CREATE INDEX IF NOT EXISTS idx_sat_detections_active
    ON satellite_imagery_detections(is_active);