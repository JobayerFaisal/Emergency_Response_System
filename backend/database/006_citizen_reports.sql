-- ============================================================================
-- 006_citizen_reports.sql
-- Tables for citizen-submitted ground reports and Agent 1 validation results
-- ============================================================================

CREATE TABLE IF NOT EXISTS citizen_reports (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    phone       TEXT,
    latitude    DOUBLE PRECISION NOT NULL,
    longitude   DOUBLE PRECISION NOT NULL,
    category    TEXT NOT NULL DEFAULT 'Flooding',
    message     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'validated', 'rejected', 'dispatched')),
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_citizen_reports_status     ON citizen_reports (status);
CREATE INDEX IF NOT EXISTS idx_citizen_reports_created_at ON citizen_reports (created_at DESC);

-- Stores Agent 1's cross-validation of each citizen report against
-- satellite / weather data. One-to-one with citizen_reports.
CREATE TABLE IF NOT EXISTS citizen_report_validations (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id        UUID NOT NULL REFERENCES citizen_reports (id) ON DELETE CASCADE,
    flood_risk_level TEXT NOT NULL CHECK (flood_risk_level IN ('low','moderate','high','critical','extreme')),
    risk_score       NUMERIC(5,2),           -- 0–100
    claim_validity   BOOLEAN NOT NULL DEFAULT TRUE,
    validation_notes TEXT,
    created_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_crv_report_id ON citizen_report_validations (report_id);
