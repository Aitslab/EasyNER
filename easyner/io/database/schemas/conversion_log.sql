CREATE TABLE IF NOT EXISTS conversion_log (
    file_path TEXT PRIMARY KEY,
    file_name TEXT NOT NULL, -- Filename with suffix, without parent directories
    file_hash TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    conversion_timestamp TIMESTAMP NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversion_log_name ON conversion_log (file_name);
CREATE INDEX IF NOT EXISTS idx_conversion_log_hash ON conversion_log (file_hash);
