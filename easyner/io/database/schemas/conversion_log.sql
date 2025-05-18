CREATE TABLE IF NOT EXISTS conversion_log (
    file_path VARCHAR NOT NULL, -- Full path to the file
    file_name VARCHAR NOT NULL, -- Filename with suffix, without parent directories
    file_hash VARCHAR NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    conversion_timestamp TIMESTAMP NOT NULL,
    status VARCHAR NOT NULL
);
