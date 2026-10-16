-- Enable TimescaleDB (if available) and UUID extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Companies Table
CREATE TABLE IF NOT EXISTS companies (
    ticker VARCHAR(20) PRIMARY KEY,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE
);

-- 2. Daily Market Data Table
CREATE TABLE IF NOT EXISTS market_data_daily (
    date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL REFERENCES companies(ticker) ON DELETE CASCADE,
    open_price DECIMAL(15, 4),
    high_price DECIMAL(15, 4),
    low_price DECIMAL(15, 4),
    close_price DECIMAL(15, 4),
    adj_close DECIMAL(15, 4),
    volume BIGINT,
    PRIMARY KEY (date, ticker)
);

-- Note: We will attempt to create the hypertable dynamically in Python to handle environments where TimescaleDB is not installed.

-- 3. Financial Reports Table
CREATE TABLE IF NOT EXISTS financial_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(20) NOT NULL REFERENCES companies(ticker) ON DELETE CASCADE,
    filing_date DATE NOT NULL,
    report_type VARCHAR(50),
    source_url VARCHAR(500),
    headline TEXT,
    received_time TIMESTAMP,
    disseminated_time TIMESTAMP,
    raw_text JSONB
);

-- 4. NLP Sentiment Chunks Table
CREATE TABLE IF NOT EXISTS sentiment_chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    report_id UUID NOT NULL REFERENCES financial_reports(report_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT,
    score_positive FLOAT DEFAULT 0.0,
    score_negative FLOAT DEFAULT 0.0,
    score_neutral FLOAT DEFAULT 0.0,
    dominant_label VARCHAR(20)
);

-- 5. ML Daily Sentiment Aggregates Table
CREATE TABLE IF NOT EXISTS daily_sentiment_aggregates (
    date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL REFERENCES companies(ticker) ON DELETE CASCADE,
    net_sentiment_score FLOAT DEFAULT 0.0,
    bullish_intensity FLOAT DEFAULT 0.0,
    bearish_intensity FLOAT DEFAULT 0.0,
    total_chunks_analyzed INT DEFAULT 0,
    PRIMARY KEY (date, ticker)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_market_data_ticker ON market_data_daily(ticker);
CREATE INDEX IF NOT EXISTS idx_financial_reports_ticker ON financial_reports(ticker);
CREATE INDEX IF NOT EXISTS idx_financial_reports_filing_date ON financial_reports(filing_date);
CREATE INDEX IF NOT EXISTS idx_sentiment_chunks_report ON sentiment_chunks(report_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_aggregates_ticker ON daily_sentiment_aggregates(ticker);
