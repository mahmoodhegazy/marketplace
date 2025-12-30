-- Freak AI Database Initialization Script
-- PostgreSQL with pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema
CREATE SCHEMA IF NOT EXISTS freak;

-- ===========================================
-- Items Table
-- ===========================================
CREATE TABLE IF NOT EXISTS freak.items (
    item_id SERIAL PRIMARY KEY,
    external_id VARCHAR(100) UNIQUE NOT NULL,
    category_id INTEGER NOT NULL,
    brand_id INTEGER,
    custom_brand VARCHAR(255),
    condition_id INTEGER NOT NULL,
    size_id INTEGER,
    price DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Item embedding (512-dimensional FashionCLIP)
    embedding vector(512),
    
    -- Metadata
    title VARCHAR(500),
    description TEXT,
    image_urls TEXT[],
    
    -- Indexes
    CONSTRAINT positive_price CHECK (price >= 0)
);

-- Create index on embedding for similarity search
CREATE INDEX IF NOT EXISTS items_embedding_idx ON freak.items 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS items_category_idx ON freak.items(category_id);
CREATE INDEX IF NOT EXISTS items_brand_idx ON freak.items(brand_id);
CREATE INDEX IF NOT EXISTS items_active_idx ON freak.items(is_active);
CREATE INDEX IF NOT EXISTS items_created_at_idx ON freak.items(created_at DESC);

-- ===========================================
-- Users Table
-- ===========================================
CREATE TABLE IF NOT EXISTS freak.users (
    user_id SERIAL PRIMARY KEY,
    external_id VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP,
    
    -- User preferences (computed)
    preferred_categories INTEGER[],
    preferred_brands INTEGER[],
    price_range_min DECIMAL(10, 2),
    price_range_max DECIMAL(10, 2),
    
    -- User embedding (32-dimensional from model)
    embedding vector(32)
);

CREATE INDEX IF NOT EXISTS users_external_id_idx ON freak.users(external_id);
CREATE INDEX IF NOT EXISTS users_embedding_idx ON freak.users 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- ===========================================
-- User Events Table
-- ===========================================
CREATE TABLE IF NOT EXISTS freak.user_events (
    event_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES freak.users(user_id),
    item_id INTEGER REFERENCES freak.items(item_id),
    event_type VARCHAR(20) NOT NULL,  -- 'save', 'cart', 'order'
    event_weight DECIMAL(3, 1) DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_event_type CHECK (event_type IN ('save', 'cart', 'order'))
);

CREATE INDEX IF NOT EXISTS events_user_idx ON freak.user_events(user_id);
CREATE INDEX IF NOT EXISTS events_item_idx ON freak.user_events(item_id);
CREATE INDEX IF NOT EXISTS events_created_at_idx ON freak.user_events(created_at DESC);
CREATE INDEX IF NOT EXISTS events_user_item_idx ON freak.user_events(user_id, item_id);

-- ===========================================
-- Categories Table
-- ===========================================
CREATE TABLE IF NOT EXISTS freak.categories (
    category_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    name_ar VARCHAR(100),  -- Arabic name
    parent_id INTEGER REFERENCES freak.categories(category_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- Brands Table
-- ===========================================
CREATE TABLE IF NOT EXISTS freak.brands (
    brand_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    name_ar VARCHAR(255),
    is_luxury BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- Conditions Table
-- ===========================================
CREATE TABLE IF NOT EXISTS freak.conditions (
    condition_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    name_ar VARCHAR(100),
    quality_score INTEGER CHECK (quality_score BETWEEN 1 AND 5)
);

-- ===========================================
-- Recommendations Cache Table
-- ===========================================
CREATE TABLE IF NOT EXISTS freak.recommendations_cache (
    cache_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES freak.users(user_id),
    recommendation_type VARCHAR(50) NOT NULL,  -- 'personalized', 'similar', 'trending'
    item_ids INTEGER[] NOT NULL,
    scores DECIMAL(5, 4)[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    
    UNIQUE(user_id, recommendation_type)
);

CREATE INDEX IF NOT EXISTS cache_user_type_idx ON freak.recommendations_cache(user_id, recommendation_type);
CREATE INDEX IF NOT EXISTS cache_expires_idx ON freak.recommendations_cache(expires_at);

-- ===========================================
-- Model Metadata Table
-- ===========================================
CREATE TABLE IF NOT EXISTS freak.models (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_path VARCHAR(500) NOT NULL,
    metrics JSONB,
    config JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, model_version)
);

-- ===========================================
-- A/B Test Results Table
-- ===========================================
CREATE TABLE IF NOT EXISTS freak.ab_test_results (
    result_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,
    variant VARCHAR(50) NOT NULL,
    user_id INTEGER REFERENCES freak.users(user_id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10, 4) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ab_test_experiment_idx ON freak.ab_test_results(experiment_name, variant);

-- ===========================================
-- Functions
-- ===========================================

-- Function to find similar items by embedding
CREATE OR REPLACE FUNCTION freak.find_similar_items(
    query_embedding vector(512),
    limit_count INTEGER DEFAULT 20,
    category_filter INTEGER DEFAULT NULL
)
RETURNS TABLE(
    item_id INTEGER,
    external_id VARCHAR,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        i.item_id,
        i.external_id,
        1 - (i.embedding <=> query_embedding) as similarity
    FROM freak.items i
    WHERE i.is_active = TRUE
      AND (category_filter IS NULL OR i.category_id = category_filter)
      AND i.embedding IS NOT NULL
    ORDER BY i.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get user recommendations
CREATE OR REPLACE FUNCTION freak.get_user_recommendations(
    p_user_id INTEGER,
    limit_count INTEGER DEFAULT 20
)
RETURNS TABLE(
    item_id INTEGER,
    external_id VARCHAR,
    score FLOAT
) AS $$
DECLARE
    user_emb vector(32);
BEGIN
    -- Get user embedding
    SELECT embedding INTO user_emb FROM freak.users WHERE user_id = p_user_id;
    
    IF user_emb IS NULL THEN
        -- Cold start: return trending items
        RETURN QUERY
        SELECT 
            i.item_id,
            i.external_id,
            COUNT(e.event_id)::FLOAT as score
        FROM freak.items i
        LEFT JOIN freak.user_events e ON i.item_id = e.item_id
        WHERE i.is_active = TRUE
          AND e.created_at > NOW() - INTERVAL '7 days'
        GROUP BY i.item_id, i.external_id
        ORDER BY score DESC
        LIMIT limit_count;
    ELSE
        -- Personalized recommendations using item embeddings
        -- This is a simplified version; actual implementation would use
        -- the two-tower model's item embeddings
        RETURN QUERY
        SELECT 
            i.item_id,
            i.external_id,
            RANDOM() as score  -- Placeholder; actual score from model
        FROM freak.items i
        WHERE i.is_active = TRUE
          AND i.item_id NOT IN (
              SELECT DISTINCT ue.item_id 
              FROM freak.user_events ue 
              WHERE ue.user_id = p_user_id
          )
        ORDER BY score DESC
        LIMIT limit_count;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- Triggers
-- ===========================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION freak.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER items_updated_at
    BEFORE UPDATE ON freak.items
    FOR EACH ROW
    EXECUTE FUNCTION freak.update_updated_at();

-- ===========================================
-- Initial Data
-- ===========================================

-- Insert default conditions
INSERT INTO freak.conditions (name, name_ar, quality_score) VALUES
    ('New with Tags', 'جديد بالتاج', 5),
    ('Like New', 'كالجديد', 4),
    ('Good', 'جيد', 3),
    ('Fair', 'مقبول', 2),
    ('Poor', 'ضعيف', 1)
ON CONFLICT DO NOTHING;

-- Insert sample categories
INSERT INTO freak.categories (name, name_ar) VALUES
    ('Dresses', 'فساتين'),
    ('Tops', 'بلايز'),
    ('Bottoms', 'بناطيل'),
    ('Outerwear', 'جاكيتات'),
    ('Bags', 'حقائب'),
    ('Accessories', 'اكسسوارات'),
    ('Shoes', 'أحذية')
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA freak TO freak;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA freak TO freak;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA freak TO freak;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA freak TO freak;
