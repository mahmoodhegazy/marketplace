# Freak AI Makefile
# Convenience commands for development and deployment

.PHONY: help install dev test lint format train serve docker-up docker-down clean

# Default target
help:
	@echo "Freak AI - Available Commands"
	@echo "=============================="
	@echo ""
	@echo "Development:"
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Start development server"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo ""
	@echo "Training:"
	@echo "  make embeddings  - Generate FashionCLIP embeddings"
	@echo "  make train       - Train the recommendation model"
	@echo "  make evaluate    - Evaluate model on test set"
	@echo ""
	@echo "Serving:"
	@echo "  make serve       - Start API server"
	@echo "  make docker-up   - Start all services with Docker"
	@echo "  make docker-down - Stop all services"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Remove cached files"
	@echo "  make docs        - Generate documentation"

# ===========================================
# Development
# ===========================================

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

dev:
	uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/
	ruff check src/ tests/ scripts/ --fix

# ===========================================
# Training Pipeline
# ===========================================

embeddings:
	python scripts/generate_embeddings.py \
		--items data/raw/items.csv \
		--output data/embeddings/ \
		--batch-size 32

train:
	python scripts/train.py \
		--config configs/config.yaml \
		--epochs 50

train-gpu:
	CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
		--config configs/config.yaml \
		--device cuda \
		--epochs 50

evaluate:
	python scripts/train.py \
		--config configs/config.yaml \
		--evaluate-only \
		--model-path checkpoints/best_model.pt

# ===========================================
# Serving
# ===========================================

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --workers 4

serve-prod:
	gunicorn src.serving.api:app \
		--workers 4 \
		--worker-class uvicorn.workers.UvicornWorker \
		--bind 0.0.0.0:8000 \
		--access-logfile - \
		--error-logfile -

# ===========================================
# Docker
# ===========================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-train:
	docker-compose --profile training up trainer

docker-embed:
	docker-compose --profile embedding up embedder

docker-prod:
	docker-compose --profile production up -d

# ===========================================
# Database
# ===========================================

db-init:
	docker-compose exec postgres psql -U freak -d freak_db -f /docker-entrypoint-initdb.d/init.sql

db-migrate:
	@echo "Run migrations here"

db-shell:
	docker-compose exec postgres psql -U freak -d freak_db

# ===========================================
# Utilities
# ===========================================

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name "*.egg-info" -delete
	rm -rf htmlcov/
	rm -rf .coverage

clean-all: clean
	rm -rf data/processed/*
	rm -rf data/embeddings/*
	rm -rf checkpoints/*
	rm -rf mlruns/*
	rm -rf logs/*

docs:
	@echo "Generate documentation"
	pdoc --html --output-dir docs/ src/

# ===========================================
# MLflow
# ===========================================

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# ===========================================
# Health Checks
# ===========================================

health:
	curl -s http://localhost:8000/health | python -m json.tool

status:
	@echo "API Status:"
	@curl -s http://localhost:8000/health || echo "API not running"
	@echo "\nRedis Status:"
	@docker-compose exec redis redis-cli ping || echo "Redis not running"
	@echo "\nPostgres Status:"
	@docker-compose exec postgres pg_isready -U freak || echo "Postgres not running"
