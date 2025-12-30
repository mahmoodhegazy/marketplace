# Freak AI - Fashion Resale Recommendation System

<p align="center">
  <img src="docs/logo.png" alt="Freak AI Logo" width="200"/>
</p>

> AI-powered recommendation system for Freak fashion resale marketplace, serving the MENA region.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

Freak AI is a production-ready recommendation system designed for fashion resale marketplaces. It combines state-of-the-art visual embeddings (FashionCLIP) with a two-tower neural network architecture to deliver personalized recommendations.

### Key Features

- **Visual Understanding**: FashionCLIP embeddings capture fashion-specific visual features
- **Two-Tower Architecture**: Efficient retrieval with separate user and item towers
- **Real-time Serving**: Sub-100ms recommendations with FAISS and Redis caching
- **Cold Start Handling**: Trending and session-based recommendations for new users
- **MENA Localization**: Arabic NLP support with CAMeL Tools
- **Production Ready**: Docker deployment, MLflow tracking, A/B testing

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Freak AI System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  User App   │───▶│  FastAPI    │───▶│  Recommendation     │  │
│  │  (Mobile)   │    │  Server     │    │  Engine             │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                            │                     │               │
│                            ▼                     ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Redis      │    │  PostgreSQL │    │  FAISS Index        │  │
│  │  Cache      │    │  + pgvector │    │  (Item Embeddings)  │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                       Model Components                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Two-Tower Model                         │   │
│  │  ┌────────────┐              ┌────────────────────────┐  │   │
│  │  │ User Tower │              │     Item Tower         │  │   │
│  │  │            │              │                        │  │   │
│  │  │ user_id    │              │ item_id + category +   │  │   │
│  │  │ + prefs    │              │ brand + FashionCLIP    │  │   │
│  │  │     ↓      │              │         ↓              │  │   │
│  │  │ [128, 64]  │              │ [256, 128, 64]         │  │   │
│  │  │     ↓      │              │         ↓              │  │   │
│  │  │  32-dim    │◄────────────▶│      32-dim            │  │   │
│  │  └────────────┘   Dot Product└────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU training)
- Docker & Docker Compose
- 16GB RAM minimum

### Installation

```bash
# Clone the repository
git clone https://github.com/freak-app/freak-ai.git
cd freak-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

### Running with Docker

```bash
# Start all services
docker-compose up -d

# Start with GPU training support
docker-compose --profile training up -d

# View logs
docker-compose logs -f api
```

### Local Development

```bash
# Generate embeddings
python scripts/generate_embeddings.py \
    --items data/raw/items.csv \
    --output data/embeddings/

# Train the model
python scripts/train.py \
    --config configs/config.yaml \
    --epochs 50

# Start the API server
uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000
```

## 📁 Project Structure

```
freak-ai/
├── configs/
│   └── config.yaml           # Main configuration
├── data/
│   ├── raw/                  # Raw CSV data
│   ├── processed/            # Processed datasets
│   └── embeddings/           # FashionCLIP embeddings
├── src/
│   ├── data/
│   │   ├── processor.py      # Data loading & preprocessing
│   │   ├── dataset.py        # PyTorch datasets
│   │   └── features.py       # Feature engineering
│   ├── embeddings/
│   │   └── fashion_clip.py   # FashionCLIP embedder
│   ├── models/
│   │   ├── two_tower.py      # Two-tower architecture
│   │   └── losses.py         # Loss functions
│   ├── training/
│   │   └── trainer.py        # Training loop with MLflow
│   ├── serving/
│   │   ├── api.py            # FastAPI endpoints
│   │   ├── cache.py          # Redis caching
│   │   └── retriever.py      # FAISS retrieval
│   ├── evaluation/
│   │   └── metrics.py        # Ranking metrics
│   └── utils/
│       ├── config.py         # Configuration management
│       └── logger.py         # Logging setup
├── scripts/
│   ├── train.py              # Training script
│   ├── generate_embeddings.py
│   └── init_db.sql           # Database schema
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🔧 Configuration

The system is configured via `configs/config.yaml`:

```yaml
# Key configuration options
embeddings:
  model_name: "patrickjohncyh/fashion-clip"
  embedding_dim: 512

two_tower:
  user_tower:
    hidden_layers: [128, 64]
  item_tower:
    hidden_layers: [256, 128, 64]
    use_visual_features: true
  final_embedding_dim: 32
  temperature: 0.1

training:
  batch_size: 1024
  epochs: 50
  learning_rate: 0.001
  num_negatives: 4
  hard_negative_ratio: 0.3
```

## 📡 API Endpoints

### Get Recommendations

```bash
# Personalized recommendations
curl -X GET "http://localhost:8000/recommendations/user/1001?top_k=20"

# Similar items
curl -X GET "http://localhost:8000/recommendations/similar/42?top_k=10"

# Session-based (cold start)
curl -X POST "http://localhost:8000/recommendations/session" \
  -H "Content-Type: application/json" \
  -d '{"viewed_items": [1, 5, 12], "top_k": 20}'
```

### Response Format

```json
{
  "user_id": "1001",
  "recommendations": [
    {
      "item_id": "42",
      "score": 0.89,
      "category": "Dresses",
      "brand": "Zara",
      "price": 250.00
    }
  ],
  "strategy": "personalized",
  "latency_ms": 45
}
```

## 📈 Model Training

### Data Format

**items.csv**:
```csv
item_id,category_id,brand_id,condition_id,size_id,price,image_urls
1,1,10,2,3,250.00,"[""https://...""]"
```

**user_events.csv**:
```csv
user_id,item_id,event,timestamp
1001,1,save,2024-01-20 10:30:00
1001,1,cart,2024-01-20 10:35:00
1001,1,order,2024-01-20 10:40:00
```

### Training Pipeline

```bash
# Full pipeline
python scripts/train.py \
    --config configs/config.yaml \
    --data-path data/raw \
    --output-dir checkpoints

# With MLflow tracking
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/train.py
```

### Metrics

The system evaluates on standard ranking metrics:
- **Precision@K**: Fraction of relevant items in top-K
- **Recall@K**: Fraction of relevant items retrieved
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Coverage**: Catalog coverage
- **Diversity**: Intra-list diversity

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## 🚢 Deployment

### Production Checklist

- [ ] Configure SSL certificates in `nginx/ssl/`
- [ ] Set strong passwords in `.env`
- [ ] Enable rate limiting
- [ ] Configure monitoring (Prometheus/Grafana)
- [ ] Set up log aggregation
- [ ] Configure backup for PostgreSQL

### Scaling

```bash
# Scale API servers
docker-compose up -d --scale api=3

# Use with load balancer
docker-compose --profile production up -d
```

## 📊 Monitoring

### MLflow Dashboard

Access at `http://localhost:5000` to view:
- Training metrics over time
- Model comparisons
- Hyperparameter tracking
- Artifact storage

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "model_loaded": true, "cache_connected": true}
```

## 🌍 MENA Localization

The system supports Arabic through CAMeL Tools:
- Arabic text search queries
- Arabizi (Franco-Arab) handling
- Egyptian and Gulf dialect support

```python
from src.utils.arabic import normalize_arabic_query
query = normalize_arabic_query("فستان زارا")
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FashionCLIP](https://github.com/patrickjohncyh/fashion-clip) for visual embeddings
- [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) for Arabic NLP
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- Anthropic Claude for development assistance

---

<p align="center">
  Made with ❤️ by the Freak Team
</p>
