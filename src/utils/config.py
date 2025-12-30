"""
Configuration Management for Freak AI
======================================

Handles loading, validation, and access to configuration parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv


@dataclass
class DataConfig:
    """Data-related configuration."""
    raw_items_path: str = "data/raw/items.csv"
    raw_events_path: str = "data/raw/user_events.csv"
    processed_dir: str = "data/processed"
    embeddings_dir: str = "data/embeddings"
    event_weights: Dict[str, float] = field(default_factory=lambda: {
        "save": 1.0,
        "cart": 3.0,
        "order": 5.0
    })
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 42


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "patrickjohncyh/fashion-clip"
    embedding_dim: int = 512
    batch_size: int = 32
    device: str = "cpu"
    cache_embeddings: bool = True
    image_size: int = 224


@dataclass
class TowerConfig:
    """Tower (user/item) configuration."""
    embedding_dim: int = 64
    hidden_layers: list = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    l2_reg: float = 1e-5
    use_visual_features: bool = False
    visual_embedding_dim: int = 512


@dataclass
class TwoTowerConfig:
    """Two-tower model configuration."""
    user_tower: TowerConfig = field(default_factory=TowerConfig)
    item_tower: TowerConfig = field(default_factory=lambda: TowerConfig(
        hidden_layers=[256, 128, 64],
        use_visual_features=True
    ))
    final_embedding_dim: int = 32
    temperature: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 1024
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = "adam"
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    min_lr: float = 1e-6
    num_negatives: int = 4
    hard_negative_ratio: float = 0.3
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True


@dataclass
class ServingConfig:
    """Serving/API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 3600
    default_top_k: int = 20
    fallback_strategy: str = "trending"
    cold_user_threshold: int = 5
    warm_user_threshold: int = 20


class Config:
    """
    Central configuration manager for Freak AI.
    
    Loads configuration from YAML file and environment variables,
    providing typed access to all configuration parameters.
    
    Example:
    --------
        config = Config("configs/config.yaml")
        print(config.data.raw_items_path)
        print(config.training.batch_size)
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Parameters
        ----------
        config_path : str or Path, optional
            Path to YAML configuration file.
        """
        # Load environment variables
        load_dotenv()
        
        # Default configuration
        self._config: Dict[str, Any] = {}
        
        # Load from file if provided
        if config_path:
            self._config = self._load_yaml(config_path)
        
        # Override with environment variables
        self._apply_env_overrides()
        
        # Initialize typed config objects
        self._init_typed_configs()
    
    def _load_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            "FREAK_DATA_PATH": ("data", "raw_items_path"),
            "FREAK_EVENTS_PATH": ("data", "raw_events_path"),
            "FREAK_EMBEDDING_DEVICE": ("embeddings", "device"),
            "FREAK_BATCH_SIZE": ("training", "batch_size"),
            "FREAK_LEARNING_RATE": ("training", "learning_rate"),
            "FREAK_REDIS_HOST": ("serving", "redis", "host"),
            "FREAK_REDIS_PORT": ("serving", "redis", "port"),
            "OPENAI_API_KEY": ("openai", "api_key"),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested(config_path, value)
    
    def _set_nested(self, path: tuple, value: Any):
        """Set a nested configuration value."""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Type conversion
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
        
        current[path[-1]] = value
    
    def _init_typed_configs(self):
        """Initialize typed configuration objects."""
        # Data config
        data_dict = self._config.get("data", {})
        self.data = DataConfig(
            raw_items_path=data_dict.get("raw_items_path", DataConfig.raw_items_path),
            raw_events_path=data_dict.get("raw_events_path", DataConfig.raw_events_path),
            processed_dir=data_dict.get("processed_dir", DataConfig.processed_dir),
            embeddings_dir=data_dict.get("embeddings_dir", DataConfig.embeddings_dir),
            event_weights=data_dict.get("event_weights", DataConfig().event_weights),
            test_size=data_dict.get("test_size", DataConfig.test_size),
            validation_size=data_dict.get("validation_size", DataConfig.validation_size),
            random_seed=data_dict.get("random_seed", DataConfig.random_seed),
        )
        
        # Embedding config
        emb_dict = self._config.get("embeddings", {})
        self.embeddings = EmbeddingConfig(
            model_name=emb_dict.get("model_name", EmbeddingConfig.model_name),
            embedding_dim=emb_dict.get("embedding_dim", EmbeddingConfig.embedding_dim),
            batch_size=emb_dict.get("batch_size", EmbeddingConfig.batch_size),
            device=emb_dict.get("device", EmbeddingConfig.device),
            cache_embeddings=emb_dict.get("cache_embeddings", EmbeddingConfig.cache_embeddings),
            image_size=emb_dict.get("image_size", EmbeddingConfig.image_size),
        )
        
        # Two-tower config
        tt_dict = self._config.get("two_tower", {})
        user_dict = tt_dict.get("user_tower", {})
        item_dict = tt_dict.get("item_tower", {})
        
        self.two_tower = TwoTowerConfig(
            user_tower=TowerConfig(
                embedding_dim=user_dict.get("embedding_dim", 64),
                hidden_layers=user_dict.get("hidden_layers", [128, 64]),
                dropout=user_dict.get("dropout", 0.2),
                l2_reg=user_dict.get("l2_reg", 1e-5),
            ),
            item_tower=TowerConfig(
                embedding_dim=item_dict.get("embedding_dim", 64),
                hidden_layers=item_dict.get("hidden_layers", [256, 128, 64]),
                dropout=item_dict.get("dropout", 0.2),
                l2_reg=item_dict.get("l2_reg", 1e-5),
                use_visual_features=item_dict.get("use_visual_features", True),
                visual_embedding_dim=item_dict.get("visual_embedding_dim", 512),
            ),
            final_embedding_dim=tt_dict.get("final_embedding_dim", 32),
            temperature=tt_dict.get("temperature", 0.1),
        )
        
        # Training config
        train_dict = self._config.get("training", {})
        self.training = TrainingConfig(
            batch_size=train_dict.get("batch_size", TrainingConfig.batch_size),
            epochs=train_dict.get("epochs", TrainingConfig.epochs),
            learning_rate=train_dict.get("learning_rate", TrainingConfig.learning_rate),
            optimizer=train_dict.get("optimizer", TrainingConfig.optimizer),
            early_stopping_patience=train_dict.get("early_stopping_patience", TrainingConfig.early_stopping_patience),
            reduce_lr_patience=train_dict.get("reduce_lr_patience", TrainingConfig.reduce_lr_patience),
            min_lr=train_dict.get("min_lr", TrainingConfig.min_lr),
            num_negatives=train_dict.get("num_negatives", TrainingConfig.num_negatives),
            hard_negative_ratio=train_dict.get("hard_negative_ratio", TrainingConfig.hard_negative_ratio),
            checkpoint_dir=train_dict.get("checkpoint_dir", TrainingConfig.checkpoint_dir),
            save_best_only=train_dict.get("save_best_only", TrainingConfig.save_best_only),
        )
        
        # Serving config
        serve_dict = self._config.get("serving", {})
        redis_dict = serve_dict.get("redis", {})
        self.serving = ServingConfig(
            host=serve_dict.get("host", ServingConfig.host),
            port=serve_dict.get("port", ServingConfig.port),
            redis_host=redis_dict.get("host", ServingConfig.redis_host),
            redis_port=redis_dict.get("port", ServingConfig.redis_port),
            redis_db=redis_dict.get("db", ServingConfig.redis_db),
            cache_ttl=redis_dict.get("cache_ttl", ServingConfig.cache_ttl),
            default_top_k=serve_dict.get("default_top_k", ServingConfig.default_top_k),
            fallback_strategy=serve_dict.get("fallback_strategy", ServingConfig.fallback_strategy),
            cold_user_threshold=serve_dict.get("cold_user_threshold", ServingConfig.cold_user_threshold),
            warm_user_threshold=serve_dict.get("warm_user_threshold", ServingConfig.warm_user_threshold),
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-notation key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a dictionary."""
        return self._config.copy()
    
    def __repr__(self) -> str:
        return f"Config(data={self.data}, training={self.training})"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Convenience function to load configuration.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file. If not provided, uses default.
    
    Returns
    -------
    Config
        Loaded configuration object.
    """
    if config_path is None:
        # Try to find config in standard locations
        default_paths = [
            "configs/config.yaml",
            "config.yaml",
            "../configs/config.yaml",
        ]
        for path in default_paths:
            if Path(path).exists():
                config_path = path
                break
    
    return Config(config_path)
