"""
Configuration Management System.
Handles loading and validation of configuration files.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration."""
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32


@dataclass
class TradingConfig:
    """Trading configuration."""
    max_position_size: float = 0.02
    commission_rate: float = 0.001
    slippage_model: str = "linear"
    max_order_size: int = 10000


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_drawdown: float = 0.05
    daily_loss_limit: float = 0.03
    stop_loss: float = 0.02
    take_profit: float = 0.04


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    bootstrap_servers: str = "localhost:9092"
    topics: Dict[str, str] = field(default_factory=dict)
    consumer_group: str = "trading-platform"


@dataclass
class Config:
    """Main configuration class."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        
        # Load configurations
        self.trading = self._load_trading_config()
        self.models = self._load_models_config()
        self.kafka = self._load_kafka_config()
        self.risk = self._load_risk_config()
        self.backtesting = self._load_backtesting_config()
        self.api = self._load_api_config()
        self.monitoring = self._load_monitoring_config()
        self.logging = self._load_logging_config()
        self.tableau = self._load_tableau_config()
        
        # Load environment variables
        self._load_env_variables()
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            self.logger.warning(f"Config file not found: {filepath}. Using defaults.")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
                return config or {}
        except Exception as e:
            self.logger.error(f"Error loading config from {filepath}: {e}")
            return {}
    
    def _load_trading_config(self) -> Dict[str, Any]:
        """Load trading configuration."""
        config = self._load_yaml("trading_config.yaml")
        return config.get('trading', {
            'max_order_size': 10000,
            'commission_rate': 0.001,
            'slippage_model': 'linear',
            'risk': {
                'max_position_size': 0.02,
                'max_drawdown': 0.05,
                'stop_loss': 0.02,
                'take_profit': 0.04
            },
            'portfolio': {
                'initial_capital': 100000,
                'margin_requirement': 0.5
            },
            'strategies': {
                'lstm_momentum': {
                    'enabled': True,
                    'allocation': 0.4,
                    'signal_threshold': 0.6
                },
                'transformer_trend': {
                    'enabled': True,
                    'allocation': 0.4,
                    'signal_threshold': 0.65
                },
                'ensemble_ml': {
                    'enabled': True,
                    'allocation': 0.2,
                    'confidence_threshold': 0.7
                }
            }
        })
    
    def _load_models_config(self) -> Dict[str, Any]:
        """Load models configuration."""
        config = self._load_yaml("trading_config.yaml")
        return config.get('models', {
            'lstm': {
                'sequence_length': 60,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32
            },
            'transformer': {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'learning_rate': 0.0001
            }
        })
    
    def _load_kafka_config(self) -> Dict[str, Any]:
        """Load Kafka configuration."""
        config = self._load_yaml("kafka_config.yaml")
        return config.get('kafka', {
            'bootstrap_servers': 'localhost:9092',
            'topics': {
                'market_data': 'market-data',
                'trades': 'trades',
                'orders': 'orders',
                'signals': 'signals'
            },
            'consumer_group': 'trading-platform',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'buffer_size': 1000
        })
    
    def _load_risk_config(self) -> Dict[str, Any]:
        """Load risk management configuration."""
        trading_config = self._load_yaml("trading_config.yaml")
        return trading_config.get('trading', {}).get('risk', {
            'max_position_size': 0.02,
            'max_portfolio_heat': 0.06,
            'max_drawdown': 0.05,
            'daily_loss_limit': 0.03,
            'stop_loss': 0.02,
            'take_profit': 0.04
        })
    
    def _load_backtesting_config(self) -> Dict[str, Any]:
        """Load backtesting configuration."""
        config = self._load_yaml("trading_config.yaml")
        return config.get('backtesting', {
            'commission': 0.001,
            'slippage': 0.0005,
            'initial_capital': 100000,
            'max_position_size': 0.1,
            'monte_carlo': {
                'simulations': 10000,
                'confidence_intervals': [0.95, 0.99]
            },
            'models': {
                'lstm': {'sequence_length': 60, 'hidden_size': 128},
                'transformer': {'d_model': 256, 'nhead': 8}
            }
        })
    
    def _load_api_config(self) -> Dict[str, Any]:
        """Load API configuration."""
        config = self._load_yaml("trading_config.yaml")
        return config.get('api', {
            'host': '0.0.0.0',
            'port': 8000,
            'cors_origins': ['*'],
            'rate_limit': '100/minute'
        })
    
    def _load_monitoring_config(self) -> Dict[str, Any]:
        """Load monitoring configuration."""
        config = self._load_yaml("trading_config.yaml")
        return config.get('monitoring', {
            'prometheus_port': 8090,
            'metrics_interval': 10,
            'health_check_interval': 30
        })
    
    def _load_logging_config(self) -> Dict[str, Any]:
        """Load logging configuration."""
        config = self._load_yaml("trading_config.yaml")
        return config.get('logging', {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/trading_platform.log',
            'max_bytes': 10485760,
            'backup_count': 5
        })
    
    def _load_tableau_config(self) -> Dict[str, Any]:
        """Load Tableau configuration."""
        return {
            'server_url': os.getenv('TABLEAU_SERVER_URL', ''),
            'username': os.getenv('TABLEAU_USERNAME', ''),
            'password': os.getenv('TABLEAU_PASSWORD', ''),
            'site_id': os.getenv('TABLEAU_SITE_ID', ''),
            'dashboard_refresh_interval': 60
        }
    
    def _load_env_variables(self):
        """Load configuration from environment variables."""
        # Database
        if os.getenv('POSTGRES_HOST'):
            self.database = {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'database': os.getenv('POSTGRES_DB', 'trading_platform'),
                'user': os.getenv('POSTGRES_USER', 'trading_user'),
                'password': os.getenv('POSTGRES_PASSWORD', '')
            }
        
        # Override Kafka bootstrap servers if set
        if os.getenv('KAFKA_BOOTSTRAP_SERVERS'):
            self.kafka['bootstrap_servers'] = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
        
        # Override Redis if set
        if os.getenv('REDIS_HOST'):
            self.kafka['redis_host'] = os.getenv('REDIS_HOST')
            self.kafka['redis_port'] = int(os.getenv('REDIS_PORT', '6379'))
        
        # Trading parameters
        if os.getenv('INITIAL_CAPITAL'):
            self.trading['portfolio']['initial_capital'] = float(os.getenv('INITIAL_CAPITAL'))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'trading': self.trading,
            'models': self.models,
            'kafka': self.kafka,
            'risk': self.risk,
            'backtesting': self.backtesting,
            'api': self.api,
            'monitoring': self.monitoring,
            'logging': self.logging
        }
