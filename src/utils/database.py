"""
Database Connection and ORM Management.
Handles PostgreSQL connections and data persistence.
"""

import logging
from typing import Optional, Any, Dict
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from datetime import datetime
import os


# Create base class for models
Base = declarative_base()


# Database Models
class Trade(Base):
    """Trade execution record."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), index=True)
    symbol = Column(String(20), index=True)
    side = Column(String(10))
    quantity = Column(Float)
    price = Column(Float)
    commission = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    strategy_id = Column(String(50))
    pnl = Column(Float, nullable=True)


class Position(Base):
    """Current position record."""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, index=True)
    quantity = Column(Float)
    avg_price = Column(Float)
    current_price = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)
    last_update = Column(DateTime, default=datetime.utcnow)


class PortfolioSnapshot(Base):
    """Daily portfolio snapshot."""
    __tablename__ = 'portfolio_snapshots'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    total_value = Column(Float)
    cash_balance = Column(Float)
    equity_value = Column(Float)
    total_pnl = Column(Float)
    daily_pnl = Column(Float)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)


class ModelPrediction(Base):
    """ML model prediction record."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    model_name = Column(String(50))
    prediction = Column(Float)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    features = Column(JSON, nullable=True)


class SentimentRecord(Base):
    """Sentiment analysis record."""
    __tablename__ = 'sentiment'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True, nullable=True)
    source = Column(String(50))
    text = Column(Text)
    sentiment = Column(String(20))
    score = Column(Float)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    inference_time_ms = Column(Float, nullable=True)


class NewsArticle(Base):
    """News article record."""
    __tablename__ = 'news_articles'
    
    id = Column(Integer, primary_key=True)
    article_id = Column(String(100), unique=True, index=True)
    title = Column(String(500))
    content = Column(Text)
    source = Column(String(50))
    symbols = Column(JSON)
    sentiment_score = Column(Float, nullable=True)
    impact_score = Column(Float, nullable=True)
    published_at = Column(DateTime, index=True)
    processed_at = Column(DateTime, default=datetime.utcnow)


class PerformanceMetric(Base):
    """Performance metrics record."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(50), index=True)
    metric_value = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    context = Column(JSON, nullable=True)


class RiskAlert(Base):
    """Risk management alert."""
    __tablename__ = 'risk_alerts'
    
    id = Column(Integer, primary_key=True)
    alert_type = Column(String(50))
    severity = Column(String(20))
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    resolved = Column(Boolean, default=False)
    resolution_time = Column(DateTime, nullable=True)


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        
        # Get database configuration
        if config is None:
            config = self._get_config_from_env()
        
        # Create connection string
        self.connection_string = self._build_connection_string(config)
        
        # Create engine
        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self.logger.info("Database manager initialized")
    
    def _get_config_from_env(self) -> Dict[str, Any]:
        """Get database configuration from environment variables."""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'trading_platform'),
            'user': os.getenv('POSTGRES_USER', 'trading_user'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
    
    def _build_connection_string(self, config: Dict[str, Any]) -> str:
        """Build PostgreSQL connection string."""
        return (
            f"postgresql://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['database']}"
        )
    
    def initialize_database(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def save_trade(self, trade_data: Dict[str, Any]):
        """Save a trade to the database."""
        with self.get_session() as session:
            trade = Trade(**trade_data)
            session.add(trade)
    
    def save_position(self, position_data: Dict[str, Any]):
        """Save or update a position."""
        with self.get_session() as session:
            # Check if position exists
            existing = session.query(Position).filter_by(
                symbol=position_data['symbol']
            ).first()
            
            if existing:
                # Update existing position
                for key, value in position_data.items():
                    setattr(existing, key, value)
                existing.last_update = datetime.utcnow()
            else:
                # Create new position
                position = Position(**position_data)
                session.add(position)
    
    def save_portfolio_snapshot(self, snapshot_data: Dict[str, Any]):
        """Save a portfolio snapshot."""
        with self.get_session() as session:
            snapshot = PortfolioSnapshot(**snapshot_data)
            session.add(snapshot)
    
    def save_prediction(self, prediction_data: Dict[str, Any]):
        """Save a model prediction."""
        with self.get_session() as session:
            prediction = ModelPrediction(**prediction_data)
            session.add(prediction)
    
    def save_sentiment(self, sentiment_data: Dict[str, Any]):
        """Save sentiment analysis result."""
        with self.get_session() as session:
            sentiment = SentimentRecord(**sentiment_data)
            session.add(sentiment)
    
    def save_news_article(self, article_data: Dict[str, Any]):
        """Save a news article."""
        with self.get_session() as session:
            # Check if article already exists
            existing = session.query(NewsArticle).filter_by(
                article_id=article_data['article_id']
            ).first()
            
            if not existing:
                article = NewsArticle(**article_data)
                session.add(article)
    
    def save_performance_metric(self, metric_name: str, metric_value: float, 
                               context: Optional[Dict[str, Any]] = None):
        """Save a performance metric."""
        with self.get_session() as session:
            metric = PerformanceMetric(
                metric_name=metric_name,
                metric_value=metric_value,
                context=context
            )
            session.add(metric)
    
    def save_risk_alert(self, alert_data: Dict[str, Any]):
        """Save a risk alert."""
        with self.get_session() as session:
            alert = RiskAlert(**alert_data)
            session.add(alert)
    
    def get_recent_trades(self, symbol: Optional[str] = None, limit: int = 100):
        """Get recent trades."""
        with self.get_session() as session:
            query = session.query(Trade).order_by(Trade.timestamp.desc())
            
            if symbol:
                query = query.filter_by(symbol=symbol)
            
            return query.limit(limit).all()
    
    def get_current_positions(self):
        """Get all current positions."""
        with self.get_session() as session:
            return session.query(Position).filter(Position.quantity != 0).all()
    
    def get_portfolio_history(self, days: int = 30):
        """Get portfolio history."""
        with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            return session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.timestamp >= cutoff_date
            ).order_by(PortfolioSnapshot.timestamp).all()
    
    def get_sentiment_history(self, symbol: str, hours: int = 24):
        """Get sentiment history for a symbol."""
        with self.get_session() as session:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            return session.query(SentimentRecord).filter(
                SentimentRecord.symbol == symbol,
                SentimentRecord.timestamp >= cutoff_time
            ).order_by(SentimentRecord.timestamp).all()
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data from database."""
        with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old predictions
            session.query(ModelPrediction).filter(
                ModelPrediction.timestamp < cutoff_date
            ).delete()
            
            # Delete old sentiment records
            session.query(SentimentRecord).filter(
                SentimentRecord.timestamp < cutoff_date
            ).delete()
            
            # Delete old news articles
            session.query(NewsArticle).filter(
                NewsArticle.processed_at < cutoff_date
            ).delete()
            
            self.logger.info(f"Cleaned up data older than {days} days")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.initialize_database()
    
    return _db_manager


def get_db_session() -> Session:
    """Get a database session."""
    manager = get_db_manager()
    return manager.SessionLocal()
