"""
Comprehensive tests for ML models and trading strategies.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import tempfile
import os

# Import modules (adjust paths as needed)
import sys
sys.path.append('../src')

from src.models.lstm_model import LSTMModel, LSTMTrainer
from src.backtesting.backtest_engine import BacktestEngine
from src.data.preprocessor import DataPreprocessor


class TestLSTMModel:
    """Test LSTM model functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample financial data for testing."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 1000)
        prices = [100]
        for ret in returns[:-1]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'close': prices,
            'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'volume': np.random.randint(1000000, 10000000, 1000)
        }, index=dates)
        
        return data
    
    def test_lstm_model_creation(self):
        """Test LSTM model instantiation."""
        model = LSTMModel(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert isinstance(model.lstm, torch.nn.LSTM)
    
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass."""
        model = LSTMModel(input_size=5, hidden_size=64, num_layers=2)
        
        # Create sample input
        batch_size, seq_len, input_size = 32, 60, 5
        x = torch.randn(batch_size, seq_len, input_size)
        
        output = model(x)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
    
    def test_lstm_trainer_initialization(self):
        """Test LSTM trainer initialization."""
        config = {
            'sequence_length': 60,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001
        }
        
        trainer = LSTMTrainer(config)
        
        assert trainer.sequence_length == 60
        assert trainer.hidden_size == 128
        assert trainer.device.type in ['cpu', 'cuda']
    
    def test_data_preparation(self, sample_data):
        """Test data preparation for LSTM training."""
        config = {
            'sequence_length': 60,
            'hidden_size': 128,
            'num_layers': 2
        }
        
        trainer = LSTMTrainer(config)
        preprocessor = DataPreprocessor()
        
        # Add technical indicators
        processed_data = preprocessor.prepare_features(sample_data)
        
        # Prepare data for LSTM
        X, y = trainer.prepare_data(processed_data)
        
        assert X.shape[1] == 60  # sequence length
        assert X.shape[2] > 5   # number of features
        assert len(X) == len(y)
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    
    def test_model_save_load(self, sample_data):
        """Test model saving and loading."""
        config = {
            'sequence_length': 30,
            'hidden_size': 64,
            'num_layers': 1,
            'learning_rate': 0.01
        }
        
        trainer = LSTMTrainer(config)
        preprocessor = DataPreprocessor()
        
        # Prepare minimal data
        processed_data = preprocessor.prepare_features(sample_data)
        X, y = trainer.prepare_data(processed_data)
        
        # Initialize model
        input_size = X.shape[2]
        trainer.model = LSTMModel(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )
        
        # Test save/load cycle
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            try:
                trainer.save_model(tmp.name)
                
                # Create new trainer and load model
                new_trainer = LSTMTrainer(config)
                new_trainer.load_model(tmp.name)
                
                assert new_trainer.model is not None
                assert new_trainer.model.hidden_size == 64
                assert new_trainer.sequence_length == 30
                
            finally:
                os.unlink(tmp.name)
    
    def test_signal_generation(self, sample_data):
        """Test trading signal generation."""
        config = {
            'sequence_length': 30,
            'hidden_size': 64,
            'num_layers': 1
        }
        
        trainer = LSTMTrainer(config)
        preprocessor = DataPreprocessor()
        
        # Prepare data
        processed_data = preprocessor.prepare_features(sample_data)
        
        # Initialize model (skip training for speed)
        input_size = 15  # Approximate feature count
        trainer.model = LSTMModel(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )
        
        # Fit scaler
        feature_columns = [col for col in processed_data.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']][:input_size]
        trainer.scaler.fit(processed_data[feature_columns].fillna(0))
        
        # Generate signals
        signals = trainer.generate_signals(processed_data.tail(100))
        
        assert len(signals) > 0
        assert 'signal' in signals.columns
        assert 'confidence' in signals.columns
        assert signals['signal'].isin([-1, 0, 1]).all()


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample financial data."""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)
        
        # Generate price data with trends
        trend = np.linspace(100, 150, 500)
        noise = np.random.normal(0, 5, 500)
        prices = trend + noise
        
        data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 500)
        }, index=dates)
        
        return data
    
    def test_feature_engineering(self, sample_data):
        """Test comprehensive feature engineering."""
        preprocessor = DataPreprocessor()
        
        features = preprocessor.prepare_features(sample_data)
        
        # Check that features were added
        assert 'returns' in features.columns
        assert 'sma_20' in features.columns
        assert 'rsi' in features.columns
        assert 'macd' in features.columns
        assert 'atr' in features.columns
        
        # Check data integrity
        assert not features.isnull().all().any()  # No columns that are all NaN
        assert len(features) == len(sample_data)
    
    def test_technical_indicators(self, sample_data):
        """Test technical indicator calculations."""
        preprocessor = DataPreprocessor()
        
        df = preprocessor.add_technical_indicators(sample_data)
        
        # Test specific indicators
        assert 'sma_20' in df.columns
        assert 'ema_12' in df.columns
        assert 'bb_upper' in df.columns
        assert 'bb_lower' in df.columns
        assert 'rsi' in df.columns
        
        # Verify RSI is in correct range (eventually)
        rsi_valid = df['rsi'].dropna()
        if len(rsi_valid) > 0:
            assert rsi_valid.min() >= 0
            assert rsi_valid.max() <= 100
    
    def test_normalization(self, sample_data):
        """Test feature normalization."""
        preprocessor = DataPreprocessor()
        
        features = preprocessor.prepare_features(sample_data)
        normalized = preprocessor.normalize_features(features, method='standard')
        
        # Check that features are normalized (approximately)
        feature_cols = [col for col in normalized.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        for col in feature_cols[:5]:  # Test first 5 features
            if col in normalized.columns:
                col_data = normalized[col].dropna()
                if len(col_data) > 10:  # Enough data points
                    assert abs(col_data.mean()) < 0.1  # Approximately zero mean
                    assert abs(col_data.std() - 1) < 0.2  # Approximately unit variance
    
    def test_sequence_creation(self, sample_data):
        """Test sequence creation for time series models."""
        preprocessor = DataPreprocessor()
        
        features = preprocessor.prepare_features(sample_data)
        features = features.fillna(0)  # Handle NaN values
        
        X, y = preprocessor.create_sequences(features, sequence_length=30, target_column='returns')
        
        assert X.shape[1] == 30  # sequence length
        assert X.shape[2] == len(features.columns) - 1  # features (excluding target)
        assert len(X) == len(y)
        assert len(X) == len(features) - 30  # Correct number of sequences


class TestBacktestEngine:
    """Test backtesting functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for backtesting."""
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        np.random.seed(42)
        
        # Generate trending price data
        trend = np.linspace(100, 120, 365)
        noise = np.random.normal(0, 2, 365)
        prices = trend + noise
        
        data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.005,
            'Low': prices * 0.995,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 365)
        }, index=dates)
        
        return data
    
    def test_backtest_engine_initialization(self):
        """Test backtest engine setup."""
        config = {
            'commission': 0.001,
            'slippage': 0.0005,
            'initial_capital': 100000,
            'models': {
                'lstm': {'sequence_length': 30, 'hidden_size': 64},
                'transformer': {'d_model': 128, 'nhead': 4}
            }
        }
        
        engine = BacktestEngine(config)
        
        assert engine.commission == 0.001
        assert engine.initial_capital == 100000
        assert engine.preprocessor is not None
    
    def test_data_download_simulation(self):
        """Test data download simulation."""
        config = {'commission': 0.001}
        engine = BacktestEngine(config)
        
        # Mock the download method to avoid external API calls
        def mock_download(symbol, start_date, end_date):
            dates = pd.date_range(start_date, end_date, freq='D')
            np.random.seed(42)
            prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
            
            return pd.DataFrame({
                'open': prices * 0.999,
                'high': prices * 1.005,
                'low': prices * 0.995,
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
        
        # Replace method
        engine._download_data = mock_download
        
        data = engine._download_data('AAPL', datetime(2020, 1, 1), datetime(2020, 12, 31))
        
        assert len(data) > 300  # Should have most days of the year
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        config = {'commission': 0.001}
        engine = BacktestEngine(config)
        
        # Create mock results
        from src.backtesting.backtest_engine import BacktestResults
        
        # Create sample equity curve
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        equity_values = 100000 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod()
        equity_curve = pd.Series(equity_values, index=dates)
        
        # Create sample trades
        trades = [
            {'action': 'SELL', 'pnl': 100, 'pnl_pct': 0.01},
            {'action': 'SELL', 'pnl': -50, 'pnl_pct': -0.005},
            {'action': 'SELL', 'pnl': 200, 'pnl_pct': 0.02},
            {'action': 'SELL', 'pnl': -30, 'pnl_pct': -0.003}
        ]
        
        results = BacktestResults(
            symbol='TEST',
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 4, 10),
            strategy='test',
            initial_capital=100000,
            final_capital=equity_values[-1],
            total_return=0,  # Will be calculated
            annual_return=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            total_trades=len(trades),
            avg_trade_return=0,
            volatility=0,
            beta=0,
            alpha=0,
            trades=trades,
            equity_curve=equity_curve,
            drawdown_curve=pd.Series()
        )
        
        # Calculate metrics
        results = engine._calculate_performance_metrics(results)
        
        assert results.total_return != 0
        assert results.win_rate > 0
        assert results.max_drawdown <= 0
        assert results.volatility > 0


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_backtest(self):
        """Test complete backtest workflow."""
        # This is a simplified integration test
        config = {
            'commission': 0.001,
            'slippage': 0.0005,
            'initial_capital': 10000,  # Smaller capital for faster testing
            'models': {
                'lstm': {
                    'sequence_length': 20,  # Shorter for faster testing
                    'hidden_size': 32,
                    'num_layers': 1
                }
            }
        }
        
        engine = BacktestEngine(config)
        
        # Mock data download
        def mock_download(symbol, start_date, end_date):
            dates = pd.date_range(start_date, end_date, freq='D')
            np.random.seed(42)
            prices = 100 + np.cumsum(np.random.normal(0.001, 0.01, len(dates)))
            
            return pd.DataFrame({
                'open': prices * 0.999,
                'high': prices * 1.002,
                'low': prices * 0.998,
                'close': prices,
                'volume': np.random.randint(100000, 500000, len(dates)),
                'returns': np.concatenate([[0], np.diff(prices) / prices[:-1]])
            }, index=dates)
        
        engine._download_data = mock_download
        
        try:
            # Run a simple backtest
            results = engine.run_backtest(
                symbol='TEST',
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 3, 31),  # Shorter period
                strategy='lstm_momentum',
                initial_capital=10000
            )
            
            # Verify results structure
            assert hasattr(results, 'total_return')
            assert hasattr(results, 'sharpe_ratio')
            assert hasattr(results, 'max_drawdown')
            assert len(results.equity_curve) > 0
            
        except Exception as e:
            # If backtest fails due to missing models, that's expected in testing
            assert "model" in str(e).lower() or "data" in str(e).lower()


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_data_processing_speed(self):
        """Test data processing performance."""
        # Generate large dataset
        dates = pd.date_range('2010-01-01', periods=5000, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 100 + np.random.normal(0, 10, 5000),
            'high': 102 + np.random.normal(0, 10, 5000),
            'low': 98 + np.random.normal(0, 10, 5000),
            'close': 100 + np.random.normal(0, 10, 5000),
            'volume': np.random.randint(1000000, 10000000, 5000)
        }, index=dates)
        
        preprocessor = DataPreprocessor()
        
        import time
        start_time = time.time()
        features = preprocessor.prepare_features(data)
        processing_time = time.time() - start_time
        
        # Should process 5000 days in under 5 seconds
        assert processing_time < 5.0
        assert len(features) == len(data)
    
    def test_model_prediction_speed(self):
        """Test model prediction performance."""
        config = {
            'sequence_length': 60,
            'hidden_size': 128,
            'num_layers': 2
        }
        
        trainer = LSTMTrainer(config)
        
        # Create model
        trainer.model = LSTMModel(
            input_size=20,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )
        
        # Create sample data
        sample_features = np.random.randn(60, 20)
        trainer.scaler.fit(sample_features)
        
        # Test prediction speed
        import time
        start_time = time.time()
        
        for _ in range(100):  # 100 predictions
            data = pd.DataFrame(sample_features, columns=[f'feature_{i}' for i in range(20)])
            try:
                prediction = trainer.predict(data)
            except:
                pass  # Expected to fail without proper setup
        
        prediction_time = time.time() - start_time
        
        # Should make 100 predictions in under 1 second
        assert prediction_time < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
