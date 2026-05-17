"""
Advanced Data Preprocessor for Financial Time Series.
Handles feature engineering, technical indicators, and data normalization.
"""

import numpy as np
import pandas as pd
import ta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DataPreprocessor:
    """Advanced financial data preprocessing and feature engineering."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive feature set for ML models."""
        
        self.logger.info("Starting feature engineering...")
        df = data.copy()
        
        # Basic price features
        df = self._add_price_features(df)
        
        # Technical indicators
        df = self.add_technical_indicators(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Market microstructure features
        df = self._add_microstructure_features(df)
        
        # Clean data
        df = self._clean_data(df)
        
        self.logger.info(f"Feature engineering complete. Features: {df.shape[1]}")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features."""
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price gaps
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # High-Low spread
        df['hl_pct'] = (df['high'] - df['low']) / df['close']
        
        # Price position within range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators."""
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # MACD
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_diff'] = ta.trend.macd_diff(df['close'])
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'])
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_30'] = ta.momentum.rsi(df['close'], window=30)
        
        # Bollinger Bands
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['atr_pct'] = df['atr'] / df['close']
        
        # Stochastic Oscillator
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # Commodity Channel Index
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # Money Flow Index
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        # On-Balance Volume
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        
        # Chaikin A/D Line
        df['ad'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Parabolic SAR
        df['psar'] = ta.trend.psar(df['high'], df['low'], df['close'])
        
        # Aroon
        df['aroon_up'] = ta.trend.aroon_up(df['high'], df['low'])
        df['aroon_down'] = ta.trend.aroon_down(df['high'], df['low'])
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        
        # Rolling statistics
        for window in [5, 10, 20]:
            # Volatility
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            
            # Skewness
            df[f'skewness_{window}'] = df['returns'].rolling(window=window).skew()
            
            # Kurtosis
            df[f'kurtosis_{window}'] = df['returns'].rolling(window=window).kurt()
            
            # Z-score
            mean_ret = df['returns'].rolling(window=window).mean()
            std_ret = df['returns'].rolling(window=window).std()
            df[f'zscore_{window}'] = (df['returns'] - mean_ret) / std_ret
        
        # Momentum features
        for lag in [1, 3, 5, 10]:
            df[f'momentum_{lag}'] = df['close'] / df['close'].shift(lag) - 1
            df[f'return_{lag}d'] = df['close'].pct_change(lag)
        
        # Mean reversion features
        df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
        df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Market session indicators
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        
        # Spread proxies
        df['bid_ask_spread'] = (df['high'] - df['low']) / df['close']  # Proxy
        
        # Price impact
        df['price_impact'] = abs(df['returns']) / (df['volume'] / df['volume'].rolling(20).mean())
        
        # Amihud illiquidity measure
        df['amihud'] = abs(df['returns']) / (df['volume'] * df['close'])
        
        # Roll measure (effective spread estimator)
        df['roll_measure'] = 2 * np.sqrt(-df['returns'].rolling(2).cov().iloc[-1, 0]) if len(df) > 2 else 0
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset."""
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Forward fill then backward fill
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        # Fill remaining NaN with 0
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Remove outliers using IQR method
        for col in ['returns', 'volume']:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=Q1, upper=Q3)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Normalize features using specified method."""
        
        feature_columns = [col for col in df.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        df_normalized = df.copy()
        df_normalized[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        # Store scaler for inverse transformation
        self.scalers[method] = scaler
        
        return df_normalized
    
    def create_sequences(self, data: pd.DataFrame, sequence_length: int, 
                        target_column: str = 'returns') -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        
        feature_columns = [col for col in data.columns if col != target_column]
        
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            # Features: sequence of past observations
            X.append(data[feature_columns].iloc[i-sequence_length:i].values)
            
            # Target: next period return (or classification)
            y.append(data[target_column].iloc[i])
        
        return np.array(X), np.array(y)
    
    def create_labels(self, returns: pd.Series, method: str = 'classification',
                     threshold: float = 0.001) -> pd.Series:
        """Create labels for supervised learning."""
        
        if method == 'classification':
            # Binary classification: up/down
            labels = (returns > threshold).astype(int)
        
        elif method == 'multi_class':
            # Multi-class: strong down, down, neutral, up, strong up
            labels = pd.cut(returns, 
                          bins=[-np.inf, -0.01, -threshold, threshold, 0.01, np.inf],
                          labels=[0, 1, 2, 3, 4])
        
        elif method == 'regression':
            # Direct regression on returns
            labels = returns
        
        else:
            raise ValueError(f"Unknown labeling method: {method}")
        
        return labels
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features."""
        
        # Volatility regime
        vol_20 = df['returns'].rolling(20).std()
        vol_threshold = vol_20.quantile(0.7)
        df['high_vol_regime'] = (vol_20 > vol_threshold).astype(int)
        
        # Trend regime
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        df['uptrend_regime'] = (sma_20 > sma_50).astype(int)
        
        # VIX regime (if available)
        if 'vix' in df.columns:
            df['fear_regime'] = (df['vix'] > 20).astype(int)
        
        return df
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        
        metrics = {}
        
        # Basic metrics
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # VaR and CVaR
        metrics['var_95'] = returns.quantile(0.05)
        metrics['var_99'] = returns.quantile(0.01)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = returns.mean() * 252 - 0.02
        metrics['sharpe_ratio'] = excess_returns / metrics['volatility']
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = excess_returns / downside_vol
        else:
            metrics['sortino_ratio'] = np.inf
        
        return metrics
