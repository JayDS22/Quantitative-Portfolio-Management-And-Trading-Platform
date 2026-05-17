"""
High-Frequency Trading Engine with Algorithmic Execution.
Implements risk management, order management, and portfolio tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from ..data.data_pipeline import DataPipeline
from ..models.lstm_model import LSTMTrainer
from ..models.transformer_model import TransformerTrainer
from .portfolio import PortfolioManager
from .risk_manager import RiskManager
from .strategy import StrategyManager
from ..utils.database import get_db_session


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Trading order structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    strategy_id: str = ""
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Trade:
    """Executed trade structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    commission: float = 0.0
    strategy_id: str = ""


@dataclass
class Position:
    """Trading position structure."""
    symbol: str = ""
    quantity: float = 0.0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)


class TradingEngine:
    """High-performance algorithmic trading engine."""
    
    def __init__(self, config: Dict[str, Any], data_pipeline: DataPipeline):
        self.config = config
        self.data_pipeline = data_pipeline
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.portfolio_manager = PortfolioManager(config.get('portfolio', {}))
        self.risk_manager = RiskManager(config.get('risk', {}))
        self.strategy_manager = StrategyManager(config.get('strategies', {}))
        
        # Model ensembles
        self.lstm_model = LSTMTrainer(config.get('models', {}).get('lstm', {}))
        self.transformer_model = TransformerTrainer(config.get('models', {}).get('transformer', {}))
        
        # Order management
        self.pending_orders: Dict[str, Order] = {}
        self.executed_trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Execution settings
        self.max_order_size = config.get('max_order_size', 10000)
        self.commission_rate = config.get('commission_rate', 0.001)
        self.slippage_model = config.get('slippage_model', 'linear')
        
        # State management
        self.is_running = False
        self.trading_enabled = True
        
        # Subscribe to data pipeline events
        self.data_pipeline.subscribe('market_tick', self._on_market_tick)
        self.data_pipeline.subscribe('ohlc_update', self._on_ohlc_update)
        
    async def start(self):
        """Start the trading engine."""
        self.logger.info("Starting trading engine...")
        self.is_running = True
        
        # Load models
        await self._load_models()
        
        # Initialize portfolio
        await self.portfolio_manager.initialize()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._order_management_loop()),
            asyncio.create_task(self._strategy_execution_loop()),
            asyncio.create_task(self._risk_monitoring_loop()),
            asyncio.create_task(self._performance_tracking_loop())
        ]
        
        self.logger.info("Trading engine started successfully")
        
        # Wait for tasks
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Trading engine tasks cancelled")
    
    async def stop(self):
        """Stop the trading engine."""
        self.logger.info("Stopping trading engine...")
        self.is_running = False
        
        # Cancel all pending orders
        for order in self.pending_orders.values():
            await self._cancel_order(order.id)
        
        # Save portfolio state
        await self.portfolio_manager.save_state()
        
        self.logger.info("Trading engine stopped")
    
    async def _load_models(self):
        """Load trained ML models."""
        try:
            self.lstm_model.load_model("data/models/lstm_best.pth")
            self.transformer_model.load_model("data/models/transformer_best.pth")
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    async def _on_market_tick(self, tick_data):
        """Handle incoming market tick data."""
        try:
            # Update positions with current market prices
            symbol = tick_data.symbol
            current_price = tick_data.last
            
            if symbol in self.positions:
                position = self.positions[symbol]
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                position.last_update = datetime.utcnow()
            
            # Check for order fills
            await self._check_order_fills(symbol, current_price)
            
        except Exception as e:
            self.logger.error(f"Error handling market tick: {e}")
    
    async def _on_ohlc_update(self, ohlc_data):
        """Handle OHLC data updates for strategy execution."""
        try:
            symbol = ohlc_data['symbol']
            timeframe = ohlc_data['timeframe']
            
            # Only process 1-minute bars for high-frequency strategies
            if timeframe == '1min':
                await self._generate_trading_signals(symbol, ohlc_data['ohlc'])
                
        except Exception as e:
            self.logger.error(f"Error handling OHLC update: {e}")
    
    async def _generate_trading_signals(self, symbol: str, ohlc_data: Dict[str, Any]):
        """Generate trading signals using ML models."""
        try:
            # Get historical data for prediction
            historical_data = await self._get_historical_data(symbol, periods=100)
            if historical_data is None or len(historical_data) < 60:
                return
            
            # Generate signals from models
            lstm_signal = await self._get_lstm_signal(symbol, historical_data)
            transformer_signal = await self._get_transformer_signal(symbol, historical_data)
            
            # Ensemble prediction
            ensemble_signal = self._combine_signals(lstm_signal, transformer_signal)
            
            # Generate trading decision
            if abs(ensemble_signal) > 0.6:  # High confidence threshold
                await self._execute_strategy_signal(symbol, ensemble_signal, ohlc_data)
                
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
    
    async def _get_lstm_signal(self, symbol: str, data: pd.DataFrame) -> float:
        """Get signal from LSTM model."""
        try:
            prediction = self.lstm_model.predict(data, return_probabilities=True)[0]
            # Convert probability to signal (-1 to 1)
            signal = (prediction - 0.5) * 2
            return float(signal)
        except Exception as e:
            self.logger.error(f"Error getting LSTM signal: {e}")
            return 0.0
    
    async def _get_transformer_signal(self, symbol: str, data: pd.DataFrame) -> float:
        """Get signal from Transformer model."""
        try:
            prediction = self.transformer_model.predict(data, return_probabilities=True)[0]
            # Convert probability to signal (-1 to 1)
            signal = (prediction - 0.5) * 2
            return float(signal)
        except Exception as e:
            self.logger.error(f"Error getting Transformer signal: {e}")
            return 0.0
    
    def _combine_signals(self, lstm_signal: float, transformer_signal: float, 
                        lstm_weight: float = 0.6, transformer_weight: float = 0.4) -> float:
        """Combine signals from multiple models."""
        ensemble_signal = (lstm_signal * lstm_weight + 
                          transformer_signal * transformer_weight)
        
        # Apply additional filters
        signal_strength = abs(ensemble_signal)
        if signal_strength < 0.3:
            return 0.0  # No signal
        
        return ensemble_signal
    
    async def _execute_strategy_signal(self, symbol: str, signal: float, market_data: Dict[str, Any]):
        """Execute trading strategy based on signal."""
        try:
            # Check risk limits
            if not await self.risk_manager.check_trading_allowed(symbol):
                return
            
            current_price = float(market_data['close'])
            
            # Calculate position size based on signal strength and risk management
            base_position_size = await self.portfolio_manager.get_max_position_size(symbol)
            signal_strength = abs(signal)
            position_size = base_position_size * signal_strength
            
            # Apply risk management
            position_size = await self.risk_manager.adjust_position_size(
                symbol, position_size, current_price
            )
            
            if position_size < 10:  # Minimum trade size
                return
            
            # Determine order side
            side = OrderSide.BUY if signal > 0 else OrderSide.SELL
            
            # Check current position
            current_position = self.positions.get(symbol, Position(symbol=symbol))
            
            # Calculate target position
            if side == OrderSide.BUY:
                target_quantity = current_position.quantity + position_size
            else:
                target_quantity = current_position.quantity - position_size
            
            # Place order
            order = await self._place_order(
                symbol=symbol,
                side=side,
                quantity=position_size,
                order_type=OrderType.MARKET,
                strategy_id="ensemble_ml"
            )
            
            if order:
                self.logger.info(
                    f"Placed {side.value} order for {symbol}: "
                    f"{position_size:.2f} @ {current_price:.4f} "
                    f"(Signal: {signal:.3f})"
                )
            
        except Exception as e:
            self.logger.error(f"Error executing strategy signal: {e}")
    
    async def _place_order(self, 
                          symbol: str,
                          side: OrderSide,
                          quantity: float,
                          order_type: OrderType = OrderType.MARKET,
                          price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          strategy_id: str = "") -> Optional[Order]:
        """Place a trading order."""
        try:
            # Validate order
            if quantity <= 0:
                return None
            
            if quantity > self.max_order_size:
                self.logger.warning(f"Order size {quantity} exceeds maximum {self.max_order_size}")
                return None
            
            # Create order
            order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                strategy_id=strategy_id
            )
            
            # Add to pending orders
            self.pending_orders[order.id] = order
            
            # For market orders, simulate immediate execution
            if order_type == OrderType.MARKET:
                await self._simulate_order_execution(order)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def _simulate_order_execution(self, order: Order):
        """Simulate order execution (replace with actual broker integration)."""
        try:
            # Get current market price
            tick_data = await self.data_pipeline.get_latest_tick(order.symbol)
            if not tick_data:
                order.status = OrderStatus.REJECTED
                return
            
            # Calculate execution price with slippage
            if order.side == OrderSide.BUY:
                execution_price = float(tick_data['ask'])
            else:
                execution_price = float(tick_data['bid'])
            
            # Apply slippage model
            slippage = self._calculate_slippage(order.quantity, execution_price)
            if order.side == OrderSide.BUY:
                execution_price += slippage
            else:
                execution_price -= slippage
            
            # Calculate commission
            commission = order.quantity * execution_price * self.commission_rate
            
            # Create trade
            trade = Trade(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                commission=commission,
                strategy_id=order.strategy_id
            )
            
            # Update order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = execution_price
            order.commission = commission
            order.slippage = slippage
            
            # Record trade
            self.executed_trades.append(trade)
            
            # Update position
            await self._update_position(trade)
            
            # Update portfolio
            await self.portfolio_manager.process_trade(trade)
            
            # Remove from pending orders
            if order.id in self.pending_orders:
                del self.pending_orders[order.id]
            
            # Update performance metrics
            self.total_trades += 1
            
            self.logger.info(
                f"Order executed: {order.symbol} {order.side.value} "
                f"{order.quantity:.2f} @ {execution_price:.4f} "
                f"(Slippage: {slippage:.4f}, Commission: {commission:.2f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error executing order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
    
    def _calculate_slippage(self, quantity: float, price: float) -> float:
        """Calculate slippage based on order size and market conditions."""
        if self.slippage_model == 'linear':
            # Simple linear slippage model
            slippage_rate = min(0.001 * (quantity / 1000), 0.01)  # Max 1% slippage
            return price * slippage_rate
        elif self.slippage_model == 'sqrt':
            # Square root model
            slippage_rate = 0.001 * np.sqrt(quantity / 1000)
            return price * slippage_rate
        else:
            return 0.0
    
    async def _update_position(self, trade: Trade):
        """Update position based on executed trade."""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        
        if trade.side == OrderSide.BUY:
            # Calculate new average price
            total_cost = (position.quantity * position.avg_price + 
                         trade.quantity * trade.price)
            total_quantity = position.quantity + trade.quantity
            
            if total_quantity > 0:
                position.avg_price = total_cost / total_quantity
            position.quantity = total_quantity
            
        else:  # SELL
            # Update realized P&L
            if position.quantity > 0:
                pnl_per_share = trade.price - position.avg_price
                realized_pnl = min(trade.quantity, position.quantity) * pnl_per_share
                position.realized_pnl += realized_pnl
                
                if trade.quantity >= position.quantity:
                    # Closing entire position
                    position.quantity = 0
                    position.avg_price = 0
                else:
                    # Partial close
                    position.quantity -= trade.quantity
        
        # Update market value
        if position.quantity != 0:
            current_price = trade.price
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
        else:
            position.market_value = 0
            position.unrealized_pnl = 0
        
        position.last_update = datetime.utcnow()
    
    async def _check_order_fills(self, symbol: str, current_price: float):
        """Check if any pending limit orders should be filled."""
        orders_to_remove = []
        
        for order_id, order in self.pending_orders.items():
            if order.symbol != symbol or order.status != OrderStatus.PENDING:
                continue
            
            should_fill = False
            
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_price <= order.price:
                    should_fill = True
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    should_fill = True
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    should_fill = True
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    should_fill = True
            
            if should_fill:
                await self._simulate_order_execution(order)
                orders_to_remove.append(order_id)
        
        # Clean up filled orders
        for order_id in orders_to_remove:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
    
    async def _cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.pending_orders[order_id]
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        return False
    
    async def _order_management_loop(self):
        """Background task for order management."""
        while self.is_running:
            try:
                # Clean up old orders
                current_time = datetime.utcnow()
                expired_orders = []
                
                for order_id, order in self.pending_orders.items():
                    # Cancel orders older than 1 hour
                    if (current_time - order.timestamp).total_seconds() > 3600:
                        expired_orders.append(order_id)
                
                for order_id in expired_orders:
                    await self._cancel_order(order_id)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in order management loop: {e}")
                await asyncio.sleep(60)
    
    async def _strategy_execution_loop(self):
        """Background task for strategy execution."""
        while self.is_running:
            try:
                # Execute strategies at regular intervals
                await self.strategy_manager.execute_strategies()
                await asyncio.sleep(1)  # Execute every second for high-frequency
                
            except Exception as e:
                self.logger.error(f"Error in strategy execution loop: {e}")
                await asyncio.sleep(30)
    
    async def _risk_monitoring_loop(self):
        """Background task for risk monitoring."""
        while self.is_running:
            try:
                # Check risk limits
                portfolio_value = await self.portfolio_manager.get_total_value()
                
                # Calculate current drawdown
                if portfolio_value > self.peak_equity:
                    self.peak_equity = portfolio_value
                
                current_drawdown = (self.peak_equity - portfolio_value) / self.peak_equity
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown
                
                # Risk management actions
                if await self.risk_manager.should_halt_trading():
                    self.trading_enabled = False
                    self.logger.warning("Trading halted due to risk limits")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _performance_tracking_loop(self):
        """Background task for performance tracking."""
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(300)
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Calculate total P&L
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            self.total_pnl = total_realized_pnl + total_unrealized_pnl
            
            # Calculate win rate
            if self.executed_trades:
                profitable_trades = sum(1 for trade in self.executed_trades 
                                      if self._calculate_trade_pnl(trade) > 0)
                self.winning_trades = profitable_trades
            
            # Store metrics in database
            await self._store_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a single trade."""
        # This is a simplified calculation
        # In reality, you'd need to match buy/sell trades
        return 0.0  # Placeholder
    
    async def _store_performance_metrics(self):
        """Store performance metrics in database."""
        # Implementation would store metrics in database
        pass
    
    async def _get_historical_data(self, symbol: str, periods: int = 100) -> Optional[pd.DataFrame]:
        """Get historical OHLC data for analysis."""
        try:
            # This would typically fetch from database or data provider
            # For now, return None to avoid errors
            return None
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None
    
    # Public API methods
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return await self.portfolio_manager.get_status()
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        return {
            symbol: {
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'last_update': pos.last_update.isoformat()
            }
            for symbol, pos in self.positions.items()
            if pos.quantity != 0
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        runtime = (datetime.utcnow() - self.start_time).total_seconds() / 86400  # Days
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'runtime_days': runtime,
            'sharpe_ratio': await self._calculate_sharpe_ratio(),
            'active_positions': len([p for p in self.positions.values() if p.quantity != 0]),
            'pending_orders': len(self.pending_orders)
        }
    
    async def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        # Simplified calculation - in reality would use daily returns
        if self.total_trades == 0:
            return 0.0
        
        # Placeholder calculation
        avg_return = self.total_pnl / max(self.total_trades, 1)
        return avg_return * 0.1  # Simplified
