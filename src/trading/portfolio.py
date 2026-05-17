"""
Advanced Portfolio Management System.
Handles position sizing, risk allocation, and P&L tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..utils.database import get_db_session


@dataclass
class Position:
    """Portfolio position representation."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    cost_basis: float
    side: str  # 'long' or 'short'
    entry_time: datetime
    last_update: datetime


@dataclass
class PortfolioSummary:
    """Portfolio summary metrics."""
    total_value: float
    cash_balance: float
    equity_value: float
    margin_used: float
    margin_available: float
    total_pnl: float
    daily_pnl: float
    positions_count: int
    leverage: float
    portfolio_beta: float
    sharpe_ratio: float
    max_drawdown: float


class PortfolioManager:
    """Advanced portfolio management with risk controls."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.initial_capital = config.get('initial_capital', 100000)
        self.cash_balance = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Risk parameters
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% max position
        self.max_leverage = config.get('max_leverage', 2.0)
        self.margin_requirement = config.get('margin_requirement', 0.5)
        self.cash_reserve = config.get('cash_reserve', 0.1)  # 10% cash reserve
        
        # Performance tracking
        self.portfolio_history = []
        self.daily_returns = []
        self.benchmark_returns = []
        
        # Kelly Criterion parameters
        self.kelly_lookback = config.get('kelly_lookback', 252)  # 1 year
        self.max_kelly_fraction = config.get('max_kelly_fraction', 0.25)  # Max 25%
        
    async def initialize(self):
        """Initialize portfolio manager."""
        self.logger.info("Initializing portfolio manager...")
        
        # Load existing positions from database
        await self._load_portfolio_state()
        
        # Initialize performance tracking
        await self._initialize_performance_tracking()
        
        self.logger.info(f"Portfolio initialized with {len(self.positions)} positions")
    
    async def _load_portfolio_state(self):
        """Load portfolio state from database."""
        try:
            # This would load from database in production
            # For now, start with clean slate
            self.positions = {}
            self.cash_balance = self.initial_capital
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {e}")
    
    async def _initialize_performance_tracking(self):
        """Initialize performance tracking systems."""
        try:
            # Load historical performance data
            self.portfolio_history = []
            self.daily_returns = []
            
            # Record initial portfolio value
            await self._record_portfolio_snapshot()
            
        except Exception as e:
            self.logger.error(f"Error initializing performance tracking: {e}")
    
    async def process_trade(self, trade: Dict[str, Any]):
        """Process a trade and update portfolio."""
        try:
            symbol = trade['symbol']
            side = trade['side']
            quantity = trade['quantity']
            price = trade['price']
            commission = trade.get('commission', 0)
            
            # Update position
            await self._update_position(symbol, side, quantity, price, commission)
            
            # Update cash balance
            if side.lower() == 'buy':
                self.cash_balance -= (quantity * price + commission)
            else:  # sell
                self.cash_balance += (quantity * price - commission)
            
            # Record trade
            self.trade_history.append({
                **trade,
                'timestamp': datetime.utcnow(),
                'cash_balance_after': self.cash_balance
            })
            
            # Update performance metrics
            await self._update_performance_metrics()
            
            self.logger.info(f"Processed trade: {symbol} {side} {quantity} @ {price}")
            
        except Exception as e:
            self.logger.error(f"Error processing trade: {e}")
    
    async def _update_position(self, symbol: str, side: str, quantity: float, 
                             price: float, commission: float):
        """Update position for a symbol."""
        
        if symbol not in self.positions:
            # New position
            if side.lower() == 'buy':
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    cost_basis=quantity * price + commission,
                    side='long',
                    entry_time=datetime.utcnow(),
                    last_update=datetime.utcnow()
                )
        else:
            # Existing position
            position = self.positions[symbol]
            
            if side.lower() == 'buy':
                # Adding to long position
                total_cost = position.cost_basis + (quantity * price + commission)
                total_quantity = position.quantity + quantity
                
                position.avg_price = (total_cost - commission) / total_quantity
                position.quantity = total_quantity
                position.cost_basis = total_cost
                position.market_value = total_quantity * position.current_price
                
            else:  # sell
                # Reducing/closing position
                if quantity >= position.quantity:
                    # Closing entire position
                    realized_pnl = (price - position.avg_price) * position.quantity - commission
                    position.realized_pnl += realized_pnl
                    
                    # Remove position if fully closed
                    if quantity == position.quantity:
                        del self.positions[symbol]
                        return
                    else:
                        # Going short (if allowed)
                        position.quantity = quantity - position.quantity
                        position.side = 'short'
                        position.avg_price = price
                else:
                    # Partial close
                    realized_pnl = (price - position.avg_price) * quantity - commission
                    position.realized_pnl += realized_pnl
                    position.quantity -= quantity
                    position.cost_basis -= (position.avg_price * quantity + commission)
            
            position.last_update = datetime.utcnow()
    
    async def update_market_prices(self, prices: Dict[str, float]):
        """Update current market prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = price
                position.market_value = position.quantity * price
                
                # Update unrealized P&L
                if position.side == 'long':
                    position.unrealized_pnl = (price - position.avg_price) * position.quantity
                else:  # short
                    position.unrealized_pnl = (position.avg_price - price) * position.quantity
                
                position.last_update = datetime.utcnow()
    
    async def get_portfolio_summary(self) -> PortfolioSummary:
        """Get comprehensive portfolio summary."""
        
        # Calculate portfolio metrics
        equity_value = sum(pos.market_value for pos in self.positions.values())
        total_value = self.cash_balance + equity_value
        
        total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values())
        
        # Calculate daily P&L
        daily_pnl = await self._calculate_daily_pnl()
        
        # Calculate margin usage
        margin_used = sum(abs(pos.market_value) * self.margin_requirement 
                         for pos in self.positions.values())
        margin_available = total_value - margin_used
        
        # Calculate leverage
        gross_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        leverage = gross_exposure / total_value if total_value > 0 else 0
        
        # Calculate portfolio beta (simplified)
        portfolio_beta = await self._calculate_portfolio_beta()
        
        # Calculate Sharpe ratio
        sharpe_ratio = await self._calculate_sharpe_ratio()
        
        # Calculate max drawdown
        max_drawdown = await self._calculate_max_drawdown()
        
        return PortfolioSummary(
            total_value=total_value,
            cash_balance=self.cash_balance,
            equity_value=equity_value,
            margin_used=margin_used,
            margin_available=margin_available,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            positions_count=len(self.positions),
            leverage=leverage,
            portfolio_beta=portfolio_beta,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown
        )
    
    async def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L."""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        current_value = await self.get_total_value()
        previous_value = self.portfolio_history[-2]['total_value']
        
        return current_value - previous_value
    
    async def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta against benchmark."""
        if len(self.daily_returns) < 30 or len(self.benchmark_returns) < 30:
            return 1.0  # Default beta
        
        portfolio_returns = np.array(self.daily_returns[-252:
