"""Performance Metrics Calculations."""
import numpy as np

class PerformanceMetrics:
    def __init__(self):
        pass
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility > 0 else 0
    
    def calculate_max_drawdown(self, equity_curve):
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()
