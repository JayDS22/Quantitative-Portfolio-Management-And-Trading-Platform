"""
Logging Utilities for Trading Platform.
Structured logging with performance tracking.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        
        return json.dumps(log_data)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup comprehensive logging for the platform.
    
    Args:
        config: Logging configuration dictionary
    
    Returns:
        Configured logger instance
    """
    
    # Get configuration
    log_level = config.get('level', 'INFO')
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('file', 'logs/trading_platform.log')
    max_bytes = config.get('max_bytes', 10485760)  # 10MB
    backup_count = config.get('backup_count', 5)
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Structured JSON handler for analysis
    json_log_file = log_path.parent / f"{log_path.stem}_structured.json"
    json_handler = logging.handlers.RotatingFileHandler(
        json_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    json_handler.setLevel(logging.INFO)
    json_handler.setFormatter(StructuredFormatter())
    logger.addHandler(json_handler)
    
    # Error-only handler
    error_log_file = log_path.parent / f"{log_path.stem}_errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(error_handler)
    
    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('kafka').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized. Level: {log_level}, File: {log_file}")
    
    return logger


class ColoredFormatter(logging.Formatter):
    """Colored console output formatter."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format message
        result = super().format(record)
        
        # Reset levelname
        record.levelname = levelname
        
        return result


class PerformanceLogger:
    """Logger for tracking performance metrics."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"performance.{name}")
        self.metrics = {}
    
    def log_metric(self, metric_name: str, value: float, **kwargs):
        """Log a performance metric."""
        log_data = {
            'metric': metric_name,
            'value': value,
            **kwargs
        }
        
        # Store in memory for aggregation
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        
        # Log to file
        self.logger.info(f"{metric_name}={value}", extra={'extra_data': log_data})
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics."""
        import numpy as np
        
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        return summary


class TradeLogger:
    """Specialized logger for trade execution."""
    
    def __init__(self):
        self.logger = logging.getLogger("trades")
        self.trade_count = 0
    
    def log_trade(self, trade: Dict[str, Any]):
        """Log a trade execution."""
        self.trade_count += 1
        
        log_message = (
            f"TRADE #{self.trade_count}: "
            f"{trade['symbol']} "
            f"{trade['side'].upper()} "
            f"{trade['quantity']:.2f} @ ${trade['price']:.4f}"
        )
        
        self.logger.info(log_message, extra={'extra_data': trade})
    
    def log_order(self, order: Dict[str, Any]):
        """Log an order placement."""
        log_message = (
            f"ORDER: {order['symbol']} "
            f"{order['side'].upper()} "
            f"{order['quantity']:.2f} "
            f"{order['order_type'].upper()}"
        )
        
        if order.get('price'):
            log_message += f" @ ${order['price']:.4f}"
        
        self.logger.info(log_message, extra={'extra_data': order})
    
    def log_pnl(self, symbol: str, pnl: float, pnl_pct: float):
        """Log P&L update."""
        sign = "+" if pnl >= 0 else ""
        self.logger.info(
            f"P&L: {symbol} {sign}${pnl:.2f} ({sign}{pnl_pct:.2%})",
            extra={'extra_data': {'symbol': symbol, 'pnl': pnl, 'pnl_pct': pnl_pct}}
        )


class AlertLogger:
    """Logger for critical alerts and notifications."""
    
    def __init__(self):
        self.logger = logging.getLogger("alerts")
    
    def log_alert(self, alert_type: str, message: str, severity: str = "INFO", **kwargs):
        """Log an alert."""
        alert_data = {
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            **kwargs
        }
        
        level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(level, f"[{alert_type}] {message}", extra={'extra_data': alert_data})
    
    def log_risk_alert(self, message: str, **kwargs):
        """Log a risk management alert."""
        self.log_alert("RISK", message, "WARNING", **kwargs)
    
    def log_system_alert(self, message: str, **kwargs):
        """Log a system alert."""
        self.log_alert("SYSTEM", message, "ERROR", **kwargs)
    
    def log_trading_halt(self, reason: str, **kwargs):
        """Log a trading halt event."""
        self.log_alert("TRADING_HALT", f"Trading halted: {reason}", "CRITICAL", **kwargs)
