#!/usr/bin/env python3
"""
Main application entry point for the Quantitative Trading Platform.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI
from prometheus_client import start_http_server

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.analytics.dashboard import DashboardManager
from src.data.data_pipeline import DataPipeline
from src.trading.engine import TradingEngine
from src.utils.config import Config
from src.utils.logger import setup_logging


class TradingPlatform:
    """Main trading platform orchestrator."""
    
    def __init__(self, config_path: str = "config"):
        self.config = Config(config_path)
        self.logger = setup_logging(self.config.logging)
        self.running = False
        
        # Core components
        self.data_pipeline = None
        self.trading_engine = None
        self.dashboard_manager = None
        
        # FastAPI app for REST API
        self.app = FastAPI(
            title="Quantitative Trading Platform",
            version="1.0.0",
            description="High-frequency algorithmic trading platform"
        )
        
        self._setup_signal_handlers()
        self._setup_routes()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "components": {
                    "data_pipeline": self.data_pipeline.is_running if self.data_pipeline else False,
                    "trading_engine": self.trading_engine.is_running if self.trading_engine else False,
                    "dashboard": self.dashboard_manager.is_running if self.dashboard_manager else False
                }
            }
        
        @self.app.get("/api/v1/portfolio")
        async def get_portfolio():
            if self.trading_engine:
                return await self.trading_engine.get_portfolio_status()
            return {"error": "Trading engine not initialized"}
        
        @self.app.get("/api/v1/positions")
        async def get_positions():
            if self.trading_engine:
                return await self.trading_engine.get_positions()
            return {"error": "Trading engine not initialized"}
        
        @self.app.get("/api/v1/performance")
        async def get_performance():
            if self.trading_engine:
                return await self.trading_engine.get_performance_metrics()
            return {"error": "Trading engine not initialized"}
    
    async def initialize(self):
        """Initialize all platform components."""
        try:
            self.logger.info("Initializing Quantitative Trading Platform...")
            
            # Initialize data pipeline
            self.logger.info("Starting data pipeline...")
            self.data_pipeline = DataPipeline(self.config.kafka)
            await self.data_pipeline.start()
            
            # Initialize trading engine
            self.logger.info("Starting trading engine...")
            self.trading_engine = TradingEngine(
                self.config.trading,
                self.data_pipeline
            )
            await self.trading_engine.start()
            
            # Initialize dashboard manager
            self.logger.info("Starting dashboard manager...")
            self.dashboard_manager = DashboardManager(self.config.tableau)
            await self.dashboard_manager.start()
            
            # Start Prometheus metrics server
            start_http_server(self.config.monitoring.prometheus_port)
            self.logger.info(f"Prometheus metrics server started on port {self.config.monitoring.prometheus_port}")
            
            self.running = True
            self.logger.info("Platform initialization complete!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize platform: {e}")
            await self.shutdown()
            raise
    
    async def run(self):
        """Run the main platform loop."""
        await self.initialize()
        
        try:
            # Start the FastAPI server in the background
            config = uvicorn.Config(
                self.app,
                host=self.config.api.host,
                port=self.config.api.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            # Run the server
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Platform error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown all components."""
        if not self.running:
            return
            
        self.logger.info("Shutting down platform...")
        self.running = False
        
        # Shutdown components in reverse order
        if self.dashboard_manager:
            await self.dashboard_manager.stop()
        
        if self.trading_engine:
            await self.trading_engine.stop()
        
        if self.data_pipeline:
            await self.data_pipeline.stop()
        
        self.logger.info("Platform shutdown complete")


@click.group()
def cli():
    """Quantitative Trading Platform CLI."""
    pass


@cli.command()
@click.option("--config", "-c", default="config", help="Configuration directory path")
def run(config):
    """Run the trading platform."""
    platform = TradingPlatform(config)
    
    try:
        asyncio.run(platform.run())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.option("--symbol", "-s", required=True, help="Trading symbol")
@click.option("--start-date", "-sd", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", "-ed", required=True, help="End date (YYYY-MM-DD)")
@click.option("--strategy", "-st", default="lstm_momentum", help="Trading strategy")
@click.option("--initial-capital", "-ic", default=100000, type=float, help="Initial capital")
def backtest(symbol, start_date, end_date, strategy, initial_capital):
    """Run backtesting for a specific strategy."""
    from src.backtesting.backtest_engine import BacktestEngine
    from datetime import datetime
    
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Initialize backtest engine
    config = Config("config")
    engine = BacktestEngine(config.backtesting)
    
    # Run backtest
    results = engine.run_backtest(
        symbol=symbol,
        start_date=start_dt,
        end_date=end_dt,
        strategy=strategy,
        initial_capital=initial_capital
    )
    
    # Print results
    print(f"\nBacktest Results for {symbol} ({start_date} to {end_date}):")
    print(f"Strategy: {strategy}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Trades: {results['total_trades']}")


@cli.command()
@click.option("--model", "-m", required=True, help="Model type (lstm, transformer)")
@click.option("--symbol", "-s", required=True, help="Trading symbol")
@click.option("--epochs", "-e", default=100, help="Training epochs")
def train(model, symbol, epochs):
    """Train a specific model."""
    from src.models.lstm_model import LSTMModel
    from src.models.transformer_model import TransformerModel
    
    config = Config("config")
    
    if model.lower() == "lstm":
        trainer = LSTMModel(config.models.lstm)
    elif model.lower() == "transformer":
        trainer = TransformerModel(config.models.transformer)
    else:
        raise ValueError(f"Unknown model type: {model}")
    
    print(f"Training {model} model for {symbol}...")
    trainer.train(symbol=symbol, epochs=epochs)
    print("Training complete!")


@cli.command()
@click.option("--simulations", "-n", default=10000, help="Number of Monte Carlo simulations")
@click.option("--confidence", "-c", default=0.95, help="Confidence interval")
def monte_carlo(simulations, confidence):
    """Run Monte Carlo risk analysis."""
    from src.backtesting.monte_carlo import MonteCarloSimulation
    
    config = Config("config")
    mc = MonteCarloSimulation(config.backtesting)
    
    results = mc.run_simulation(
        num_simulations=simulations,
        confidence_interval=confidence
    )
    
    print(f"\nMonte Carlo Simulation Results ({simulations:,} runs):")
    print(f"Expected Return: {results['expected_return']:.2%}")
    print(f"Volatility: {results['volatility']:.2%}")
    print(f"VaR ({confidence*100:.0f}%): {results['var']:.2%}")
    print(f"CVaR ({confidence*100:.0f}%): {results['cvar']:.2%}")


if __name__ == "__main__":
    cli()
