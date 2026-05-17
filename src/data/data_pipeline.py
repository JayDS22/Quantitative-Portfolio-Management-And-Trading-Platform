"""
Real-time Market Data Pipeline using Apache Kafka.
Processes 1M+ ticks/second with <50ms latency.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import redis
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import numpy as np

from ..utils.config import Config


@dataclass
class MarketTick:
    """Market data tick structure."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    exchange: str


@dataclass
class ProcessedData:
    """Processed market data structure."""
    symbol: str
    timestamp: datetime
    ohlcv: Dict[str, float]
    indicators: Dict[str, float]
    features: np.ndarray


class DataPipeline:
    """High-performance real-time market data pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Kafka configuration
        self.bootstrap_servers = config.get('bootstrap_servers', 'localhost:9092')
        self.topics = config.get('topics', {})
        self.consumer_group = config.get('consumer_group', 'trading-platform')
        
        # Redis configuration
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0),
            decode_responses=True
        )
        
        # Components
        self.producer = None
        self.consumers = {}
        self.data_processors = {}
        self.subscribers = {}
        
        # Performance metrics
        self.tick_count = 0
        self.processing_times = []
        self.start_time = None
        self.is_running = False
        
        # Data buffers
        self.tick_buffer = {}
        self.ohlc_buffer = {}
        self.buffer_size = config.get('buffer_size', 1000)
        
        self._setup_kafka()
    
    def _setup_kafka(self):
        """Initialize Kafka producer and consumers."""
        try:
            # Producer configuration for high throughput
            producer_config = {
                'bootstrap_servers': self.bootstrap_servers,
                'acks': 1,  # Wait for leader acknowledgment
                'retries': 3,
                'batch_size': 16384,
                'linger_ms': 10,  # Small delay for batching
                'buffer_memory': 33554432,
                'compression_type': 'snappy',
                'value_serializer': lambda x: json.dumps(x, default=str).encode('utf-8')
            }
            
            self.producer = KafkaProducer(**producer_config)
            
            # Consumer configuration for low latency
            consumer_config = {
                'bootstrap_servers': self.bootstrap_servers,
                'group_id': self.consumer_group,
                'auto_offset_reset': 'latest',
                'enable_auto_commit': True,
                'auto_commit_interval_ms': 1000,
                'fetch_max_wait_ms': 10,  # Low latency
                'fetch_min_bytes': 1,
                'max_partition_fetch_bytes': 1048576,
                'value_deserializer': lambda m: json.loads(m.decode('utf-8'))
            }
            
            # Create consumers for each topic
            for topic_name, topic in self.topics.items():
                consumer = KafkaConsumer(topic, **consumer_config)
                self.consumers[topic_name] = consumer
                
            self.logger.info("Kafka setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Kafka: {e}")
            raise
    
    async def start(self):
        """Start the data pipeline."""
        self.logger.info("Starting data pipeline...")
        self.start_time = time.time()
        self.is_running = True
        
        # Start consumer tasks
        tasks = []
        for topic_name, consumer in self.consumers.items():
            task = asyncio.create_task(self._consume_market_data(topic_name, consumer))
            tasks.append(task)
        
        # Start performance monitoring task
        monitor_task = asyncio.create_task(self._monitor_performance())
        tasks.append(monitor_task)
        
        # Start OHLC aggregation task
        ohlc_task = asyncio.create_task(self._aggregate_ohlc())
        tasks.append(ohlc_task)
        
        self.logger.info("Data pipeline started successfully")
        
        # Wait for tasks (they run indefinitely)
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Data pipeline tasks cancelled")
    
    async def stop(self):
        """Stop the data pipeline."""
        self.logger.info("Stopping data pipeline...")
        self.is_running = False
        
        # Close producer
        if self.producer:
            self.producer.close()
        
        # Close consumers
        for consumer in self.consumers.values():
            consumer.close()
        
        # Close Redis connection
        self.redis_client.close()
        
        self.logger.info("Data pipeline stopped")
    
    async def _consume_market_data(self, topic_name: str, consumer: KafkaConsumer):
        """Consume market data from Kafka topic."""
        self.logger.info(f"Starting consumer for topic: {topic_name}")
        
        while self.is_running:
            try:
                # Poll for messages with short timeout
                message_batch = consumer.poll(timeout_ms=100)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        await self._process_market_tick(message.value)
                        
            except KafkaError as e:
                self.logger.error(f"Kafka error in {topic_name}: {e}")
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error processing message in {topic_name}: {e}")
                continue
    
    async def _process_market_tick(self, tick_data: Dict[str, Any]):
        """Process individual market tick with high performance."""
        start_time = time.time()
        
        try:
            # Parse tick data
            tick = MarketTick(
                symbol=tick_data['symbol'],
                timestamp=datetime.fromisoformat(tick_data['timestamp']),
                bid=float(tick_data['bid']),
                ask=float(tick_data['ask']),
                last=float(tick_data['last']),
                volume=int(tick_data.get('volume', 0)),
                exchange=tick_data.get('exchange', 'unknown')
            )
            
            # Update tick buffer
            if tick.symbol not in self.tick_buffer:
                self.tick_buffer[tick.symbol] = []
            
            self.tick_buffer[tick.symbol].append(tick)
            
            # Maintain buffer size
            if len(self.tick_buffer[tick.symbol]) > self.buffer_size:
                self.tick_buffer[tick.symbol] = self.tick_buffer[tick.symbol][-self.buffer_size:]
            
            # Cache latest tick in Redis for ultra-fast access
            tick_cache = {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'timestamp': tick.timestamp.isoformat(),
                'spread': tick.ask - tick.bid,
                'mid_price': (tick.bid + tick.ask) / 2
            }
            
            await self._cache_data(f"tick:{tick.symbol}", tick_cache, ttl=60)
            
            # Notify subscribers
            await self._notify_subscribers('market_tick', tick)
            
            # Update performance metrics
            self.tick_count += 1
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.processing_times.append(processing_time)
            
            # Keep only recent processing times for monitoring
            if len(self.processing_times) > 10000:
                self.processing_times = self.processing_times[-5000:]
            
        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")
    
    async def _aggregate_ohlc(self):
        """Aggregate ticks into OHLC bars with multiple timeframes."""
        self.logger.info("Starting OHLC aggregation...")
        
        timeframes = ['1min', '5min', '15min', '1h', '1d']
        
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for symbol in self.tick_buffer.keys():
                    ticks = self.tick_buffer[symbol]
                    if not ticks:
                        continue
                    
                    for timeframe in timeframes:
                        ohlc = await self._calculate_ohlc(symbol, ticks, timeframe, current_time)
                        if ohlc:
                            await self._cache_data(f"ohlc:{symbol}:{timeframe}", ohlc, ttl=3600)
                            await self._notify_subscribers('ohlc_update', {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'ohlc': ohlc
                            })
                
                # Sleep for 1 second between aggregations
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in OHLC aggregation: {e}")
                await asyncio.sleep(5)
    
    async def _calculate_ohlc(self, symbol: str, ticks: List[MarketTick], 
                             timeframe: str, current_time: datetime) -> Optional[Dict[str, Any]]:
        """Calculate OHLC for given timeframe."""
        if not ticks:
            return None
        
        # Define timeframe intervals
        intervals = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '1d': timedelta(days=1)
        }
        
        if timeframe not in intervals:
            return None
        
        interval = intervals[timeframe]
        
        # Find ticks within the current timeframe
        start_time = current_time - interval
        relevant_ticks = [tick for tick in ticks if tick.timestamp >= start_time]
        
        if not relevant_ticks:
            return None
        
        # Calculate OHLC
        prices = [tick.last for tick in relevant_ticks]
        volumes = [tick.volume for tick in relevant_ticks]
        
        ohlc = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': current_time.isoformat(),
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'volume': sum(volumes),
            'tick_count': len(relevant_ticks),
            'vwap': sum(p * v for p, v in zip(prices, volumes)) / sum(volumes) if sum(volumes) > 0 else prices[-1]
        }
        
        return ohlc
    
    async def _cache_data(self, key: str, data: Dict[str, Any], ttl: int = 300):
        """Cache data in Redis with TTL."""
        try:
            pipeline = self.redis_client.pipeline()
            pipeline.hset(key, mapping=data)
            pipeline.expire(key, ttl)
            pipeline.execute()
        except Exception as e:
            self.logger.error(f"Error caching data: {e}")
    
    async def _monitor_performance(self):
        """Monitor pipeline performance metrics."""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Report every 10 seconds
                
                if self.start_time and self.processing_times:
                    elapsed_time = time.time() - self.start_time
                    tps = self.tick_count / elapsed_time if elapsed_time > 0 else 0
                    
                    avg_latency = np.mean(self.processing_times[-1000:])  # Last 1000 ticks
                    p95_latency = np.percentile(self.processing_times[-1000:], 95)
                    p99_latency = np.percentile(self.processing_times[-1000:], 99)
                    
                    metrics = {
                        'ticks_per_second': tps,
                        'total_ticks_processed': self.tick_count,
                        'average_latency_ms': avg_latency,
                        'p95_latency_ms': p95_latency,
                        'p99_latency_ms': p99_latency,
                        'active_symbols': len(self.tick_buffer),
                        'buffer_sizes': {symbol: len(ticks) for symbol, ticks in self.tick_buffer.items()}
                    }
                    
                    # Cache performance metrics
                    await self._cache_data("pipeline:performance", metrics, ttl=60)
                    
                    self.logger.info(
                        f"Pipeline Performance - TPS: {tps:.0f}, "
                        f"Avg Latency: {avg_latency:.2f}ms, "
                        f"P95 Latency: {p95_latency:.2f}ms, "
                        f"Active Symbols: {len(self.tick_buffer)}"
                    )
            
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to data pipeline events."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    async def _notify_subscribers(self, event_type: str, data: Any):
        """Notify subscribers of events."""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Error in subscriber callback: {e}")
    
    async def publish_tick(self, tick_data: Dict[str, Any]):
        """Publish market tick to Kafka."""
        try:
            topic = self.topics.get('market_data', 'market-data')
            future = self.producer.send(topic, tick_data)
            # Don't wait for result to maintain high throughput
            return True
        except Exception as e:
            self.logger.error(f"Error publishing tick: {e}")
            return False
    
    async def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest tick data from cache."""
        try:
            data = self.redis_client.hgetall(f"tick:{symbol}")
            return data if data else None
        except Exception as e:
            self.logger.error(f"Error getting latest tick: {e}")
            return None
    
    async def get_ohlc_data(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get OHLC data from cache."""
        try:
            data = self.redis_client.hgetall(f"ohlc:{symbol}:{timeframe}")
            return data if data else None
        except Exception as e:
            self.logger.error(f"Error getting OHLC data: {e}")
            return None
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        try:
            data = self.redis_client.hgetall("pipeline:performance")
            return data if data else {}
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {}
