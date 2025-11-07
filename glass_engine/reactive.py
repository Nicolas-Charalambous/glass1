import reactivex as rx
from reactivex.subject import Subject, BehaviorSubject
from reactivex import operators as ops
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple
import time
import threading
from dataclasses import dataclass
from functools import lru_cache
import weakref

@dataclass
class MarketTick:
    symbol: str
    timestamp: float
    price: float
    field: str = "close"  # Default field

@dataclass
class ExpressionResult:
    expression_id: str
    timestamp: float
    value: float
    dependencies: List[str]

class CircularBuffer:
    """Efficient circular buffer for time series data"""
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.full(size, np.nan)
        self.index = 0
        self.count = 0
        self.is_full = False
    
    def append(self, value: float):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size
        if not self.is_full:
            self.count += 1
            if self.count == self.size:
                self.is_full = True
    
    def get_array(self) -> np.ndarray:
        """Get the data in chronological order"""
        if not self.is_full:
            return self.buffer[:self.count].copy()
        else:
            # Reorder: [index:] + [:index]
            return np.concatenate([self.buffer[self.index:], self.buffer[:self.index]])
    
    def get_latest(self) -> float:
        """Get the most recent value"""
        if self.count == 0:
            return np.nan
        prev_index = (self.index - 1) % self.size
        return self.buffer[prev_index]

class ReactiveSymbolResolver:
    """Enhanced symbol resolver with reactive capabilities"""
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.buffers: Dict[str, CircularBuffer] = {}
        self.streams: Dict[str, Subject] = {}
        self.latest_values: Dict[str, float] = {}
        
    def register_symbol(self, symbol: str):
        """Register a symbol for reactive updates"""
        if symbol not in self.buffers:
            self.buffers[symbol] = CircularBuffer(self.buffer_size)
            self.streams[symbol] = Subject()
            self.latest_values[symbol] = np.nan
    
    def update_symbol(self, symbol: str, value: float, timestamp: float):
        """Update a symbol with new data"""
        if symbol not in self.buffers:
            self.register_symbol(symbol)
        
        self.buffers[symbol].append(value)
        self.latest_values[symbol] = value
        
        # Emit to reactive stream
        tick = MarketTick(symbol=symbol, timestamp=timestamp, price=value)
        self.streams[symbol].on_next(tick)
    
    def get_data(self, symbol: str) -> np.ndarray:
        """Get historical data for a symbol"""
        if symbol not in self.buffers:
            raise ValueError(f"Unknown symbol: {symbol}")
        return self.buffers[symbol].get_array()
    
    def get_field(self, symbol: str, field: str) -> np.ndarray:
        """Get field data (maintaining compatibility with original interface)"""
        full_symbol = f"{symbol}.{field}"
        return self.get_data(full_symbol)
    
    def get_stream(self, symbol: str) -> Subject:
        """Get the reactive stream for a symbol"""
        if symbol not in self.streams:
            self.register_symbol(symbol)
        return self.streams[symbol]
    
    def get_latest_value(self, symbol: str) -> float:
        """Get the latest value for a symbol"""
        return self.latest_values.get(symbol, np.nan)

class ExpressionDependencyAnalyzer:
    """Analyzes expression dependencies for optimized updates"""
    
    @staticmethod
    def extract_symbols(expression_tree) -> List[str]:
        """Extract all symbols referenced in an expression"""
        symbols = []
        
        def traverse(node):
            if hasattr(node, 'data'):
                if node.data == 'symbol':
                    symbols.append(str(node.children[0]))
                elif node.data == 'field_access':
                    symbol = str(node.children[0])
                    field = str(node.children[1])
                    symbols.append(f"{symbol}.{field}")
                else:
                    for child in node.children:
                        if hasattr(child, 'data'):
                            traverse(child)
        
        traverse(expression_tree)
        return list(set(symbols))  # Remove duplicates

class ReactiveExpressionEngine:
    """Reactive wrapper around your existing ExpressionEngine"""
    
    def __init__(self, expression_engine, buffer_size: int = 1000):
        self.engine = expression_engine
        self.reactive_resolver = ReactiveSymbolResolver(buffer_size)
        self.expressions: Dict[str, Dict] = {}  # expression_id -> expression_info
        self.result_streams: Dict[str, Subject] = {}
        self.update_scheduler = Subject()
        self.dependency_analyzer = ExpressionDependencyAnalyzer()
        
        # Replace the engine's symbol resolver with our reactive one
        self.engine.symbol_resolver = self.reactive_resolver
        self.engine.transformer.symbols = self.reactive_resolver
        
        # Set up batch update processing
        self._setup_batch_processing()
    
    def _setup_batch_processing(self):
        """Set up batched expression evaluation to handle multiple updates efficiently"""
        self.update_scheduler.pipe(
            ops.buffer_with_time(0.1),  # Batch updates every 100ms
            ops.filter(lambda batch: len(batch) > 0),
            ops.map(self._process_batch_updates)
        ).subscribe(
            on_next=lambda results: self._emit_batch_results(results),
            on_error=lambda e: print(f"Batch processing error: {e}")
        )
    
    def register_expression(self, expression_id: str, expression: str, 
                          dependencies: Optional[List[str]] = None) -> Subject:
        """Register an expression for reactive evaluation"""
        
        # Parse expression to extract dependencies if not provided
        if dependencies is None:
            try:
                tree = self.engine.parse(expression)
                dependencies = self.dependency_analyzer.extract_symbols(tree)
            except Exception as e:
                print(f"Failed to parse expression {expression}: {e}")
                dependencies = []
        
        # Create result stream
        result_stream = Subject()
        self.result_streams[expression_id] = result_stream
        
        # Store expression info
        self.expressions[expression_id] = {
            'expression': expression,
            'dependencies': dependencies,
            'last_update': 0,
            'cached_result': np.nan
        }
        
        # Set up reactive pipeline for this expression
        self._setup_expression_pipeline(expression_id, dependencies)
        
        return result_stream
    
    def _setup_expression_pipeline(self, expression_id: str, dependencies: List[str]):
        """Set up reactive pipeline for an expression based on its dependencies"""
        
        if not dependencies:
            return
        
        # Create combined stream from all dependencies
        dependency_streams = []
        for dep in dependencies:
            self.reactive_resolver.register_symbol(dep)
            stream = self.reactive_resolver.get_stream(dep)
            dependency_streams.append(stream)
        
        # Merge all dependency streams
        if len(dependency_streams) == 1:
            combined_stream = dependency_streams[0]
        else:
            combined_stream = rx.merge(*dependency_streams)
        
        # Set up the reactive pipeline
        combined_stream.pipe(
            # Debounce rapid updates
            ops.debounce(0.01),  # 10ms debounce
            # Add expression_id context
            ops.map(lambda tick: (expression_id, tick)),
            # Rate limiting to prevent overwhelming
            ops.sample(0.05)  # Sample every 50ms max
        ).subscribe(
            on_next=lambda data: self.update_scheduler.on_next(data),
            on_error=lambda e: print(f"Expression {expression_id} stream error: {e}")
        )
    
    def _process_batch_updates(self, batch: List[Tuple[str, MarketTick]]) -> List[ExpressionResult]:
        """Process a batch of expression updates"""
        results = []
        
        # Group by expression_id to avoid duplicate calculations
        expressions_to_update = set()
        latest_timestamp = 0
        
        for expression_id, tick in batch:
            expressions_to_update.add(expression_id)
            latest_timestamp = max(latest_timestamp, tick.timestamp)
        
        # Evaluate each unique expression
        for expression_id in expressions_to_update:
            try:
                expr_info = self.expressions[expression_id]
                
                # Skip if updated very recently (additional protection)
                if latest_timestamp - expr_info['last_update'] < 0.01:
                    continue
                
                # Evaluate expression
                result_array = self.engine.evaluate(expr_info['expression'])
                
                # Extract latest value
                if len(result_array) > 0:
                    latest_value = float(result_array[-1])
                    if not np.isnan(latest_value):
                        expr_info['cached_result'] = latest_value
                        expr_info['last_update'] = latest_timestamp
                        
                        results.append(ExpressionResult(
                            expression_id=expression_id,
                            timestamp=latest_timestamp,
                            value=latest_value,
                            dependencies=expr_info['dependencies']
                        ))
                        
            except Exception as e:
                print(f"Error evaluating expression {expression_id}: {e}")
        
        return results
    
    def _emit_batch_results(self, results: List[ExpressionResult]):
        """Emit results to their respective streams"""
        for result in results:
            if result.expression_id in self.result_streams:
                self.result_streams[result.expression_id].on_next(result)
    
    def update_market_data(self, symbol: str, value: float, timestamp: float):
        """Update market data (called from your data providers)"""
        self.reactive_resolver.update_symbol(symbol, value, timestamp)
    
    def get_latest_result(self, expression_id: str) -> float:
        """Get the latest computed result for an expression"""
        if expression_id in self.expressions:
            return self.expressions[expression_id]['cached_result']
        return np.nan
    
    def get_historical_result(self, expression_id: str) -> np.ndarray:
        """Get historical results by re-evaluating with full historical data"""
        if expression_id in self.expressions:
            try:
                return self.engine.evaluate(self.expressions[expression_id]['expression'])
            except Exception as e:
                print(f"Error getting historical data for {expression_id}: {e}")
                return np.array([])
        return np.array([])


# Integration with your existing data providers
class ReactiveDataProvider:
    """Wrapper to integrate your data providers with ReactiveX"""
    
    def __init__(self, data_provider, reactive_engine: ReactiveExpressionEngine):
        self.data_provider = data_provider
        self.reactive_engine = reactive_engine
        
        # Replace the callback to feed into reactive engine
        original_callback = data_provider.cb_on_message
        self.data_provider.cb_on_message = self._reactive_callback
    
    def _reactive_callback(self, message):
        """Process incoming market data and feed to reactive engine"""
        try:
            if isinstance(message, tuple) and len(message) == 2:
                timestamp, price = message
                # Assuming the ticker is available from the data provider
                symbol = getattr(self.data_provider, 'ticker', 'UNKNOWN')
                
                # Update reactive engine
                self.reactive_engine.update_market_data(
                    symbol=symbol, 
                    value=float(price), 
                    timestamp=float(timestamp)
                )
                
        except Exception as e:
            print(f"Error processing market data: {e}")
    
    def subscribe(self):
        """Start the data provider"""
        return self.data_provider.subscribe()


# Example usage
def example_usage():
    """Example of how to use the reactive system"""
    from transformers import ExpressionEngine  # Your existing engine
    from glass_engine.constants import SYNTAX_PATH
    
    # Create reactive engine
    engine = ExpressionEngine(SYNTAX_PATH)  # Your existing engine
    reactive_engine = ReactiveExpressionEngine(engine, buffer_size=1000)
    
    # Register expressions
    sma_stream = reactive_engine.register_expression(
        'sma_20', 
        'sma(BINANCE:BTCUSDT, 20)'
    )
    
    rsi_stream = reactive_engine.register_expression(
        'rsi_14', 
        'rsi(BINANCE:BTCUSDT, 14)'
    )
    
    ratio_stream = reactive_engine.register_expression(
        'price_ratio',
        'BINANCE:BTCUSDT / sma(BINANCE:BTCUSDT, 50)'
    )
    
    # Subscribe to results
    sma_stream.subscribe(
        lambda result: print(f"SMA(20): {result.value:.2f} at {result.timestamp}")
    )
    
    rsi_stream.subscribe(
        lambda result: print(f"RSI(14): {result.value:.2f} at {result.timestamp}")
    )
    
    ratio_stream.subscribe(
        lambda result: print(f"Price/SMA50 Ratio: {result.value:.4f} at {result.timestamp}")
    )
    
    # Set up data provider
    from data_provider import FinnhubDataProvider  # Your existing provider
    
    data_provider = FinnhubDataProvider("BINANCE:BTCUSDT", lambda x: None)
    reactive_provider = ReactiveDataProvider(data_provider, reactive_engine)
    
    # Start receiving live data
    reactive_provider.subscribe()

if __name__ == "__main__":
    example_usage()