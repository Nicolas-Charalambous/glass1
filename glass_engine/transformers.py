from lark import Lark, Transformer, v_args
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, List, Union
from glass_engine.functions import FUNCTIONS

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymbolResolver:
    """Handles resolution of symbols to actual timeseries data"""
    def __init__(self):
        self.data = {}
    
    def register_symbol(self, symbol: str, data: np.ndarray):
        """Register a symbol with its timeseries data"""
        self.data[symbol] = data
    
    def get_data(self, symbol: str) -> np.ndarray:
        """Get timeseries data for a symbol"""
        if symbol not in self.data:
            raise ValueError(f"Unknown symbol: {symbol}")
        return self.data[symbol]
    
    def get_field(self, symbol: str, field: str) -> np.ndarray:
        """Get a specific field from a symbol (e.g., AAPL.close)"""
        full_symbol = f"{symbol}.{field}"
        return self.get_data(full_symbol)
    
    def update(self, symbol_data: Dict[str, np.ndarray]):
        """Update multiple symbols at once"""
        self.data.update(symbol_data)

class TimeseriesTransformer(Transformer):
    """Transforms parse tree into computed timeseries results"""
    
    def __init__(self, symbol_resolver: SymbolResolver):
        self.symbols = symbol_resolver
        self.functions = FUNCTIONS
    
    # Arithmetic operations
    @v_args(inline=True)
    def add(self, left: np.ndarray, right: Union[np.ndarray, float]) -> np.ndarray:
        return left + right
    
    @v_args(inline=True)
    def sub(self, left: np.ndarray, right: Union[np.ndarray, float]) -> np.ndarray:
        return left - right
    
    @v_args(inline=True)
    def mul(self, left: np.ndarray, right: Union[np.ndarray, float]) -> np.ndarray:
        return left * right
    
    @v_args(inline=True)
    def div(self, left: np.ndarray, right: Union[np.ndarray, float]) -> np.ndarray:
        return left / right
    
    @v_args(inline=True)
    def pow(self, left: np.ndarray, right: Union[np.ndarray, float]) -> np.ndarray:
        return np.power(left, right)
    
    # Data access
    @v_args(inline=True)
    def number(self, value: str) -> float:
        return float(value)
    
    @v_args(inline=True)
    def symbol(self, name: str) -> np.ndarray:
        return self.symbols.get_data(str(name))
    
    @v_args(inline=True)
    def string(self, value: str) -> str:
        # Remove the surrounding quotes
        return str(value)[1:-1]
    
    @v_args(inline=True)
    def field_access(self, symbol: str, field: str) -> np.ndarray:
        return self.symbols.get_field(str(symbol), str(field))
    
    # Function calls
    @v_args(inline=True)
    def func_call(self, func_name: str, *args) -> np.ndarray:
        func_name = str(func_name)
        if func_name in self.functions:
            return self.functions[func_name](*args[0].children if args else [])
        else:
            raise ValueError(f"Unknown function: {func_name}")


class ExpressionEngine:
    """Main engine for parsing and evaluating timeseries expressions"""
    
    def __init__(self, grammar_file: str = None):
        self.symbol_resolver = SymbolResolver()
        self.transformer = TimeseriesTransformer(self.symbol_resolver)
        
        # Load grammar from file or use default
        if grammar_file:
            self.parser = Lark.open(grammar_file, parser='lalr', transformer=self.transformer)
        else:
            # Use inline grammar if no file provided
            grammar = """
            ?start: expression
            ?expression: term | expression "+" term -> add | expression "-" term -> sub
            ?term: factor | term "*" factor -> mul | term "/" factor -> div
            ?factor: atom | factor "^" atom -> pow
            ?atom: NUMBER -> number | SYMBOL -> symbol | STRING -> string | field_access | function_call | "(" expression ")"
            field_access: SYMBOL "." SYMBOL -> field_access
            function_call: SYMBOL "(" arguments? ")" -> func_call
            arguments: expression ("," expression)*
            SYMBOL: /[a-zA-Z_][a-zA-Z0-9_.]*/
            NUMBER: /\d+(\.\d*)?/
            STRING: /'[^']*'/ | /"[^"]*"/
            %import common.WS
            %ignore WS
            """
            self.parser = Lark(grammar, parser='lalr', transformer=self.transformer)
    
    def register_symbol(self, symbol: str, data: np.ndarray):
        """Register a symbol with timeseries data"""
        self.symbol_resolver.register_symbol(symbol, data)
    
    def register_symbols(self, symbol_data: Dict[str, np.ndarray]):
        """Register multiple symbols at once"""
        for symbol, data in symbol_data.items():
            self.register_symbol(symbol, data)
    
    def evaluate(self, expression: str) -> np.ndarray:
        """Parse and evaluate an expression"""
        try:
            result = self.parser.parse(expression)
            if isinstance(result, (int, float)):
                # Convert scalar results to arrays for consistency
                return np.array([result])
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")
    
    def register_custom_function(self, name: str, func: Callable):
        """Register a custom function"""
        self.transformer.functions[name] = func

# Example usage and testing
if __name__ == "__main__":
    # Create engine
    engine = ExpressionEngine(r'C:\Users\nicol\Projects\glass_dag\glass_engine\syntax.lark')
    
    # Sample data
    np.random.seed(42)
    prices = np.random.randn(100).cumsum() + 100
    volumes = np.random.randint(1000, 10000, 100)
    
    # Register symbols
    engine.register_symbols({
        'AAPL.close': prices,
        'AAPL.volume': volumes,
        'GOOGL.close': prices * 1.5 + np.random.randn(100) * 5,
    })
    
    # Test expressions
    test_expressions = [
        "AAPL.close + 10",
        "sma(AAPL.close, 20)",
        "AAPL.close + GOOGL.close",
        "sma(AAPL.close, 20) + rsi(GOOGL.close, 14)",
        "(AAPL.close + GOOGL.close) / 2",
    ]
    
    import time
    start = time.time()
    for expr in test_expressions:
        try:
            result = engine.evaluate(expr)
            logger.info(f"✓ {expr} -> Result shape: {result.shape}, Sample: {result[-5:]}")
        except Exception as e:
            logger.info(f"✗ {expr} -> Error: {e}")
    end = time.time()
    logger.info(f"Evaluation time: {end - start:.6f} seconds")

    start = time.time()
    for expr in test_expressions:
        try:
            result = engine.evaluate(expr)
            logger.info(f"✓ {expr} -> Result shape: {result.shape}, Sample: {result[-5:]}")
        except Exception as e:
            logger.info(f"✗ {expr} -> Error: {e}")
    end = time.time()
    logger.info(f"Second Evaluation time: {end - start:.6f} seconds")

