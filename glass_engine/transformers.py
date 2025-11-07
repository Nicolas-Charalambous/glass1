from lark import Lark, Transformer, v_args
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, List, Union
from glass_engine.constants import SYNTAX_PATH
from glass_engine.functions import FUNCTIONS


class SymbolResolver:
    """Handles resolution of symbols to actual timeseries data"""
    def __init__(self):
        self.data = {}
        self.subscribers = []
    
    def register_symbol(self, symbol: str, data: np.ndarray = None):
        """Register a symbol with its timeseries data"""
        self.data[symbol] = data or np.random.randn(100).cumsum() + 100
    
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

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)

    def on_message(self, msg):
        for subscriber in self.subscribers:
            subscriber.on_message(msg)

    def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to updates for a symbol"""
        pass
        

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
    
    def __init__(self, grammar_file: str):
        self.symbol_resolver = SymbolResolver()
        self.transformer = TimeseriesTransformer(self.symbol_resolver)
        self.evaluator = Lark.open(grammar_file, parser='lalr', transformer=self.transformer)
        self.parser = Lark.open(grammar_file, parser='lalr')

    
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
            result = self.evaluator.parse(expression)
            if isinstance(result, (int, float)):
                # Convert scalar results to arrays for consistency
                return np.array([result])
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")
        
    def parse(self, expression: str) -> Any:
        """Parse an expression to get the parse tree (without evaluation)"""
        return self.parser.parse(expression)
    
    def register_custom_function(self, name: str, func: Callable):
        """Register a custom function"""
        self.transformer.functions[name] = func

    def serialize(self, tree):
        if isinstance(tree, Token):
            return f"T({tree.type}:{tree.value})"
        elif isinstance(tree, Tree):
            children_repr = ','.join(self.serialize(child) for child in tree.children)
            return f"R({tree.data}:[{children_repr}])"
        else:
            return str(tree)