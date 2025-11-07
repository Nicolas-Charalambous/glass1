from datetime import datetime
from lark import Token, Tree
from glass_engine.transformers import TimeseriesTransformer, ExpressionEngine
from glass_engine.constants import SYNTAX_PATH
import reactivex as rx

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenType:
    SYMBOL = 'SYMBOL'
    NUMBER = 'NUMBER'
    STRING = 'STRING'
    FIELD_ACCESS = 'FIELD_ACCESS'
    FUNC_CALL = 'FUNC_CALL'


class ExpressionFactory:

    pass

class ExpressionEvaluator:
    def __init__(self, expression_or_tree, engine=None):
        self.engine = engine or ExpressionEngine(SYNTAX_PATH)
        self.expression_tree = self.parse(expression_or_tree) if isinstance(expression_or_tree, str) else expression_or_tree
        self.root = self.expression_tree.data
        self.id = self.engine.serialize(self.expression_tree)
        self.stream = rx.subject.Subject()
    
    def parse(self, data):
        return self.engine.parse(self.expression)
    
    def get_historical(self, data):
        """Get historical data for the expression"""
        pass

    def get_tokens_of_type(self, tree, token_type):
        return [token for token in tree.scan_values(lambda v: isinstance(v, Token) and v.type == token_type)]

    def get_subtrees_of_type(self, tree, rule_name):
        return [subtree for subtree in tree.scan_values(lambda v: hasattr(v, 'data') and v.data == rule_name)]

    def get_deps(self, tree):
        """Get all dependencies (symbols) used in the expression"""
        symbols =  self.get_tokens_of_type(tree, TokenType.SYMBOL)
    
    def subscribe_to_deps(self, deps):
        """Subscribe to updates for all dependencies"""
        pass

    def make_plan(self, tree):
        tree = self.engine.parse(self.expression)
        for child in tree.children:
            pass

    def sub_for_live(self):
        if self.root.data == TokenType.SYMBOL:
            self.engine.symbol_resolver.subscribe(self.root.children[0].value, self.on_update)

    def on_update(self, data):
        def format_timestamp_now():
            return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.stream.pipe(
                rx.operators.debounce(1),
                rx.operators.map(lambda msg: f'[{format_timestamp_now()}] :{msg}')
            ).subscribe(
            lambda x: print(f"{format_timestamp_now()} Received:", x),
            lambda e: print("Error:", e),
            lambda: print("Stream completed")
            )
        


            
   


    # Example usage and testing
if __name__ == "__main__":
    # Create engine
    engine = ExpressionEngine(SYNTAX_PATH)

    # Register symbols
    engine.register_symbols({
        'AAPL.close': None,
        'AAPL.volume': None,
        'GOOGL.close': None,
    })
    
    # Test expressions
    test_expressions = [
        "AAPL.close + 10",
        "sma(AAPL.close, 20)",
        "AAPL.close + GOOGL.close",
        "sma(AAPL.close, 20) + rsi(GOOGL.close, 14)",
        "(AAPL.close + GOOGL.close) / 2",
    ]
    
    tree = engine.parse("sma(AAPL.close, 20) + rsi(GOOGL.close, 14)")

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

