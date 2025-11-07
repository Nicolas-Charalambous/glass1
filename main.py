from typing import Dict, Union
import numpy as np
import logging
from glass_engine.transformers import TimeseriesTransformer, ExpressionEngine
from glass_engine.functions import FUNCTIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def test_transformer():
    #Create engine
    engine = ExpressionEngine(SYNTAX_PATH)
    
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

def test_parser():
    #Create engine
    engine = ExpressionEngine(SYNTAX_PATH)
    # Example expression
    expression = "realized_vol((mavg(AAPL.close,20) + rsi(GOOG,34)) / 210) + rand()"

    # Parse the expression to get the parse tree (not evaluated)
    parse_tree = engine.parse(expression)

    # Print the parse tree
    print(parse_tree.pretty())

# Example usage and testing
if __name__ == "__main__":
    test_parser()