from numba import njit
import numpy as np
from typing import Union



@njit(cache=True)
def simple_moving_average(data: np.ndarray, period: Union[int, float]) -> np.ndarray:
    """Simple Moving Average"""
    period = int(period)
    n = len(data)
    result = np.full(n, np.nan)
    if n < period:
        return result
    for i in range(period - 1, n):
        result[i] = np.mean(data[i - period + 1:i + 1])
    return result

@njit(cache=True)
def exponential_moving_average(data: np.ndarray, period: Union[int, float]) -> np.ndarray:
    """Exponential Moving Average"""
    period = int(period)
    n = len(data)
    result = np.empty(n)
    alpha = 2 / (period + 1)
    result[0] = data[0]
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result

@njit(cache=True)
def relative_strength_index(data: np.ndarray, period: Union[int, float]) -> np.ndarray:
    """Relative Strength Index"""
    period = int(period)
    n = len(data)
    result = np.full(n, np.nan)
    if n < period + 1:
        return result
    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.empty(n - 1)
    avg_loss = np.empty(n - 1)
    avg_gain[:period] = np.nan
    avg_loss[:period] = np.nan
    avg_gain[period - 1] = np.mean(gain[:period])
    avg_loss[period - 1] = np.mean(loss[:period])
    for i in range(period, n - 1):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i - 1] == 0:
            result[i] = 100.0
        else:
            rs = avg_gain[i - 1] / avg_loss[i - 1]
            result[i] = 100 - (100 / (1 + rs))
    return result

@njit(cache=True)
def rolling_std(data: np.ndarray, period: Union[int, float]) -> np.ndarray:
    """Rolling Standard Deviation"""
    period = int(period)
    n = len(data)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = np.std(data[i - period + 1:i + 1])
    return result

@njit(cache=True)
def rolling_min(data: np.ndarray, period: Union[int, float]) -> np.ndarray:
    """Rolling Minimum"""
    period = int(period)
    n = len(data)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = np.min(data[i - period + 1:i + 1])
    return result

@njit(cache=True)
def rolling_max(data: np.ndarray, period: Union[int, float]) -> np.ndarray:
    """Rolling Maximum"""
    period = int(period)
    n = len(data)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = np.max(data[i - period + 1:i + 1])
    return result

@njit(cache=True)
def rolling_sum(data: np.ndarray, period: Union[int, float]) -> np.ndarray:
    """Rolling Sum"""
    period = int(period)
    n = len(data)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = np.sum(data[i - period + 1:i + 1])
    return result

@njit(cache=True)
def rolling_mean(data: np.ndarray, period: Union[int, float]) -> np.ndarray:
    """Rolling Mean (alias for SMA)"""
    return simple_moving_average(data, period)

FUNCTIONS = {
            'sma': simple_moving_average,
            'rsi': relative_strength_index,
            'ema': exponential_moving_average,
            'std': rolling_std,
            'min': rolling_min,
            'max': rolling_max,
            'sum': rolling_sum,
            'mean': rolling_mean,
            'rolling_mean': rolling_mean,
}