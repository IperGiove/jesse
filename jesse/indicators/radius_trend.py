from collections import namedtuple
import numpy as np
from numba import njit
from jesse.helpers import slice_candles

RadiusTrend = namedtuple('RadiusTrend', ['trend', 'main_band', 'outer_band'])

@njit
def radius_trend_numba(high, low, close, step=0.15, multi=2.0, n=3):
    """
    Calculate Radius Trend indicator using Numba for performance
    """
    size = len(close)
    
    # Initialize output arrays
    trend_arr = np.zeros(size, dtype=np.bool_)
    band_arr = np.full(size, np.nan)
    main_band = np.full(size, np.nan)
    outer_band = np.full(size, np.nan)
    
    # Calculate high-low range and its SMA
    hl_range = np.abs(high - low)
    range_sma100 = np.full(size, np.nan)
    for i in range(99, size):
        range_sma100[i] = np.mean(hl_range[i-99:i+1])
    
    # Calculate hl2 SMA for reference
    hl2 = (high + low) / 2
    hl2_sma20 = np.full(size, np.nan)
    for i in range(19, size):
        hl2_sma20[i] = np.mean(hl2[i-19:i+1])
    
    # Initialize multipliers for steps
    multi1 = 0.0
    multi2 = 0.0
    
    # Initialize at bar 101 if we have enough data
    if size > 101:
        trend_arr[101] = True
        band_arr[101] = low[101] * 0.8
        
        # Main calculation loop
        for i in range(102, size):
            # Copy previous state
            trend_arr[i] = trend_arr[i-1]
            band_arr[i] = band_arr[i-1]
            
            # Skip if we don't have SMA values yet
            if np.isnan(range_sma100[i]):
                continue
                
            # Calculate distances for band placement
            distance = range_sma100[i] * multi
            distance1 = range_sma100[i] * 0.2
            
            # Update trend based on price position relative to band
            if close[i] < band_arr[i]:
                trend_arr[i] = False
            if close[i] > band_arr[i]:
                trend_arr[i] = True
            
            # Adjust band on trend changes
            if trend_arr[i-1] == False and trend_arr[i] == True:
                band_arr[i] = low[i] - distance
            if trend_arr[i-1] == True and trend_arr[i] == False:
                band_arr[i] = high[i] + distance
            
            # Apply step angle to trend lines
            if i % n == 0:
                if trend_arr[i]:
                    multi1 = 0.0
                    multi2 += step
                    band_arr[i] += distance1 * multi2
                else:
                    multi1 += step
                    multi2 = 0.0
                    band_arr[i] -= distance1 * multi1
    
    # Calculate smooth band (n-period SMA of band)
    for i in range(n-1, size):
        if np.all(~np.isnan(band_arr[i-n+1:i+1])):
            main_band[i] = np.mean(band_arr[i-n+1:i+1])
    
    # Calculate upper and lower bands
    band_upper = np.full(size, np.nan)
    band_lower = np.full(size, np.nan)
    
    for i in range(n-1, size):
        if np.isnan(range_sma100[i]) or np.isnan(band_arr[i]):
            continue
            
        # Calculate upper and lower bands with SMA smoothing
        upper_values = band_arr[i-n+1:i+1] + range_sma100[i] * 0.5
        lower_values = band_arr[i-n+1:i+1] - range_sma100[i] * 0.5
        
        if np.all(~np.isnan(upper_values)) and np.all(~np.isnan(lower_values)):
            band_upper[i] = np.mean(upper_values)
            band_lower[i] = np.mean(lower_values)
    
    # Set outer band based on trend
    for i in range(size):
        if not np.isnan(band_upper[i]) and not np.isnan(band_lower[i]):
            outer_band[i] = band_upper[i] if trend_arr[i] else band_lower[i]
    
    return trend_arr, main_band, outer_band

def radius_trend(candles, step=0.15, multi=2.0, n=3, sequential=False):
    """
    RadiusTrend Indicator ported from TradingView
    
    This indicator helps identify trend direction and potential reversal points
    by creating an adaptive band that follows price movement.
    
    Args:
        candles: np.ndarray - OHLCV candles
        step: float - Radius Step (default: 0.15)
        multi: float - Start Points Distance (default: 2.0)
        n: int - Step period (default: 3)
        sequential: bool - Return sequential data (default: False)
        
    Returns:
        RadiusTrend(trend, main_band, outer_band)
    """
    # Ensure candles are in the right format
    candles = slice_candles(candles, sequential)
    
    candles_high = candles[:, 3]
    candles_low = candles[:, 4]
    candles_close = candles[:, 2]
    
    trend, main_band, outer_band = radius_trend_numba(
        candles_high, candles_low, candles_close, step, multi, n
    )
    
    if sequential:
        return RadiusTrend(trend, main_band, outer_band)
    else:
        return RadiusTrend(
            bool(trend[-1]), 
            main_band[-1], 
            outer_band[-1]
        )