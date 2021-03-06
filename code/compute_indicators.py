import pandas as pd
import numpy as np
import talib as ta
import ta as ta_ta

def technical_indicator(stock, name, func, normalize, *args, **kwargs):
    if 'normalize_by' not in kwargs:
        kwargs['normalize_by'] = args[0]
        
    stock.loc[:, name] = func(*args) / (kwargs['normalize_by'] if normalize else 1)
    
def technical_indicator_time_periods(stock, name, func, timeperiods, normalize=False, *args, **kwargs):
    if 'normalize_by' not in kwargs:
        kwargs['normalize_by'] = args[0]
    
    for tp in timeperiods:
        column = f'{name}_{tp}'
        stock.loc[:, column] = func(*args, timeperiod=tp) / (kwargs['normalize_by'] if normalize else 1)

def bollinger_bands(stock, timeperiods, close):
    for tp in timeperiods:
        column_upper = f'bb_upper_{tp}'
        column_middle = f'bb_middle_{tp}'
        column_lower = f'bb_lower_{tp}'
        column_width = f'bb_width_{tp}'
        
        upper, middle, lower = ta.BBANDS(close, timeperiod=tp)
        
        stock.loc[:, column_upper] = upper / close
        stock.loc[:, column_middle] = middle / close
        stock.loc[:, column_lower] = lower / close
        stock.loc[:, column_width] = (upper - lower) / middle
        
def double_exponential_moving_average(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'dema', ta.DEMA, timeperiods, True, close)

def exponential_moving_average(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'ema', ta.EMA, timeperiods, True, close)

def hilbert_transform_trendline(stock, close):
    technical_indicator(stock, 'ht_trendline', ta.HT_TRENDLINE, True, close)

def kaufman_adaptive_moving_average(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'kama', ta.KAMA, timeperiods, True, close)

def moving_average(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'ma', ta.MA, timeperiods, True, close)

def mesa_adaptive_moving_average(stock, close):
    mama, fama = ta.MAMA(close, fastlimit=0.9, slowlimit=0.1)

    stock.loc[:, 'mama'] = mama / close
    stock.loc[:, 'fama'] = fama / close

def midpoint_over_period(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'midpoint', ta.MIDPOINT, timeperiods, True, close)

def midpoint_price_over_period(stock, timeperiods, high, low, close):
    technical_indicator_time_periods(stock, 'midprice', ta.MIDPRICE, timeperiods, True, high, low, normalize_by=close)

def parabolic_sar(stock, high, low, close):
    technical_indicator(stock, 'sar', ta.SAR, True, high, low, normalize_by=close)

def triple_exponential_moving_average_t3(stock, timeperiods, close):
    for tp in timeperiods:
        column = f't3_{tp}'
        stock.loc[:, column] = ta.T3(close, timeperiod=tp, vfactor=0.7) / close

def triple_exponential_moving_average(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'tema', ta.TEMA, timeperiods, True, close)

def triangular_moving_average(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'trima', ta.TRIMA, timeperiods, True, close)

def weighted_moving_average(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'wma', ta.WMA, timeperiods, True, close)

## Momentum Indicators

def average_directional_movement_index(stock, timeperiods, high, low, close):
    technical_indicator_time_periods(stock, 'adx', ta.ADX, timeperiods, False, high, low, close)
    technical_indicator_time_periods(stock, 'adxr', ta.ADXR, timeperiods, False, high, low, close)

def absolute_price_oscillator(stock, close):
    technical_indicator(stock, 'apo', ta.APO, True, close)

def balance_of_power(stock, open_, high, low, close):
    technical_indicator(stock, 'bop', ta.BOP, False, open_, high, low, close)

def commodity_channel_index(stock, timeperiods, high, low, close):
    technical_indicator_time_periods(stock, 'cci', ta.CCI, timeperiods, False, high, low, close)

def chande_momentum_oscillator(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'cmo', ta.CMO, timeperiods, False, close)

def directional_movement(stock, timeperiods, high, low, close):
    technical_indicator_time_periods(stock, 'plus_dm', ta.PLUS_DM, timeperiods, True, high, low, normalize_by=close)
    technical_indicator_time_periods(stock, 'minus_dm', ta.MINUS_DM, timeperiods, True, high, low, normalize_by=close)
    technical_indicator_time_periods(stock, 'dx', ta.DX, timeperiods, False, high, low, close)

def directional_indicator(stock, timeperiods, high, low, close):
    technical_indicator_time_periods(stock, 'plus_di', ta.PLUS_DI, timeperiods, False, high, low, close)
    technical_indicator_time_periods(stock, 'minus_di', ta.MINUS_DI, timeperiods, False, high, low, close)

def moving_average_convergence_divergence(stock, close):
    line, hist, signal = ta.MACD(close)

    stock.loc[:, 'macd_line'] = line
    stock.loc[:, 'macd_hist'] = hist
    stock.loc[:, 'macd_signal'] = signal

def money_flow_index(stock, timeperiods, high, low, close, volume):
    technical_indicator_time_periods(stock, 'mfi', ta.MFI, timeperiods, False, high, low, close, volume)

def momentum(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'mom', ta.MOM, timeperiods, True, close)

def percentage_price_oscillator(stock, close):
    technical_indicator(stock, 'ppo', ta.PPO, True, close)

def rate_of_change(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'roc', ta.ROC, timeperiods, False, close)
    technical_indicator_time_periods(stock, 'rocp', ta.ROCP, timeperiods, False, close)
    technical_indicator_time_periods(stock, 'rocr', ta.ROCR, timeperiods, False, close)

def relative_strength_index(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'rsi', ta.RSI, timeperiods, False, close)

def stochastic(stock, high, low, close):
    slow_k, slow_d = ta.STOCH(high, low, close)
    rsi_k, rsi_d = ta.STOCHRSI(close)

    stock.loc[:, 'slow_k'] = slow_k
    stock.loc[:, 'slow_d'] = slow_d
    stock.loc[:, 'rsi_k'] = rsi_k
    stock.loc[:, 'rsi_d'] = rsi_d

def one_day_rate_of_change_of_a_triple_smooth_ema(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'trix', ta.TRIX, timeperiods, True, close)

def ultimate_oscillator(stock, high, low, close):
    technical_indicator(stock, 'ult_osc', ta.ULTOSC, False, high, low, close)

def williams_percent_r(stock, timeperiods, high, low, close):
    technical_indicator_time_periods(stock, 'will_r', ta.WILLR, timeperiods, False, high, low, close)

## Volume Indicators

def chaikin_ad(stock, high, low, close, volume):
    technical_indicator(stock, 'chaikin_ad_line', ta.AD, True, high, low, close, volume, normalize_by=volume)
    technical_indicator(stock, 'chaikin_ad_osc', ta.ADOSC, True, high, low, close, volume, normalize_by=volume)

def on_balance_volume(stock, close, volume):
    technical_indicator(stock, 'obv', ta.OBV, True, close, volume, normalize_by=volume)

## Volatility Indicators

def normalized_average_true_range(stock, timeperiods, high, low, close):
    technical_indicator_time_periods(stock, 'natr', ta.NATR, timeperiods, False, high, low, close)

def true_range(stock, high, low, close):
    technical_indicator(stock, 'trange', ta.TRANGE, True, high, low, close, normalize_by=close)

## Price Transform

def average_price(stock, open_, high, low, close):
    technical_indicator(stock, 'avg_price', ta.AVGPRICE, True, open_, high, low, close, normalize_by=close)

def median_price(stock, high, low, close):
    technical_indicator(stock, 'median_price', ta.MEDPRICE, True, high, low, normalize_by=close)

def typical_price(stock, high, low, close):
    technical_indicator(stock, 'typical_price', ta.TYPPRICE, True, high, low, close, normalize_by=close)

def weighted_close_price(stock, high, low, close):
    technical_indicator(stock, 'wcl_price', ta.WCLPRICE, True, high, low, close, normalize_by=close)

## Cycle Indicators

def dominant_cycle_period(stock, close):
    technical_indicator(stock, 'ht_dcperiod', ta.HT_DCPERIOD, False, close)

def dominant_cycle_phase(stock, close):
    technical_indicator(stock, 'ht_dcphase', ta.HT_DCPHASE, False, close)

def phasor_components(stock, close):
    inphase, quadrature = ta.HT_PHASOR(close)
    
    stock.loc[:, 'ht_phasor_inphase'] = inphase
    stock.loc[:, 'ht_phasor_quadrature'] = quadrature
    
def sine_wave(stock, close):
    sine, lead = ta.HT_SINE(close)
    
    stock.loc[:, 'ht_sine'] = sine
    stock.loc[:, 'ht_sine_lead'] = lead
    
def trend_mode(stock, close):
    technical_indicator(stock, 'ht_trendmode', ta.HT_TRENDMODE, False, close)

## Statistic Functions

def beta(stock, timeperiods, high, low):
    technical_indicator_time_periods(stock, 'beta', ta.BETA, timeperiods, False, high, low)

def pearsons_correlation_coefficient(stock, timeperiods, high, low):
    technical_indicator_time_periods(stock, 'correl', ta.CORREL, timeperiods, False, high, low)

def linear_regression(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'linear_regression', ta.LINEARREG, timeperiods, True, close)
    technical_indicator_time_periods(stock, 'linear_regression_angle', ta.LINEARREG_ANGLE, timeperiods, False, close)
    technical_indicator_time_periods(stock, 'linear_regression_intercept', ta.LINEARREG_INTERCEPT, timeperiods, True, close)
    technical_indicator_time_periods(stock, 'linear_regression_slope', ta.LINEARREG_SLOPE, timeperiods, True, close)

def standard_deviation(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'std', ta.STDDEV, timeperiods, True, close)

def time_series_forecast(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'tsf', ta.TSF, timeperiods, True, close)

def variance(stock, timeperiods, close):
    technical_indicator_time_periods(stock, 'var', ta.VAR, timeperiods, True, close)

def ichimoku_cloud(stock, high, low, close):
    a = ta_ta.trend.ichimoku_a(stock.high, stock.low) / stock.close
    b = ta_ta.trend.ichimoku_b(stock.high, stock.low) / stock.close
    base = ta_ta.trend.ichimoku_base_line(stock.high, stock.low) / stock.close
    conversion = ta_ta.trend.ichimoku_conversion_line(stock.high, stock.low) / stock.close
    cloud = a - b
    
    stock.loc[:, 'ichimoku_a'] = a
    stock.loc[:, 'ichimoku_b'] = b
    stock.loc[:, 'ichimoku_base'] = base
    stock.loc[:, 'ichimoku_conversion'] = conversion
    stock.loc[:, 'ichimoku_cloud'] = cloud

def predict_today(stock, columns, model, scaler):

    if stock.shape[0] == 0:
        return 0, None

    close = stock.loc[:, 'close']

    X = stock[columns]
    X_scaled = scaler.transform(X)

    return model.predict(X_scaled)[-1][1], close.iloc[-1]

def compute(stock):

    open_ = stock.loc[:, 'open']    
    high = stock.loc[:, 'high']
    low = stock.loc[:, 'low']
    close = stock.loc[:, 'close']
    volume = stock.loc[:, 'volume']
    
    timeperiods = [5, 20]
    
    # Overlap Studies
    
    bollinger_bands(stock, timeperiods, close)
        
    triple_exponential_moving_average(stock, timeperiods, close)
                        
#     triple_exponential_moving_average_t3(stock, timeperiods, close)
    
    parabolic_sar(stock, high, low, close)
    
    # Momentum Indicators
    
    average_directional_movement_index(stock, timeperiods, high, low, close)
             
    moving_average_convergence_divergence(stock, close)
    
    relative_strength_index(stock, timeperiods, close)
    
#     balance_of_power(stock, open_, high, low, close)
    
    williams_percent_r(stock, timeperiods, high, low, close)
        
    # Volume Indicators
        
    on_balance_volume(stock, close, volume)
    
    # Volatility Indicators
        
    normalized_average_true_range(stock, timeperiods, high, low, close)
    
    # Price Transform
    
    weighted_close_price(stock, high, low, close)
            
    # Statistic Functions
    
    beta(stock, timeperiods, high, low)
    
#     standard_deviation(stock, timeperiods, close)
            
    linear_regression(stock, timeperiods, close)
        
    time_series_forecast(stock, timeperiods, close)
    
    # Ichimoku Cloud
    
    ichimoku_cloud(stock, high, low, close)

    stock.dropna(inplace=True)