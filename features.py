#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 08:11:41 2025

@author: avinashreddykovvuri
"""

import pandas as pd

def create_features(df):
    """Create basic features for modeling"""
    df = df.copy()
    df = df.sort_values(['store_id', 'sku_id', 'date'])
    
    # Simple lag features
    df['units_lag_1'] = df.groupby(['store_id', 'sku_id'])['units_sold'].shift(1)
    df['units_lag_7'] = df.groupby(['store_id', 'sku_id'])['units_sold'].shift(7)
    df['units_ma_7'] = df.groupby(['store_id', 'sku_id'])['units_sold'].rolling(7).mean().values
    
    # Price features
    df['price_discount'] = (df['base_price'] - df['final_price']) / df['base_price']
    df['price_vs_competitor'] = df['final_price'] / df['competitor_price']
    
    # Time features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    return df

# Feature columns to use
FEATURE_COLS = [
    'final_price', 'promo_flag', 'promo_depth', 'competitor_price',
    'holiday_flag', 'weather_index', 'day_of_week', 'month',
    'price_discount', 'price_vs_competitor',
    'units_lag_1', 'units_lag_7', 'units_ma_7'
]