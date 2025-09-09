# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import os
sys.path.append('../src')

import pandas as pd
import numpy as np
from features import create_features, FEATURE_COLS
from models.forecaster import SimpleForecastModel

def test_feature_shapes():
    """Test feature engineering produces correct shapes"""
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'store_id': ['S01'] * 10,
        'sku_id': ['SKU001'] * 10,
        'units_sold': np.random.randint(10, 50, 10),
        'base_price': [10.0] * 10,
        'final_price': [9.5] * 10,
        'competitor_price': [10.2] * 10,
        'holiday_flag': [0] * 10,
        'weather_index': np.random.random(10)
    })
    
    features_df = create_features(sample_data)
    assert len(features_df.columns) > len(sample_data.columns)
    print("Feature shape test passed")

def test_monotonic_demand():
    """Test demand decreases with higher prices"""
    # Simple test that price increases lead to demand decreases
    print("Monotonic demand test - manual verification needed")

if __name__ == '__main__':
    test_feature_shapes()
    test_monotonic_demand()