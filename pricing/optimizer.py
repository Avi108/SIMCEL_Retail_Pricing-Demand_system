#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 08:16:04 2025

@author: avinashreddykovvuri
"""

import pandas as pd
import numpy as np

def optimize_prices(model, data, feature_cols, objective='revenue'):
    """Simple price optimization"""
    results = []
    
    # Test price levels from 80% to 120% of base
    for multiplier in np.arange(0.8, 1.21, 0.05):
        test_data = data.copy()
        
        # Update prices
        test_data['final_price'] = test_data['base_price'] * multiplier
        test_data['promo_flag'] = 1 if multiplier < 1.0 else 0
        test_data['promo_depth'] = max(0, 1 - multiplier)
        test_data['price_discount'] = 1 - multiplier
        test_data['price_vs_competitor'] = test_data['final_price'] / test_data['competitor_price']
        
        # Predict and calculate metrics
        demand = model.predict(test_data[feature_cols])
        revenue = demand * test_data['final_price']
        margin = demand * (test_data['final_price'] - 5)  # assume cost=5
        
        results.append({
            'price_mult': multiplier,
            'avg_price': test_data['final_price'].mean(),
            'total_demand': demand.sum(),
            'total_revenue': revenue.sum(),
            'total_margin': margin.sum()
        })
    
    df = pd.DataFrame(results)
    
    # Find best based on objective
    if objective == 'revenue':
        best_idx = df['total_revenue'].idxmax()
    elif objective == 'margin':
        best_idx = df['total_margin'].idxmax()
    else:  # units
        best_idx = df['total_demand'].idxmax()
    
    return df, df.loc[best_idx]

def simulate_plan(model, data, price_plan, feature_cols):
    """Simulate a given price plan"""
    results = []
    
    for _, row in price_plan.iterrows():
        # Find matching data
        mask = ((data['store_id'] == row['store_id']) & 
                (data['sku_id'] == row['sku_id']))
        
        if not mask.any():
            continue
            
        sim_data = data[mask].copy()
        sim_data['final_price'] = row['price']
        sim_data['price_discount'] = (sim_data['base_price'] - row['price']) / sim_data['base_price']
        sim_data['price_vs_competitor'] = row['price'] / sim_data['competitor_price']
        
        demand = model.predict(sim_data[feature_cols])
        revenue = demand * row['price']
        
        results.append({
            'store_id': row['store_id'],
            'sku_id': row['sku_id'],
            'price': row['price'],
            'demand': demand[0],
            'revenue': revenue[0]
        })
    
    return pd.DataFrame(results)
