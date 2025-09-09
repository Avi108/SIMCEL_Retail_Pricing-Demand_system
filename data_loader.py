#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 22:17:07 2025

@author: avinashreddykovvuri
"""

import pandas as pd

def load_data(file_path='retail_pricing_demand_2024.csv'):
    """Simple data loader"""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_train_test(df):
    """Get train/test split using existing set column"""
    train = df[df['set'] == 'train'].copy()
    test = df[df['set'] == 'test'].copy()
    return train, test