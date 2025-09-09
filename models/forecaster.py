#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 08:13:56 2025

@author: avinashreddykovvuri
"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

class SimpleForecastModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    def train(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Get performance metrics"""
        pred = self.predict(X)
        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        mape = np.mean(np.abs((y - pred) / y)) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)