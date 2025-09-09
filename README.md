# SIMCEL_Retail_Pricing-Demand_system
Retail Pricing &amp; Demand Forecasting System: An ML pipeline for dynamic pricing and demand forecasting for retail. It helps optimize prices for max revenue, units, or margins, and simulates pricing strategies. The model uses historical data and has a ~22% MAPE. Key features include historical demand, price discounts, and promotions.

A machine learning pipeline for demand forecasting and dynamic pricing
optimization for multi-store retail operations.
Overview
This system helps retailers optimize pricing strategies by:
●
Forecasting demand for products across stores
●
Optimizing prices to maximize revenue, units sold, or margins
●
Simulating the impact of pricing strategies
Dataset
The system works with retail data containing:
●
5 stores (S01-S05) and 3 SKUs (SKU001-SKU003)
●
Daily sales data with pricing, promotions, and external factors
●
Pre-split train/test data for model validation
Model Performance
●
Algorithm: Gradient Boosting Regressor
●
Performance: ~22% MAPE on test set
●
Key Features: Historical demand patterns, price discounts,
promotional depth
Business Insights
●
Promotional Impact: 65-115% demand increase with promotions
●
Price Elasticity: SKU003 most sensitive (-0.71), SKU001 least (-0.48)
●
Store Performance: Clear hierarchy with S05 outperforming S01 by
94%
Price Optimization Results
For SKU001 in Store S01:
●
Revenue Maximization: 20% discount increases revenue by 60%
●
Margin Maximization: 25% premium pricing maximizes per-unit profit
●
Trade-off: Cannot optimize revenue and margin simultaneously
Requirements
pandas
scikit-learn
numpy
joblib
Usage Examples
Training with Custom Date Range
python pipeline.py train --train-start 2024-01-01 --train-end 2024-09-30
Multi-Objective Optimization
python pipeline.py optimize --objective margin --out margin_
results.csv
Limitations
●
Assumes fixed cost structure ($5 per unit for calculations)
●
22% MAPE indicates room for improvement in forecast accuracy
●
No inventory constraints or competitive response modeling
●
Results based on historical patterns may not capture future market
changes
Future Enhancements
●
Incorporate inventory constraints
●
Add competitor response modeling
●
Implement hierarchical forecasting
●
Include customer behavior dynamics
This implementation demonstrates core ML pipeline capabilities for retail
pricing optimization while maintaining simplicity and interpretability.
