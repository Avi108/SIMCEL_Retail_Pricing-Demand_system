Retail Pricing & Demand Forecasting System: An ML pipeline for dynamic pricing and demand forecasting for retail. It helps optimize prices for max revenue, units, or margins, and simulates pricing strategies. The model uses historical data and has a ~22% MAPE. Key features include historical demand, price discounts, and promotions.


## Robustness & Causality

### Endogeneity Mitigation
- Uses lagged price features to reduce simultaneity bias
- Competitor prices as instrumental variables
- Stockout handling through demand censoring

### Stress Testing
- Competitor undercut scenarios (-5% pricing)
- Supply chain disruption (50% stock reduction)
- Economic downturn impact modeling

## Model Limitations
- 22% MAPE indicates moderate accuracy
- Assumes linear price elasticity
- No inventory constraints in optimization
- Historical patterns may not predict future disruptions
