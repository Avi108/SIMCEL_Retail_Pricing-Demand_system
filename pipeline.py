import argparse
import pandas as pd
import sys
import os

sys.path.append('src')

from data_loader import load_data, get_train_test
from features import create_features, FEATURE_COLS
from models.forecaster import SimpleForecastModel
from pricing.optimizer import optimize_prices, simulate_plan

def train_model(args):
    print("Loading data and training model...")
    df = load_data(args.data_path)
    df = create_features(df)
    train_df, test_df = get_train_test(df)
    
    train_clean = train_df.dropna(subset=FEATURE_COLS + ['units_sold'])
    test_clean = test_df.dropna(subset=FEATURE_COLS + ['units_sold'])
    
    X_train = train_clean[FEATURE_COLS]
    y_train = train_clean['units_sold']
    X_test = test_clean[FEATURE_COLS]
    y_test = test_clean['units_sold']
    
    model = SimpleForecastModel()
    model.train(X_train, y_train)
    
    metrics = model.evaluate(X_test, y_test)
    print(f"Model Performance: MAPE = {metrics['MAPE']:.1f}%")
    
    os.makedirs('models', exist_ok=True)
    model.save('models/demand_model.pkl')
    print("Model saved!")

def optimize_pricing(args):
    print(f"Optimizing for {args.objective}...")
    model = SimpleForecastModel()
    model.load('models/demand_model.pkl')
    
    df = load_data(args.data_path)
    df = create_features(df)
    latest_data = df[df['date'] == df['date'].max()]
    
    all_results = []
    
    for store in latest_data['store_id'].unique():
        for sku in latest_data['sku_id'].unique():
            subset = latest_data[(latest_data['store_id'] == store) & 
                               (latest_data['sku_id'] == sku)]
            
            if len(subset) == 0:
                continue
                
            results_df, best = optimize_prices(model, subset, FEATURE_COLS, args.objective)
            
            all_results.append({
                'store_id': store,
                'sku_id': sku,
                'optimal_price': best['avg_price'],
                'expected_revenue': best['total_revenue'],
                'expected_demand': best['total_demand']
            })
    
    pd.DataFrame(all_results).to_csv(args.out, index=False)
    print(f"Results saved to {args.out}")

def simulate_prices(args):
    print("Simulating price plan...")
    model = SimpleForecastModel()
    model.load('models/demand_model.pkl')
    
    df = load_data(args.data_path)
    df = create_features(df)
    
    # Load price plan
    price_plan = pd.read_csv(args.price_plan)
    
    # Use your existing simulate_plan function
    results = simulate_plan(model, df, price_plan, FEATURE_COLS)
    
    # Save results
    results.to_csv(args.out, index=False)
    print(f"Simulation results saved to {args.out}")
    print(f"Total predicted revenue: ${results['revenue'].sum():.2f}")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--data-path', default='retail_pricing_demand_2024.csv')
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize')
    opt_parser.add_argument('--data-path', default='retail_pricing_demand_2024.csv')
    opt_parser.add_argument('--objective', choices=['revenue', 'margin', 'units'], default='revenue')
    opt_parser.add_argument('--out', required=True)
    
    # Simulate command (THIS WAS MISSING)
    sim_parser = subparsers.add_parser('simulate')
    sim_parser.add_argument('--data-path', default='retail_pricing_demand_2024.csv')
    sim_parser.add_argument('--price-plan', required=True)
    sim_parser.add_argument('--out', required=True)
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'optimize':
        optimize_pricing(args)
    elif args.command == 'simulate':  # THIS WAS MISSING
        simulate_prices(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()