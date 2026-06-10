"""
predict_future.py - Predict future house prices using trained models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

plt.style.use('seaborn-v0_8-darkgrid')
os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)


def load_and_prepare_data(filepath='data/house_data.csv'):
    """Load data and prepare for trend analysis."""
    df = pd.read_csv(filepath, parse_dates=['sale_date'])
    
    # Aggregate by month to see trends
    df['year_month'] = df['sale_date'].dt.to_period('M').astype(str)
    monthly = df.groupby('year_month').agg({
        'price': ['mean', 'median', 'std', 'count'],
        'sqft': 'mean',
        'location_score': 'mean'
    }).round(2)
    monthly.columns = ['avg_price', 'median_price', 'price_std', 'num_sales', 'avg_sqft', 'avg_location']
    monthly = monthly.reset_index()
    monthly['year_month_dt'] = pd.to_datetime(monthly['year_month'] + '-01')
    monthly = monthly.sort_values('year_month_dt')
    
    return df, monthly


def analyze_price_trends(df, monthly):
    """Analyze historical price trends and seasonality."""
    print("\n" + "="*60)
    print("PRICE TREND ANALYSIS")
    print("="*60)
    
    # Overall trend
    print(f"\n📈 Historical Data Period:")
    print(f"   From: {df['sale_date'].min().strftime('%Y-%m-%d')}")
    print(f"   To:   {df['sale_date'].max().strftime('%Y-%m-%d')}")
    print(f"   Total sales: {len(df):,}")
    
    # Appreciation rate
    first_year = monthly[monthly['year_month_dt'] == monthly['year_month_dt'].min()]
    last_year = monthly[monthly['year_month_dt'] == monthly['year_month_dt'].max()]
    
    if len(first_year) > 0 and len(last_year) > 0:
        start_price = first_year['avg_price'].values[0]
        end_price = last_year['avg_price'].values[0]
        years_span = (monthly['year_month_dt'].max() - monthly['year_month_dt'].min()).days / 365.25
        
        if years_span > 0:
            annual_return = ((end_price / start_price) ** (1 / years_span) - 1) * 100
            print(f"\n💰 Price Appreciation:")
            print(f"   Start avg price: ${start_price:,.0f}")
            print(f"   Current avg price: ${end_price:,.0f}")
            print(f"   Annual appreciation: {annual_return:.1f}%")
            total_return = ((end_price - start_price) / start_price) * 100
            print(f"   Total return over {years_span:.1f} years: {total_return:.1f}%")
    
    # Seasonality
    df['month'] = df['sale_date'].dt.month
    monthly_avg = df.groupby('month')['price'].mean()
    peak_month = monthly_avg.idxmax()
    low_month = monthly_avg.idxmin()
    
    print(f"\n📅 Seasonality:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print(f"   Peak month: {month_names[peak_month-1]} (avg: ${monthly_avg.max():,.0f})")
    print(f"   Lowest month: {month_names[low_month-1]} (avg: ${monthly_avg.min():,.0f})")
    print(f"   Seasonal variation: ${monthly_avg.max() - monthly_avg.min():,.0f}")


def train_trend_model(monthly):
    """Train a time series trend model using month index."""
    # Use month index as X
    X = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly['avg_price'].values
    
    # Train linear trend
    lr = LinearRegression()
    lr.fit(X, y)
    trend_coef = lr.coef_[0]
    
    # Train Random Forest for non-linear trend
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)
    
    return lr, rf, trend_coef


def predict_future_prices(monthly, months_ahead=12):
    """Predict future house prices."""
    print("\n" + "="*60)
    print("FUTURE PRICE PREDICTION")
    print("="*60)
    
    # Train models on historical monthly data
    lr_model, rf_model, trend_coef = train_trend_model(monthly)
    
    last_date = monthly['year_month_dt'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                  periods=months_ahead, freq='MS')
    
    last_idx = len(monthly)
    future_indices = np.arange(last_idx, last_idx + months_ahead).reshape(-1, 1)
    
    # Linear prediction
    lr_predictions = lr_model.predict(future_indices)
    
    # RF prediction
    rf_predictions = rf_model.predict(future_indices)
    
    # Ensemble prediction (average)
    ensemble_preds = (lr_predictions + rf_predictions) / 2
    
    # Create future DataFrame
    future_df = pd.DataFrame({
        'date': future_dates,
        'linear_prediction': lr_predictions.round(2),
        'rf_prediction': rf_predictions.round(2),
        'ensemble_prediction': ensemble_preds.round(2)
    })
    
    # Price range predictions (confidence interval)
    std_dev = monthly['avg_price'].std()
    future_df['lower_bound'] = (future_df['ensemble_prediction'] - 1.96 * std_dev).round(2)
    future_df['upper_bound'] = (future_df['ensemble_prediction'] + 1.96 * std_dev).round(2)
    
    print(f"\n📊 Future Price Predictions (next {months_ahead} months):")
    print(f"{'Date':<15} {'Predicted Price':<20} {'Range':<25} {'Trend':<10}")
    print(f"{'-'*15} {'-'*20} {'-'*25} {'-'*10}")
    
    for i, row in future_df.iterrows():
        date_str = row['date'].strftime('%Y-%m')
        pred = row['ensemble_prediction']
        lower = row['lower_bound']
        upper = row['upper_bound']
        
        # Simple trend indicator
        if i > 0:
            prev = future_df.iloc[i-1]['ensemble_prediction']
            trend = "📈" if pred > prev else "📉" if pred < prev else "➡️"
        else:
            last_hist = monthly['avg_price'].iloc[-1]
            trend = "📈" if pred > last_hist else "📉"
        
        print(f" {date_str:<14} ${pred:<14,.0f}  ${lower:<10,.0f}-${upper:<10,.0f}  {trend:<8}")
    
    # Summary
    first_pred = future_df['ensemble_prediction'].iloc[0]
    last_pred = future_df['ensemble_prediction'].iloc[-1]
    change = ((last_pred - first_pred) / first_pred) * 100
    
    print(f"\n📈 Prediction Summary:")
    print(f"   1-month forecast:  ${first_pred:,.0f}")
    print(f"   {months_ahead}-month forecast: ${last_pred:,.0f}")
    print(f"   Expected change: {change:+.1f}%")
    print(f"   Monthly trend: ${trend_coef:,.2f}/month")
    
    # Save predictions
    future_df.to_csv('output/future_predictions.csv', index=False)
    print(f"\n💾 Predictions saved to: output/future_predictions.csv")
    
    return future_df


def plot_future_predictions(monthly, future_df):
    """Plot historical data and future predictions."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # 1. Historical + Future Price Trend
    axes[0].plot(monthly['year_month_dt'], monthly['avg_price'], 
                 'o-', color='#2E86AB', linewidth=2, markersize=5, label='Historical Avg Price')
    axes[0].fill_between(monthly['year_month_dt'], 
                          monthly['avg_price'] - monthly['price_std'],
                          monthly['avg_price'] + monthly['price_std'],
                          alpha=0.15, color='#2E86AB', label='±1 Std Dev')
    
    axes[0].plot(future_df['date'], future_df['ensemble_prediction'], 
                 's--', color='#A23B72', linewidth=2.5, markersize=6, label='Predicted (Ensemble)')
    axes[0].fill_between(future_df['date'], 
                          future_df['lower_bound'], future_df['upper_bound'],
                          alpha=0.2, color='#A23B72', label='95% Confidence Interval')
    
    axes[0].axvline(x=monthly['year_month_dt'].max(), color='red', 
                    linestyle=':', alpha=0.7, label='Prediction Start')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price ($)')
    axes[0].set_title('House Price Trend: Historical & Future Predictions')
    axes[0].legend(loc='upper left')
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Model comparison for future
    axes[1].plot(monthly['year_month_dt'], monthly['avg_price'], 
                 'o-', color='#2E86AB', linewidth=2, markersize=4, label='Historical')
    axes[1].plot(future_df['date'], future_df['linear_prediction'], 
                 '^--', color='#F18F01', linewidth=1.5, label='Linear Trend')
    axes[1].plot(future_df['date'], future_df['rf_prediction'], 
                 's--', color='#C73E1D', linewidth=1.5, label='Random Forest')
    axes[1].plot(future_df['date'], future_df['ensemble_prediction'], 
                 'o-', color='#A23B72', linewidth=2.5, label='Ensemble (Avg)')
    
    axes[1].axvline(x=monthly['year_month_dt'].max(), color='red', 
                    linestyle=':', alpha=0.7)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price ($)')
    axes[1].set_title('Future Predictions: Model Comparison')
    axes[1].legend(loc='upper left')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/future_predictions.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: output/future_predictions.png")


def plot_seasonality(df, monthly):
    """Plot seasonality patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Monthly seasonality
    df['month'] = df['sale_date'].dt.month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    monthly_avg = df.groupby('month')['price'].mean()
    monthly_std = df.groupby('month')['price'].std()
    
    axes[0].bar(month_names, monthly_avg.values, yerr=monthly_std.values, 
                color='#2E86AB', alpha=0.7, capsize=5)
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Average Price ($)')
    axes[0].set_title('Seasonal Price Pattern')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Number of sales over time
    axes[1].bar(monthly['year_month_dt'], monthly['num_sales'], 
                color='#A23B72', alpha=0.7, width=20)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Number of Sales')
    axes[1].set_title('Sales Volume Over Time')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/seasonality.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: output/seasonality.png")


def main():
    """Run full prediction pipeline."""
    print("="*60)
    print("HOUSE PRICE PREDICTION SYSTEM")
    print("="*60)
    
    df, monthly = load_and_prepare_data()
    analyze_price_trends(df, monthly)
    future_df = predict_future_prices(monthly, months_ahead=12)
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_future_predictions(monthly, future_df)
    plot_seasonality(df, monthly)
    
    print("\n" + "="*60)
    print("✅ Future prediction complete!")
    print("="*60)
    
    return future_df


if __name__ == "__main__":
    main()
