"""
explore_data.py - Exploratory Data Analysis and Visualization for House Prices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Create output directory
os.makedirs('output', exist_ok=True)

def load_data(filepath='data/house_data.csv'):
    """Load the house price dataset."""
    df = pd.read_csv(filepath, parse_dates=['sale_date'])
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFirst 5 rows:")
    print(df.head())
    return df

def basic_stats(df):
    """Print basic statistics."""
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    print(f"\nDataset Info:")
    print(f"Total records: {df.shape[0]}")
    print(f"Features: {df.shape[1] - 1}")  # exclude price
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    print(f"\nNumerical Features Summary:")
    print(df.describe())
    
    print(f"\nCorrelation with Price:")
    correlations = df.select_dtypes(include=[np.number]).corr()['price'].sort_values(ascending=False)
    print(correlations)

def plot_price_distribution(df):
    """Plot the distribution of house prices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['price'], bins=50, color='#2E86AB', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Price ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of House Prices')
    axes[0].axvline(df['price'].mean(), color='red', linestyle='--', label=f"Mean: ${df['price'].mean():,.0f}")
    axes[0].axvline(df['price'].median(), color='green', linestyle='--', label=f"Median: ${df['price'].median():,.0f}")
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(df['price'], vert=True, patch_artist=True, 
                    boxprops=dict(facecolor='#2E86AB', alpha=0.7))
    axes[1].set_ylabel('Price ($)')
    axes[1].set_title('House Price Box Plot')
    axes[1].set_xticks([])
    
    plt.tight_layout()
    plt.savefig('output/price_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: output/price_distribution.png")

def plot_correlation_heatmap(df):
    """Plot correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/correlation_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: output/correlation_heatmap.png")

def plot_feature_relationships(df):
    """Plot relationship between key features and price."""
    features = ['sqft', 'bedrooms', 'bathrooms', 'house_age', 'location_score', 
                'school_rating', 'crime_rate', 'distance_to_city_miles']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        axes[i].scatter(df[feature], df['price'], alpha=0.4, s=10, c='#2E86AB')
        axes[i].set_xlabel(feature.replace('_', ' ').title())
        axes[i].set_ylabel('Price ($)')
        axes[i].set_title(f'Price vs {feature.replace("_", " ").title()}')
        
        # Add trend line
        z = np.polyfit(df[feature], df['price'], 1)
        p = np.poly1d(z)
        x_sorted = np.sort(df[feature])
        axes[i].plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('output/feature_relationships.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: output/feature_relationships.png")

def plot_price_over_time(df):
    """Plot house prices over time."""
    df_sorted = df.sort_values('sale_date')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Price trend over time
    axes[0].scatter(df_sorted['sale_date'], df_sorted['price'], alpha=0.3, s=8, c='#2E86AB')
    
    # Monthly average
    df_monthly = df_sorted.set_index('sale_date').resample('M')['price'].mean()
    axes[0].plot(df_monthly.index, df_monthly.values, 'r-', linewidth=2, label='Monthly Avg')
    axes[0].set_xlabel('Sale Date')
    axes[0].set_ylabel('Price ($)')
    axes[0].set_title('House Prices Over Time')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    
    # Box plot by year
    df['year'] = df['sale_date'].dt.year
    years = sorted(df['year'].unique())
    data_by_year = [df[df['year'] == y]['price'] for y in years]
    
    bp = axes[1].boxplot(data_by_year, labels=years, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#2E86AB')
        patch.set_alpha(0.7)
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Price ($)')
    axes[1].set_title('Price Distribution by Year')
    
    plt.tight_layout()
    plt.savefig('output/price_over_time.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: output/price_over_time.png")

def plot_multivariate_insights(df):
    """Plot multivariate insights."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Sqft vs Price colored by Bedrooms
    scatter = axes[0].scatter(
        df['sqft'], df['price'], 
        c=df['bedrooms'], cmap='viridis', 
        alpha=0.5, s=30
    )
    axes[0].set_xlabel('Square Footage')
    axes[0].set_ylabel('Price ($)')
    axes[0].set_title('Sqft vs Price (colored by Bedrooms)')
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Bedrooms')
    
    # Location Score vs School Rating, size = price
    scatter2 = axes[1].scatter(
        df['location_score'], df['school_rating'],
        s=df['price'] / 10000, alpha=0.4, c=df['house_age'], cmap='plasma'
    )
    axes[1].set_xlabel('Location Score')
    axes[1].set_ylabel('School Rating')
    axes[1].set_title('Location & Schools (size=price, color=house age)')
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('House Age (years)')
    
    plt.tight_layout()
    plt.savefig('output/multivariate_insights.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: output/multivariate_insights.png")

def run_all():
    """Run all EDA visualizations."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    df = load_data()
    basic_stats(df)
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_relationships(df)
    plot_price_over_time(df)
    plot_multivariate_insights(df)
    
    print("\n" + "="*60)
    print("✅ All EDA plots saved to 'output/' folder")
    print("="*60)
    
    return df

if __name__ == "__main__":
    run_all()
