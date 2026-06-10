"""
train_model.py - Train multiple ML models to predict house prices
Models: Linear Regression, Ridge, Random Forest, Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score
)

plt.style.use('seaborn-v0_8-darkgrid')
os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)


def load_data(filepath='data/house_data.csv'):
    """Load and prepare data for modeling."""
    df = pd.read_csv(filepath, parse_dates=['sale_date'])
    
    # Feature engineering
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    
    # Drop raw date column for modeling
    X = df.drop(['price', 'sale_date'], axis=1)
    y = df['price']
    
    # Convert boolean columns if any
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    return X, y, df


def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    """Train and evaluate a model, return metrics."""
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'Model': name,
        'Train R²': round(r2_score(y_train, y_train_pred), 4),
        'Test R²': round(r2_score(y_test, y_test_pred), 4),
        'Train MAE': round(mean_absolute_error(y_train, y_train_pred), 2),
        'Test MAE': round(mean_absolute_error(y_test, y_test_pred), 2),
        'Train RMSE': round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2),
        'Test RMSE': round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2),
        'Explained Var': round(explained_variance_score(y_test, y_test_pred), 4)
    }
    
    return metrics, model, y_test_pred


def train_models():
    """Train and compare multiple regression models."""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Load data
    X, y, df = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (L2)': Ridge(alpha=1.0),
        'Lasso (L1)': Lasso(alpha=0.01, max_iter=10000),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        )
    }
    
    results = []
    trained_models = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\n{'-'*50}")
        print(f"Training: {name}")
        
        # Use scaled data for linear models, raw for tree-based
        if name in ['Linear Regression', 'Ridge (L2)', 'Lasso (L1)']:
            metrics, trained_model, y_pred = evaluate_model(
                model, X_train_scaled, X_test_scaled, y_train, y_test, name
            )
        else:
            metrics, trained_model, y_pred = evaluate_model(
                model, X_train, X_test, y_train, y_test, name
            )
        
        results.append(metrics)
        trained_models[name] = trained_model
        predictions[name] = y_pred
        
        # Print metrics
        print(f"  Train R²: {metrics['Train R²']:.4f}")
        print(f"  Test R²:  {metrics['Test R²']:.4f}")
        print(f"  Test MAE: ${metrics['Test MAE']:,.2f}")
        print(f"  Test RMSE: ${metrics['Test RMSE']:,.2f}")
    
    # Results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test R²', ascending=False)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON (sorted by Test R²)")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('output/model_comparison.csv', index=False)
    print("\nResults saved to: output/model_comparison.csv")
    
    return results_df, trained_models, scaler, X_train, X_test, y_train, y_test, predictions, X.columns.tolist()


def plot_results(results_df, y_test, predictions):
    """Visualize model performance."""
    
    # 1. Model comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R² scores
    models_list = results_df['Model'].tolist()
    train_r2 = results_df['Train R²'].tolist()
    test_r2 = results_df['Test R²'].tolist()
    
    x = np.arange(len(models_list))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_r2, width, label='Train R²', color='#2E86AB', alpha=0.8)
    axes[0, 0].bar(x + width/2, test_r2, width, label='Test R²', color='#A23B72', alpha=0.8)
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models_list, rotation=20, ha='right')
    axes[0, 0].legend()
    axes[0, 0].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    
    # MAE comparison
    train_mae = results_df['Train MAE'].tolist()
    test_mae = results_df['Test MAE'].tolist()
    
    axes[0, 1].bar(x - width/2, train_mae, width, label='Train MAE', color='#2E86AB', alpha=0.8)
    axes[0, 1].bar(x + width/2, test_mae, width, label='Test MAE', color='#A23B72', alpha=0.8)
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('MAE ($)')
    axes[0, 1].set_title('Mean Absolute Error Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models_list, rotation=20, ha='right')
    axes[0, 1].legend()
    
    # Predicted vs Actual (best model)
    best_model = results_df.iloc[0]['Model']
    y_pred = predictions[best_model]
    
    axes[1, 0].scatter(y_test, y_pred, alpha=0.4, s=20, color='#2E86AB')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('Actual Price ($)')
    axes[1, 0].set_ylabel('Predicted Price ($)')
    axes[1, 0].set_title(f'Predicted vs Actual ({best_model})')
    axes[1, 0].legend()
    
    # Residuals (best model)
    residuals = y_test - y_pred
    axes[1, 1].scatter(y_pred, residuals, alpha=0.4, s=20, color='#A23B72')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Price ($)')
    axes[1, 1].set_ylabel('Residual ($)')
    axes[1, 1].set_title(f'Residual Plot ({best_model})')
    
    plt.tight_layout()
    plt.savefig('output/model_performance.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: output/model_performance.png")


def plot_feature_importance(trained_models, feature_names):
    """Plot feature importance from tree-based models."""
    if 'Random Forest' in trained_models:
        rf = trained_models['Random Forest']
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices][::-1], color='#2E86AB', alpha=0.8)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('output/feature_importance.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("Saved: output/feature_importance.png")


def save_best_model(trained_models, results_df, scaler):
    """Save the best performing model."""
    
    best_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_name]
    
    model_path = f'models/{best_name.lower().replace(" ", "_")}.pkl'
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"\n[OK] Best model saved: {model_path}")
    print(f"   Best model: {best_name}")
    print(f"   Test R2: {results_df.iloc[0]['Test R2']:.4f}")
    print(f"   Test MAE: ${results_df.iloc[0]['Test MAE']:,.2f}")
    
    return best_name


def main():
    """Run the full training pipeline."""
    results_df, trained_models, scaler, X_train, X_test, y_train, y_test, predictions, feature_names = train_models()
    
    print("\n" + "="*60)
    print("VISUALIZING RESULTS")
    print("="*60)
    
    plot_results(results_df, y_test, predictions)
    plot_feature_importance(trained_models, feature_names)
    
    best_name = save_best_model(trained_models, results_df, scaler)
    
    print("\n" + "="*60)
    print("✅ Model training complete!")
    print("="*60)
    
    return results_df, trained_models, scaler, best_name


if __name__ == "__main__":
    main()
)
