"""
generate_data.py - Generate a realistic synthetic house price dataset
Features: sqft, bedrooms, bathrooms, year_built, location_score, distance_to_city, crime_rate, school_rating, house_age
Target: price (in USD)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_house_data(n_samples=1200):
    """Generate a realistic synthetic house price dataset."""
    
    # Core features
    sqft = np.random.normal(1800, 600, n_samples).clip(600, 5000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.35, 0.30, 0.12, 0.03])
    bathrooms = np.where(bedrooms == 1, np.random.choice([1, 1.5], n_samples),
                         np.where(bedrooms == 2, np.random.choice([1, 1.5, 2], n_samples),
                                   np.where(bedrooms == 3, np.random.choice([1.5, 2, 2.5, 3], n_samples),
                                             np.where(bedrooms == 4, np.random.choice([2, 2.5, 3, 3.5, 4], n_samples),
                                                       np.where(bedrooms == 5, np.random.choice([3, 3.5, 4, 4.5, 5], n_samples),
                                                                3.0)))))
    
    year_built = np.random.randint(1950, 2025, n_samples)
    house_age = 2025 - year_built
    
    # Location features
    location_score = np.random.uniform(1, 10, n_samples)  # 1=bad, 10=great
    distance_to_city = np.random.exponential(8, n_samples).clip(0.5, 40)  # miles
    crime_rate = np.random.uniform(0.5, 10, n_samples)  # lower is better
    school_rating = np.random.uniform(1, 10, n_samples)  # 1=bad, 10=great
    
    # Additional features
    has_garage = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    has_garden = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    floors = np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.4, 0.1])
    
    # --- Price formula with realistic interactions ---
    base_price = 50000
    
    # Sqft contributes ~$150-250 per sqft
    sqft_contribution = sqft * np.random.uniform(150, 250)
    
    # Each bedroom adds value but with diminishing returns
    bedroom_contribution = bedrooms * 25000 - (bedrooms ** 2) * 2000
    
    # Bathrooms are valuable
    bathroom_contribution = bathrooms * 15000
    
    # Newer houses cost more (age penalty: ~$500-1500 per year)
    age_penalty = house_age * np.random.uniform(400, 1200)
    
    # Location score has exponential effect
    location_contribution = (location_score ** 1.5) * 15000
    
    # Distance to city (closer = more expensive, inverse relationship)
    distance_penalty = np.sqrt(distance_to_city) * 15000
    
    # Crime rate reduces price
    crime_penalty = crime_rate * 8000
    
    # School rating increases price
    school_contribution = (school_rating ** 1.3) * 8000
    
    # Garage adds value
    garage_contribution = has_garage * 15000
    
    # Garden adds value
    garden_contribution = has_garden * 10000
    
    # More floors = more expensive (up to a point)
    floor_contribution = floors * 8000
    
    # Interaction: good location + good school = premium
    location_school_interaction = (location_score * school_rating) * 500
    
    # Interaction: large house in good location = premium
    size_location_interaction = (sqft / 1000) * location_score * 8000
    
    # Interaction: new house + good location 
    new_house_premium = np.maximum(0, (25 - house_age) / 25) * location_score * 5000
    
    # Random noise
    noise = np.random.normal(0, 25000, n_samples)
    
    # Calculate final price
    price = (base_price + sqft_contribution + bedroom_contribution + 
             bathroom_contribution - age_penalty + location_contribution - 
             distance_penalty - crime_penalty + school_contribution + 
             garage_contribution + garden_contribution + floor_contribution + 
             location_school_interaction + size_location_interaction + 
             new_house_premium + noise)
    
    # Ensure no negative prices and cap at reasonable values
    price = np.maximum(price, 50000)
    price = np.minimum(price, 2000000)
    
    # --- Add a time/date dimension for future prediction ---
    # Generate sale dates over the past 5 years
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2025-06-01')
    sale_dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    sale_dates = pd.to_datetime(np.random.permutation(sale_dates))
    
    # Add yearly appreciation (~3-8% per year)
    years_since_2020 = (sale_dates.year - 2020) + (sale_dates.month / 12)
    appreciation_factor = 1 + (years_since_2020 * np.random.uniform(0.03, 0.08))
    price = price * appreciation_factor
    
    # Create DataFrame
    df = pd.DataFrame({
        'sqft': sqft.round(0).astype(int),
        'bedrooms': bedrooms.astype(int),
        'bathrooms': bathrooms,
        'year_built': year_built.astype(int),
        'house_age': house_age.astype(int),
        'location_score': location_score.round(2),
        'distance_to_city_miles': distance_to_city.round(2),
        'crime_rate': crime_rate.round(2),
        'school_rating': school_rating.round(2),
        'has_garage': has_garage.astype(int),
        'has_garden': has_garden.astype(int),
        'floors': floors.astype(int),
        'sale_date': sale_dates,
        'price': price.round(2)
    })
    
    return df

def save_data(df, filepath='data/house_data.csv'):
    """Save dataset to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"Mean price: ${df['price'].mean():,.0f}")
    print(f"Median price: ${df['price'].median():,.0f}")

if __name__ == "__main__":
    df = generate_house_data(1200)
    save_data(df, 'data/house_data.csv')
    print("\nFirst 5 rows:")
    print(df.head())
