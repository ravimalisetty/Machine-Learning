"""
Generate Realistic E-Commerce Customer Dataset for CLV Prediction
This script creates a synthetic but realistic dataset for Customer Lifetime Value analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers
n_customers = 1000

# Generate customer data
data = {
    'customer_id': range(1, n_customers + 1),
    
    # Customer acquisition date (last 3 years)
    'acquisition_date': [
        datetime.now() - timedelta(days=np.random.randint(1, 1095))
        for _ in range(n_customers)
    ],
    
    # Number of purchases
    'num_purchases': np.random.poisson(8, n_customers) + 1,
    
    # Average order value (following realistic distribution)
    'avg_order_value': np.random.gamma(shape=2, scale=50, size=n_customers) + 20,
    
    # Days since last purchase
    'days_since_last_purchase': np.random.exponential(scale=45, size=n_customers),
    
    # Customer age
    'customer_age': np.random.normal(35, 12, n_customers).astype(int).clip(18, 70),
    
    # Product categories purchased (0-5)
    'product_categories': np.random.poisson(2.5, n_customers).clip(1, 5),
    
    # Email engagement rate (0-100%)
    'email_engagement_rate': np.random.beta(2, 5, n_customers) * 100,
    
    # Returns count
    'returns_count': np.random.poisson(0.5, n_customers),
    
    # Support tickets
    'support_tickets': np.random.poisson(0.8, n_customers),
    
    # Mobile app user (0 or 1)
    'is_mobile_user': np.random.binomial(1, 0.6, n_customers),
    
    # Newsletter subscriber (0 or 1)
    'is_newsletter_subscriber': np.random.binomial(1, 0.7, n_customers),
    
    # Premium member (0 or 1)
    'is_premium_member': np.random.binomial(1, 0.3, n_customers),
}

df = pd.DataFrame(data)

# Calculate account age in days
df['account_age_days'] = (datetime.now() - df['acquisition_date']).dt.days

# Calculate total spend (with correlation to purchases and avg order value)
df['total_spend'] = (
    df['num_purchases'] * df['avg_order_value'] * 
    np.random.uniform(0.85, 1.15, n_customers)
)

# Add bonus for premium members
df.loc[df['is_premium_member'] == 1, 'total_spend'] *= 1.3

# Calculate Customer Lifetime Value (TARGET VARIABLE)
df['customer_lifetime_value'] = df['total_spend']

# Adjust for recency
recency_factor = 1 + (1 / (1 + df['days_since_last_purchase'] / 30))
df['customer_lifetime_value'] *= recency_factor

# Adjust for frequency
frequency_factor = 1 + (df['num_purchases'] / 20)
df['customer_lifetime_value'] *= frequency_factor

# Adjust for engagement
engagement_factor = 1 + (df['email_engagement_rate'] / 200)
df['customer_lifetime_value'] *= engagement_factor

# Adjust for premium members
df.loc[df['is_premium_member'] == 1, 'customer_lifetime_value'] *= 1.2

# Penalize for returns and support tickets
df['customer_lifetime_value'] *= (1 - df['returns_count'] * 0.05)
df['customer_lifetime_value'] *= (1 - df['support_tickets'] * 0.03)

# Add some random noise
df['customer_lifetime_value'] *= np.random.uniform(0.9, 1.1, n_customers)

# Round values
df['avg_order_value'] = df['avg_order_value'].round(2)
df['total_spend'] = df['total_spend'].round(2)
df['customer_lifetime_value'] = df['customer_lifetime_value'].round(2)
df['days_since_last_purchase'] = df['days_since_last_purchase'].round(0).astype(int)
df['email_engagement_rate'] = df['email_engagement_rate'].round(2)

# Drop acquisition_date
df = df.drop('acquisition_date', axis=1)

# Reorder columns
columns_order = [
    'customer_id', 'account_age_days', 'customer_age', 'num_purchases',
    'avg_order_value', 'total_spend', 'days_since_last_purchase',
    'product_categories', 'email_engagement_rate', 'returns_count',
    'support_tickets', 'is_mobile_user', 'is_newsletter_subscriber',
    'is_premium_member', 'customer_lifetime_value'
]

df = df[columns_order]

# Save to CSV
df.to_csv('ecommerce_customer_data.csv', index=False)

print("Dataset created successfully!")
print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())