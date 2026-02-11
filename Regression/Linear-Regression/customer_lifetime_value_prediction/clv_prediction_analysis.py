"""
E-Commerce Customer Lifetime Value Prediction using Linear Regression
========================================================================

Project: Predict customer lifetime value to optimize marketing spend and retention strategies
Author: [Your Name]
Date: February 2026

Business Context:
-----------------
Customer Lifetime Value (CLV) represents the total revenue a business expects from a customer
throughout their relationship. Accurate CLV prediction helps:
- Optimize marketing budget allocation
- Identify high-value customers for retention programs
- Personalize customer experiences
- Make data-driven acquisition decisions

Dataset Features:
-----------------
- account_age_days: How long customer has been with the company
- customer_age: Customer's age
- num_purchases: Total number of purchases made
- avg_order_value: Average amount spent per order
- total_spend: Total amount spent to date
- days_since_last_purchase: Recency metric
- product_categories: Number of different product categories purchased
- email_engagement_rate: Percentage of emails opened/clicked
- returns_count: Number of returned items
- support_tickets: Number of customer service interactions
- is_mobile_user: Whether customer uses mobile app
- is_newsletter_subscriber: Newsletter subscription status
- is_premium_member: Premium membership status
- customer_lifetime_value: TARGET - Predicted future value
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("CUSTOMER LIFETIME VALUE PREDICTION PROJECT")
print("=" * 80)
print("\n")

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("STEP 1: Loading Dataset...")
print("-" * 80)

df = pd.read_csv('ecommerce_customer_data.csv')

print(f"\nDataset Shape: {df.shape[0]} customers, {df.shape[1]} features")
print("\nFirst 5 rows:")
print(df.head())

print("\n\nDataset Information:")
print(df.info())

print("\n\nMissing Values:")
print(df.isnull().sum())

print("\n\nBasic Statistics:")
print(df.describe())

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 2: Exploratory Data Analysis")
print("=" * 80)

# Create visualizations directory
import os
os.makedirs('visualizations', exist_ok=True)

# 2.1 Distribution of Target Variable
print("\n2.1 Analyzing Customer Lifetime Value Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['customer_lifetime_value'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Customer Lifetime Value ($)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Customer Lifetime Value', fontsize=14, fontweight='bold')
axes[0].axvline(df['customer_lifetime_value'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: ${df["customer_lifetime_value"].mean():.2f}')
axes[0].axvline(df['customer_lifetime_value'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: ${df["customer_lifetime_value"].median():.2f}')
axes[0].legend()

# Box plot
axes[1].boxplot(df['customer_lifetime_value'], vert=True)
axes[1].set_ylabel('Customer Lifetime Value ($)', fontsize=12)
axes[1].set_title('CLV Distribution - Box Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/01_target_distribution.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/01_target_distribution.png")
plt.close()

# 2.2 Key Metrics Analysis
print("\n2.2 Analyzing Key Business Metrics...")
key_stats = {
    'Average CLV': f"${df['customer_lifetime_value'].mean():.2f}",
    'Median CLV': f"${df['customer_lifetime_value'].median():.2f}",
    'Total CLV': f"${df['customer_lifetime_value'].sum():.2f}",
    'Std Dev': f"${df['customer_lifetime_value'].std():.2f}",
    'Min CLV': f"${df['customer_lifetime_value'].min():.2f}",
    'Max CLV': f"${df['customer_lifetime_value'].max():.2f}",
}

for metric, value in key_stats.items():
    print(f"   {metric}: {value}")

# 2.3 Correlation Analysis
print("\n2.3 Analyzing Feature Correlations...")
# Select numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('customer_id')

correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('visualizations/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/02_correlation_heatmap.png")
plt.close()

# Top correlations with CLV
clv_correlations = correlation_matrix['customer_lifetime_value'].sort_values(ascending=False)
print("\n   Top Features Correlated with CLV:")
for feature, corr in clv_correlations.head(6).items():
    if feature != 'customer_lifetime_value':
        print(f"      • {feature}: {corr:.3f}")

# 2.4 Key Feature Relationships
print("\n2.4 Visualizing Key Relationships...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Total Spend vs CLV
axes[0, 0].scatter(df['total_spend'], df['customer_lifetime_value'], alpha=0.5)
axes[0, 0].set_xlabel('Total Spend ($)', fontsize=11)
axes[0, 0].set_ylabel('Customer Lifetime Value ($)', fontsize=11)
axes[0, 0].set_title('Total Spend vs CLV', fontsize=12, fontweight='bold')

# Plot 2: Number of Purchases vs CLV
axes[0, 1].scatter(df['num_purchases'], df['customer_lifetime_value'], alpha=0.5)
axes[0, 1].set_xlabel('Number of Purchases', fontsize=11)
axes[0, 1].set_ylabel('Customer Lifetime Value ($)', fontsize=11)
axes[0, 1].set_title('Purchase Frequency vs CLV', fontsize=12, fontweight='bold')

# Plot 3: Account Age vs CLV
axes[0, 2].scatter(df['account_age_days'], df['customer_lifetime_value'], alpha=0.5)
axes[0, 2].set_xlabel('Account Age (days)', fontsize=11)
axes[0, 2].set_ylabel('Customer Lifetime Value ($)', fontsize=11)
axes[0, 2].set_title('Account Age vs CLV', fontsize=12, fontweight='bold')

# Plot 4: Email Engagement vs CLV
axes[1, 0].scatter(df['email_engagement_rate'], df['customer_lifetime_value'], alpha=0.5)
axes[1, 0].set_xlabel('Email Engagement Rate (%)', fontsize=11)
axes[1, 0].set_ylabel('Customer Lifetime Value ($)', fontsize=11)
axes[1, 0].set_title('Email Engagement vs CLV', fontsize=12, fontweight='bold')

# Plot 5: Premium Members vs Non-Premium
premium_data = [
    df[df['is_premium_member'] == 0]['customer_lifetime_value'],
    df[df['is_premium_member'] == 1]['customer_lifetime_value']
]
axes[1, 1].boxplot(premium_data, labels=['Non-Premium', 'Premium'])
axes[1, 1].set_ylabel('Customer Lifetime Value ($)', fontsize=11)
axes[1, 1].set_title('Premium Membership Impact on CLV', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Days Since Last Purchase vs CLV
axes[1, 2].scatter(df['days_since_last_purchase'], df['customer_lifetime_value'], alpha=0.5)
axes[1, 2].set_xlabel('Days Since Last Purchase', fontsize=11)
axes[1, 2].set_ylabel('Customer Lifetime Value ($)', fontsize=11)
axes[1, 2].set_title('Recency vs CLV', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/03_feature_relationships.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/03_feature_relationships.png")
plt.close()

# 2.5 Customer Segmentation Insights
print("\n2.5 Customer Segmentation Insights...")
print(f"   • Premium Members: {df['is_premium_member'].sum()} ({df['is_premium_member'].sum()/len(df)*100:.1f}%)")
print(f"   • Mobile App Users: {df['is_mobile_user'].sum()} ({df['is_mobile_user'].sum()/len(df)*100:.1f}%)")
print(f"   • Newsletter Subscribers: {df['is_newsletter_subscriber'].sum()} ({df['is_newsletter_subscriber'].sum()/len(df)*100:.1f}%)")

premium_avg_clv = df[df['is_premium_member'] == 1]['customer_lifetime_value'].mean()
non_premium_avg_clv = df[df['is_premium_member'] == 0]['customer_lifetime_value'].mean()
print(f"\n   • Average CLV - Premium Members: ${premium_avg_clv:.2f}")
print(f"   • Average CLV - Non-Premium: ${non_premium_avg_clv:.2f}")
print(f"   • Premium Member Value Multiplier: {premium_avg_clv/non_premium_avg_clv:.2f}x")

# ============================================================================
# STEP 3: DATA PREPARATION
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 3: Data Preparation for Modeling")
print("=" * 80)

# 3.1 Feature Selection
print("\n3.1 Selecting Features...")
features = [col for col in df.columns if col not in ['customer_id', 'customer_lifetime_value']]
X = df[features]
y = df['customer_lifetime_value']

print(f"   Selected {len(features)} features for modeling:")
for i, feature in enumerate(features, 1):
    print(f"      {i}. {feature}")

# 3.2 Train-Test Split
print("\n3.2 Splitting Data into Training and Testing Sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"   • Training Set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"   • Testing Set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

# 3.3 Feature Scaling
print("\n3.3 Standardizing Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✓ Features standardized using StandardScaler")

# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 4: Training Linear Regression Model")
print("=" * 80)

# Train the model
model = LinearRegression()
print("\nTraining model...")
model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully!")

# Display model coefficients
print("\n4.1 Model Coefficients (Feature Importance):")
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

for idx, row in coefficients.iterrows():
    impact = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"   {impact} {row['Feature']}: {row['Coefficient']:.2f}")

print(f"\n   Intercept: {model.intercept_:.2f}")

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 5: Model Evaluation")
print("=" * 80)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n5.1 Performance Metrics:")
print("\n   Training Set:")
print(f"      • R² Score: {train_r2:.4f} ({train_r2*100:.2f}% variance explained)")
print(f"      • RMSE: ${train_rmse:.2f}")
print(f"      • MAE: ${train_mae:.2f}")

print("\n   Testing Set:")
print(f"      • R² Score: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
print(f"      • RMSE: ${test_rmse:.2f}")
print(f"      • MAE: ${test_mae:.2f}")

# Check for overfitting
overfit_diff = train_r2 - test_r2
if overfit_diff < 0.05:
    print(f"\n   ✓ Model shows good generalization (difference: {overfit_diff:.4f})")
elif overfit_diff < 0.1:
    print(f"\n   ⚠ Model shows slight overfitting (difference: {overfit_diff:.4f})")
else:
    print(f"\n   ⚠ Model shows significant overfitting (difference: {overfit_diff:.4f})")

# Visualize predictions vs actual
print("\n5.2 Creating Prediction Visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Training set predictions
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=50)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual CLV ($)', fontsize=12)
axes[0].set_ylabel('Predicted CLV ($)', fontsize=12)
axes[0].set_title(f'Training Set Predictions\n(R² = {train_r2:.4f})', 
                  fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Testing set predictions
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=50, color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual CLV ($)', fontsize=12)
axes[1].set_ylabel('Predicted CLV ($)', fontsize=12)
axes[1].set_title(f'Testing Set Predictions\n(R² = {test_r2:.4f})', 
                  fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/04_prediction_results.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/04_prediction_results.png")
plt.close()

# Residual analysis
print("\n5.3 Creating Residual Analysis...")
residuals = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Residual plot
axes[0].scatter(y_test_pred, residuals, alpha=0.5, s=50)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted CLV ($)', fontsize=12)
axes[0].set_ylabel('Residuals ($)', fontsize=12)
axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Residual distribution
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(residuals.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: ${residuals.mean():.2f}')
axes[1].set_xlabel('Residuals ($)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/05_residual_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/05_residual_analysis.png")
plt.close()

# ============================================================================
# STEP 6: BUSINESS INSIGHTS & PREDICTIONS
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 6: Business Insights & Sample Predictions")
print("=" * 80)

# Make predictions on test set
test_results = pd.DataFrame({
    'Actual_CLV': y_test.values,
    'Predicted_CLV': y_test_pred,
    'Prediction_Error': y_test.values - y_test_pred,
    'Error_Percentage': ((y_test.values - y_test_pred) / y_test.values * 100)
})

print("\n6.1 Sample Predictions from Test Set:")
print(test_results.head(10).to_string(index=False))

# Identify high-value customers
print("\n6.2 High-Value Customer Identification:")
high_value_threshold = y_test.quantile(0.75)
high_value_customers = test_results[test_results['Predicted_CLV'] >= high_value_threshold]
print(f"   • Identified {len(high_value_customers)} high-value customers")
print(f"   • Threshold: ${high_value_threshold:.2f}")
print(f"   • Average predicted CLV of high-value customers: ${high_value_customers['Predicted_CLV'].mean():.2f}")

# Feature importance visualization
print("\n6.3 Creating Feature Importance Visualization...")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': np.abs(model.coef_)
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(12, 8))
colors = ['green' if x > 0 else 'red' for x in model.coef_[feature_importance.index]]
plt.barh(range(len(feature_importance)), feature_importance['Importance'], color=colors, alpha=0.7)
plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
plt.xlabel('Absolute Coefficient Value', fontsize=12)
plt.title('Feature Importance in CLV Prediction\n(Green = Positive Impact, Red = Negative Impact)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('visualizations/06_feature_importance.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/06_feature_importance.png")
plt.close()

# ============================================================================
# STEP 7: BUSINESS RECOMMENDATIONS
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 7: Business Recommendations")
print("=" * 80)

recommendations = """
Based on the CLV prediction model, here are key business recommendations:

1. PREMIUM MEMBERSHIP PROGRAM
   • Premium members show significantly higher CLV
   → Action: Invest in converting high-potential customers to premium tier
   → Expected ROI: {premium_multiplier:.2f}x increase in customer value

2. EMAIL ENGAGEMENT STRATEGY
   • Email engagement shows strong positive correlation with CLV
   → Action: Implement personalized email campaigns
   → Focus: Re-engage customers with low engagement rates

3. CUSTOMER RETENTION FOCUS
   • Days since last purchase negatively impacts CLV
   → Action: Implement win-back campaigns for inactive customers
   → Target: Customers with >60 days since last purchase

4. MOBILE APP ADOPTION
   • Mobile users tend to have higher engagement
   → Action: Incentivize mobile app downloads
   → Strategy: Mobile-exclusive offers and features

5. PRODUCT CATEGORY EXPANSION
   • Customers purchasing across multiple categories show higher CLV
   → Action: Cross-sell and product recommendation strategies
   → Goal: Increase average categories per customer

6. CUSTOMER SERVICE OPTIMIZATION
   • Support tickets correlate with lower CLV
   → Action: Improve first-contact resolution
   → Invest in: Self-service resources and proactive support

MODEL ACCURACY: {test_r2:.1%} of CLV variance explained
PREDICTION ERROR: Average ±${test_mae:.2f} per customer
"""

print(recommendations.format(
    premium_multiplier=premium_avg_clv/non_premium_avg_clv,
    test_r2=test_r2,
    test_mae=test_mae
))

# ============================================================================
# STEP 8: SAVE MODEL AND RESULTS
# ============================================================================
print("=" * 80)
print("STEP 8: Saving Model and Results")
print("=" * 80)

# Save model coefficients
coefficients.to_csv('model_coefficients.csv', index=False)
print("\n✓ Saved: model_coefficients.csv")

# Save predictions
test_results.to_csv('test_predictions.csv', index=False)
print("✓ Saved: test_predictions.csv")

# Save summary report
summary = f"""
CUSTOMER LIFETIME VALUE PREDICTION - MODEL SUMMARY
{'=' * 70}

Dataset Information:
- Total Customers: {len(df)}
- Features Used: {len(features)}
- Training Samples: {len(X_train)}
- Testing Samples: {len(X_test)}

Model Performance:
- R² Score (Test): {test_r2:.4f}
- RMSE (Test): ${test_rmse:.2f}
- MAE (Test): ${test_mae:.2f}
- Variance Explained: {test_r2*100:.2f}%

Key Findings:
- Average CLV: ${df['customer_lifetime_value'].mean():.2f}
- Premium Member CLV Premium: {premium_avg_clv/non_premium_avg_clv:.2f}x
- Top Predictive Feature: {coefficients.iloc[0]['Feature']}

Generated: {pd.Timestamp.now()}
"""

with open('model_summary.txt', 'w') as f:
    f.write(summary)
print("✓ Saved: model_summary.txt")

print("\n" + "=" * 80)
print("PROJECT COMPLETE! 🎉")
print("=" * 80)
print("\nAll outputs saved in current directory:")
print("   📊 visualizations/ - All charts and plots")
print("   📄 model_coefficients.csv - Feature importance")
print("   📄 test_predictions.csv - Prediction results")
print("   📄 model_summary.txt - Summary report")
print("\n" + "=" * 80)