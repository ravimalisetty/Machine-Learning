# Customer Lifetime Value Prediction Using Linear Regression

## 📊 Project Overview

This project predicts **Customer Lifetime Value (CLV)** for e-commerce customers using Linear Regression. CLV represents the total revenue a business expects from a customer throughout their relationship, making it crucial for:

- **Marketing Budget Optimization**: Allocate resources to high-value customers
- **Customer Retention**: Identify and retain valuable customers
- **Personalization**: Tailor experiences based on predicted value
- **Strategic Planning**: Make data-driven acquisition and retention decisions

**Model Performance**: 95.5% R² Score (explains 95.5% of variance in CLV)

---

## 🎯 Business Problem

E-commerce companies need to predict which customers will generate the most revenue over time to:
1. Focus retention efforts on high-value customers
2. Optimize marketing spend across customer segments
3. Personalize customer experiences based on predicted value
4. Make informed decisions about customer acquisition costs

---

## 📁 Project Structure
```
customer_lifetime_value_prediction/
├── ecommerce_customer_data.csv          # Dataset (1000 customers)
├── generate_dataset.py                   # Script to generate realistic data
├── clv_prediction_analysis.py            # Main analysis script
├── README.md                             # Project documentation
├── model_coefficients.csv                # Feature importance results
├── test_predictions.csv                  # Model predictions on test set
├── model_summary.txt                     # Summary report
└── visualizations/                       # All generated visualizations
    ├── 01_target_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_feature_relationships.png
    ├── 04_prediction_results.png
    ├── 05_residual_analysis.png
    └── 06_feature_importance.png
```

---

## 📊 Dataset Description

**Source**: Synthetically generated realistic e-commerce customer data  
**Size**: 1,000 customers with 14 features

### Features

| Feature | Description | Type |
|---------|-------------|------|
| `customer_id` | Unique customer identifier | Numeric |
| `account_age_days` | Days since account creation | Numeric |
| `customer_age` | Customer's age | Numeric |
| `num_purchases` | Total number of purchases | Numeric |
| `avg_order_value` | Average amount spent per order | Numeric |
| `total_spend` | Total amount spent to date | Numeric |
| `days_since_last_purchase` | Recency metric | Numeric |
| `product_categories` | Number of different categories purchased | Numeric |
| `email_engagement_rate` | Email open/click rate (%) | Numeric |
| `returns_count` | Number of returned items | Numeric |
| `support_tickets` | Customer service interactions | Numeric |
| `is_mobile_user` | Uses mobile app (0/1) | Binary |
| `is_newsletter_subscriber` | Newsletter subscription status (0/1) | Binary |
| `is_premium_member` | Premium membership status (0/1) | Binary |
| **`customer_lifetime_value`** | **Target variable - Predicted future value** | **Numeric** |

---

## 🔬 Methodology

### 1. Data Exploration & Analysis
- Statistical analysis of customer metrics
- Correlation analysis between features
- Customer segmentation insights
- Distribution analysis

### 2. Feature Engineering
- Selected 13 predictive features
- Standardized features using StandardScaler
- Removed customer_id (non-predictive)

### 3. Model Development
- **Algorithm**: Linear Regression
- **Train-Test Split**: 80-20
- **Feature Scaling**: StandardScaler
- **Evaluation Metrics**: R², RMSE, MAE

### 4. Model Evaluation
- Cross-validation on test set
- Residual analysis
- Overfitting check
- Feature importance analysis

---

## 📈 Key Results

### Model Performance

| Metric | Training Set | Testing Set |
|--------|--------------|-------------|
| **R² Score** | 0.9463 | **0.9548** |
| **RMSE** | $617.52 | **$498.17** |
| **MAE** | $405.50 | **$356.77** |

✅ **Model shows excellent generalization** - No overfitting detected

### Top Predictive Features

1. **Total Spend** (Coefficient: +3019.38) - Strongest positive predictor
2. **Average Order Value** (Coefficient: -619.11) - Complex relationship
3. **Days Since Last Purchase** (Coefficient: -409.62) - Recency matters
4. **Email Engagement Rate** (Coefficient: +266.86) - Engagement drives value
5. **Premium Membership** (Coefficient: +146.32) - Premium = Higher CLV

### Business Insights

- **Premium Members**: 1.62x higher CLV than non-premium customers
- **Email Engagement**: Strong positive correlation with CLV
- **Customer Recency**: Active customers have significantly higher CLV
- **Product Diversity**: Customers buying across categories show higher value

---

## 💡 Business Recommendations

### 1. Premium Membership Program
- **Finding**: Premium members show 1.62x higher CLV
- **Action**: Convert high-potential customers to premium tier
- **Expected Impact**: Significant revenue increase per converted customer

### 2. Email Engagement Strategy
- **Finding**: Email engagement strongly correlates with CLV
- **Action**: Implement personalized, targeted email campaigns
- **Focus**: Re-engage low-engagement customers

### 3. Customer Retention Focus
- **Finding**: Days since last purchase negatively impacts CLV
- **Action**: Win-back campaigns for inactive customers (>60 days)
- **Strategy**: Personalized offers and reminders

### 4. Mobile App Adoption
- **Finding**: Mobile users show higher engagement
- **Action**: Incentivize mobile app downloads
- **Tactics**: Mobile-exclusive offers and features

### 5. Cross-Selling Strategy
- **Finding**: Multi-category purchasers have higher CLV
- **Action**: Implement smart product recommendations
- **Goal**: Increase average categories per customer

### 6. Customer Service Optimization
- **Finding**: Support tickets correlate with lower CLV
- **Action**: Improve first-contact resolution
- **Investment**: Self-service resources and proactive support

---

## 🚀 How to Run This Project

### Prerequisites
```
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Installation

1. **Clone or navigate to this project**

2. **Install dependencies**:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. **Generate the dataset** (optional - dataset already included):
```bash
python generate_dataset.py
```

4. **Run the complete analysis**:
```bash
python clv_prediction_analysis.py
```

### Expected Output
- Console output with detailed analysis steps
- 6 visualization images in `visualizations/` folder
- 3 CSV/TXT files with results and summaries

---

## 📊 Sample Visualizations

The project generates professional visualizations including:
- CLV distribution analysis
- Feature correlation heatmap
- Prediction accuracy plots
- Residual analysis
- Feature importance rankings
- Key business relationships

All visualizations are saved in high resolution (300 DPI) suitable for presentations.

---

## 🎓 Skills Demonstrated

### Technical Skills
- **Python Programming**: pandas, numpy, scikit-learn
- **Data Analysis**: EDA, statistical analysis, correlation studies
- **Machine Learning**: Linear regression, feature engineering, model evaluation
- **Data Visualization**: matplotlib, seaborn
- **Model Validation**: Train-test split, cross-validation, residual analysis

### Business Skills
- **Domain Understanding**: E-commerce, customer analytics, CLV concepts
- **Strategic Thinking**: Translating model insights into actionable recommendations
- **Communication**: Clear documentation, visualization, and reporting

---

## 💼 Interview Talking Points

### "Tell me about this project"
*"I built a Customer Lifetime Value prediction model for an e-commerce business using Linear Regression. The model achieved 95.5% accuracy in predicting customer value, which helps businesses optimize marketing spend and retention strategies. I worked with a dataset of 1,000 customers and 14 features including purchase history, engagement metrics, and demographic data."*

### "What were your key findings?"
*"I discovered that premium members have 1.62x higher CLV, and email engagement is a strong predictor of customer value. The model also revealed that recency matters - customers who haven't purchased in 60+ days show declining CLV. These insights led to six actionable business recommendations."*

### "What challenges did you face?"
*"One challenge was feature engineering - deciding which features to include and how to handle multicollinearity. For example, total_spend and avg_order_value are correlated, but both add predictive value in different ways. I addressed this through careful correlation analysis and feature importance evaluation."*

### "How does this apply to real business?"
*"This model helps businesses make data-driven decisions about where to invest their marketing budget. For instance, instead of treating all customers equally, they can identify high-value customers for premium tier conversion or target at-risk customers with win-back campaigns before they churn."*

---

## 📚 Future Enhancements

1. **Advanced Models**: Compare with Random Forest, XGBoost, Neural Networks
2. **Time Series**: Incorporate temporal patterns in purchasing behavior
3. **Segmentation**: Build separate models for different customer segments
4. **Feature Engineering**: Create interaction features, polynomial features
5. **Deployment**: Create REST API for real-time predictions
6. **A/B Testing**: Validate recommendations through controlled experiments

---

## 📧 Contact

**[Your Name]**  
Email: your.email@example.com  
LinkedIn: linkedin.com/in/yourprofile  
GitHub: github.com/yourusername

---

## 📝 License

This project is for educational and portfolio purposes.

---

## 🙏 Acknowledgments

- Dataset: Synthetically generated for educational purposes
- Libraries: scikit-learn, pandas, matplotlib, seaborn
- Methodology: Based on industry best practices in CLV modeling

---

**Last Updated**: February 2026