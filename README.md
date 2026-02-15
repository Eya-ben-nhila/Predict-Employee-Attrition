# Employee Attrition Prediction System

A comprehensive machine learning system that predicts employee attrition using advanced feature engineering, multiple model comparison, and detailed feature importance analysis to help HR understand key drivers of attrition.

## Overview

This system provides a complete ML pipeline for employee attrition prediction, including data exploration, feature engineering, model training, evaluation, and deployment. It uses the IBM HR Analytics Attrition Dataset to build binary classification models that identify employees at risk of leaving the company.

## Key Features

### Complete ML Pipeline
- **Data Exploration**: Comprehensive EDA with visualizations
- **Feature Engineering**: 15+ engineered features including tenure metrics, satisfaction scores, and risk indicators
- **Model Comparison**: Random Forest, Gradient Boosting, and Logistic Regression
- **Cross-Validation**: Robust model evaluation with 5-fold CV
- **Feature Importance Analysis**: Detailed analysis of key attrition drivers

### Advanced Features
- **Experience Metrics**: TenurePerCompany, SalaryPerYear, AgeExperienceGap
- **Satisfaction Scores**: WorkLifeScore, LowEngagement flag, TrainingEngagement
- **Career Progression**: CareerGrowth, PromotionFrequency, ManagerTenureRatio
- **Risk Indicators**: IsOverworked, HighPerformerStuck, JobHopperFlag
- **Financial Analysis**: CompensationRatio, IncomeToJobLevel

### Visualization & Analysis
- Feature importance plots showing top 20 predictors
- Confusion matrix and ROC curves
- Data exploration visualizations
- Business metrics (precision, recall, F1-score)

## Requirements

- Python 3.7+
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- kagglehub>=0.1.0

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Complete ML Pipeline (Recommended)

Run the complete machine learning pipeline:

```bash
python employee_attrition_ml.py
```

This will:
- Load and explore the IBM HR Analytics dataset
- Perform comprehensive feature engineering
- Train and compare multiple models
- Generate feature importance analysis
- Create evaluation visualizations
- Save the best model for deployment

### 2. Interactive Prediction System

After training the model, use the interactive prediction system:

```bash
python sub4.py
```

Available testing options:
1. Test random employee from dataset
2. Test specific employee by ID
3. Test multiple random employees
4. Test employees who left (high risk)
5. Test employees who stayed (low risk)
6. Compare predictions vs actual for sample

## Model Performance

The system automatically selects the best performing model based on ROC AUC score. Typical results:
- **Accuracy**: 85-90%
- **ROC AUC**: 0.90-0.95
- **Cross-validation**: 5-fold CV with confidence intervals

## Feature Importance Analysis

The system provides comprehensive feature importance analysis:

### Top Predictors (Typical)
1. **OverTime** - Whether employee works overtime
2. **MonthlyIncome** - Salary level
3. **JobLevel** - Position seniority
4. **TotalWorkingYears** - Total experience
5. **WorkLifeScore** - Combined satisfaction metrics

### Engineered Features Impact
The system creates 15+ engineered features that often outperform original features:
- **IsOverworked**: Overtime + poor work-life balance
- **HighPerformerStuck**: Good performers without recent promotions
- **CareerGrowth**: Years at company relative to promotion timing
- **CompensationRatio**: Income relative to experience

## Generated Files

### Model Files
- `best_model.pkl` - Trained classification model
- `scaler.pkl` - Feature scaler for preprocessing
- `model_info.pkl` - Model metadata and performance metrics
- `label_encoders.pkl` - Categorical variable encoders

### Visualization Files
- `feature_importance.png` - Top 20 feature importance plot
- `confusion_matrix.png` - Model confusion matrix
- `roc_curve.png` - ROC curve and AUC score
- `attrition_exploration.png` - Data exploration visualizations

## Business Insights

### Key Attrition Drivers
1. **Work-Life Balance**: Employees with poor work-life balance and overtime
2. **Career Growth**: Lack of promotions despite good performance
3. **Compensation**: Below-market salary relative to experience
4. **Job Satisfaction**: Low engagement and environment satisfaction
5. **Experience Level**: Mid-career employees (5-10 years) at highest risk

### HR Action Items
- Monitor employees with high "IsOverworked" scores
- Review promotion frequency for high performers
- Analyze compensation ratios by experience level
- Focus on mid-career retention programs
- Track job satisfaction trends by department

## Data Source

Uses the IBM HR Analytics Attrition Dataset from Kaggle:
- **1,470 employee records**
- **35 original features** plus 15+ engineered features
- **Target**: Attrition (Yes/No)
- **Imbalance**: 16% attrition rate (realistic for most companies)

## Model Comparison

The system evaluates three algorithms:
1. **Random Forest**: Best for feature importance interpretation
2. **Gradient Boosting**: Often highest predictive accuracy
3. **Logistic Regression**: Good baseline with interpretable coefficients

## Advanced Analytics

### Risk Segmentation
- **High Risk**: IsOverworked + LowEngagement + JobHopperFlag
- **Medium Risk**: Poor satisfaction + long tenure without promotion
- **Low Risk**: High satisfaction + good work-life balance

### Department Analysis
- Compare attrition rates across departments
- Identify department-specific risk factors
- Tailor retention strategies by department

## Technical Details

### Feature Engineering Pipeline
1. **Experience Features**: Tenure, salary ratios, age gaps
2. **Satisfaction Features**: Composite scores, engagement flags
3. **Career Features**: Growth metrics, promotion frequency
4. **Risk Features**: Overtime, performance, commute flags
5. **Financial Features**: Compensation analysis, income ratios

### Model Evaluation
- **Stratified Train/Test Split**: Maintains attrition rate balance
- **5-Fold Cross-Validation**: Robust performance estimation
- **Multiple Metrics**: Accuracy, ROC AUC, Precision, Recall, F1
- **Business Metrics**: False positive/negative rates for cost analysis
