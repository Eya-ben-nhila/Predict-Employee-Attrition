# Employee Attrition Prediction System

A comprehensive machine learning system that predicts employee attrition using advanced feature engineering, multiple model comparison, and detailed feature importance analysis with proper handling of imbalanced datasets.

## Overview

This system provides a complete ML pipeline for employee attrition prediction, including data exploration, feature engineering, model training, evaluation, and deployment. It uses the IBM HR Analytics Attrition Dataset to build binary classification models that identify employees at risk of leaving the company, with special attention to handling the imbalanced nature of attrition data.

## Key Features

### Complete ML Pipeline
- **Data Exploration**: Comprehensive EDA with visualizations
- **Feature Engineering**: 15+ engineered features including tenure metrics, satisfaction scores, and risk indicators
- **Model Comparison**: Random Forest, Gradient Boosting, and Logistic Regression
- **Cross-Validation**: Robust model evaluation with 5-fold stratified CV
- **Feature Importance Analysis**: Detailed analysis of key attrition drivers

### Imbalanced Dataset Handling
- **Multiple Balancing Methods**: SMOTE, ADASYN, SMOTETomek, SMOTEENN, Random Undersampling
- **Class Weight Handling**: Automatic class weight calculation for imbalanced data
- **Method Comparison**: Automated comparison of different balancing techniques
- **Advanced Metrics**: Precision-Recall curves, Average Precision, F1-score

### Advanced Features
- **Experience Metrics**: TenurePerCompany, SalaryPerYear, AgeExperienceGap
- **Satisfaction Scores**: WorkLifeScore, LowEngagement flag, TrainingEngagement
- **Career Progression**: CareerGrowth, PromotionFrequency, ManagerTenureRatio
- **Risk Indicators**: IsOverworked, HighPerformerStuck, JobHopperFlag
- **Financial Analysis**: CompensationRatio, IncomeToJobLevel

### Visualization & Analysis
- Feature importance plots showing top 20 predictors
- Confusion matrix and ROC curves
- Precision-Recall curves for imbalanced evaluation
- Balancing methods comparison charts
- Data exploration visualizations

## Requirements

- Python 3.7+
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- kagglehub>=0.1.0
- imbalanced-learn>=0.8.0

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Complete ML Pipeline (Recommended)

Run the complete machine learning pipeline with imbalanced dataset handling:

```bash
python employee_attrition_ml_fixed.py
```

This will:
- Load and explore the IBM HR Analytics dataset
- Analyze and visualize class imbalance (typically 84:16 ratio)
- Perform comprehensive feature engineering
- Compare multiple balancing methods automatically
- Train and compare multiple models with proper handling
- Generate feature importance analysis
- Create evaluation visualizations including PR curves
- Save the best model for deployment

### 2. Interactive Prediction System

After training the model, use the interactive prediction system:

```bash
python sub4_clean.py
```

Available testing options:
1. Test random employee from dataset
2. Test specific employee by ID
3. Test multiple random employees
4. Test employees who left (high risk)
5. Test employees who stayed (low risk)
6. Compare predictions vs actual for sample

## Imbalanced Dataset Handling

### The Challenge
Employee attrition datasets are typically imbalanced:
- **Attrition Rate**: ~16% (Yes) vs 84% (No)
- **Imbalance Ratio**: ~5:1 (No:Yes)
- **Business Impact**: False negatives (missing leavers) are more costly than false positives

### Implemented Solutions

#### 1. Data-Level Approaches
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **ADASYN**: Adaptive Synthetic Sampling
- **SMOTETomek**: SMOTE + Tomek link cleaning
- **SMOTEENN**: SMOTE + Edited Nearest Neighbors
- **Random Undersampling**: Reduce majority class

#### 2. Algorithm-Level Approaches
- **Class Weights**: Automatic weight calculation (inverse class frequency)
- **Stratified Cross-Validation**: Maintains class distribution in folds

#### 3. Evaluation Metrics
- **ROC AUC**: Standard for imbalanced classification
- **Precision-Recall AUC**: More informative for imbalanced data
- **Average Precision**: Summary of PR curve
- **F1-Score**: Balance between precision and recall
- **Specificity**: True negative rate

### Automatic Method Selection
The system automatically:
1. Tests all balancing methods
2. Compares performance using ROC AUC
3. Selects the best method
4. Trains final model with optimal approach

## Model Performance

With proper imbalanced dataset handling:
- **Accuracy**: 85-90%
- **ROC AUC**: 0.90-0.95
- **Average Precision**: 0.60-0.75
- **Cross-validation**: 5-fold stratified CV with confidence intervals

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
- `roc_pr_curves.png` - ROC and Precision-Recall curves
- `balancing_comparison.png` - Balancing methods comparison
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

The system evaluates three algorithms with proper imbalanced handling:
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
- **5-Fold Stratified Cross-Validation**: Robust performance estimation
- **Multiple Metrics**: Accuracy, ROC AUC, Precision, Recall, F1, Average Precision
- **Business Metrics**: False positive/negative rates for cost analysis

### Imbalanced Handling Workflow
1. **Detect Imbalance**: Calculate class distribution and ratio
2. **Calculate Weights**: Automatic class weight computation
3. **Test Methods**: Compare multiple balancing approaches
4. **Select Best**: Choose optimal method based on ROC AUC
5. **Train Final**: Train model with selected approach
6. **Evaluate**: Comprehensive evaluation with appropriate metrics
