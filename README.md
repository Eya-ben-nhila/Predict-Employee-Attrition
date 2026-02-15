# Employee Attrition Prediction System

A machine learning system that predicts employee attrition using a trained model with comprehensive feature engineering and interactive testing capabilities.

## Overview

This system uses a pre-trained machine learning model to predict whether employees are likely to leave the company based on various factors including demographics, job characteristics, satisfaction metrics, and work-life balance indicators.

## Features

- **Pre-trained Model**: Loads a trained model with 87% accuracy and ROC AUC of 0.9381
- **Comprehensive Feature Engineering**: Creates 15+ engineered features including:
  - Tenure-based metrics (TenurePerCompany, YearsWithoutPromotion)
  - Financial indicators (SalaryPerYear, CompensationRatio)
  - Satisfaction scores (WorkLifeScore, LowEngagement flag)
  - Career progression indicators (CareerGrowth, PromotionFrequency)
  - Risk flags (IsOverworked, HighPerformerStuck, JobHopperFlag)
- **Interactive Testing**: Multiple testing options for model validation
- **Real Dataset Integration**: Uses IBM HR Analytics Attrition Dataset from Kaggle

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- pickle
- kagglehub

## Model Files Required

The system expects these pickle files in the same directory:
- `best_model.pkl` - Trained classification model
- `scaler.pkl` - Feature scaler for preprocessing
- `model_info.pkl` - Model metadata and feature names

## Usage

Run the main script:
```bash
python sub4.py
```

The system will present 6 testing options:

1. **Test random employee** - Tests a randomly selected employee
2. **Test specific employee by ID** - Tests an employee at a specific index
3. **Test multiple random employees** - Tests multiple random employees with accuracy summary
4. **Test employees who left** - Tests high-risk cases (actual leavers)
5. **Test employees who stayed** - Tests low-risk cases (actual stayers)
6. **Compare predictions vs actual** - Detailed comparison table with metrics

## Model Performance

- Accuracy: 87%
- ROC AUC: 0.9381
- Features: 50+ engineered and original features
- Dataset: IBM HR Analytics (1,470 employees)

## Feature Engineering

The system creates sophisticated features including:

**Experience & Tenure:**
- TenurePerCompany: Total working years divided by companies worked
- YearsWithoutPromotion: Time since last promotion
- SalaryPerYear: Monthly income relative to experience

**Satisfaction & Engagement:**
- WorkLifeScore: Combined satisfaction metrics
- LowEngagement: Flag for low job involvement or environment satisfaction
- TrainingEngagement: Training frequency multiplied by job involvement

**Risk Indicators:**
- IsOverworked: Overtime workers with poor work-life balance
- HighPerformerStuck: Good performers without recent promotions
- JobHopperFlag: Employees with 4+ previous companies
- LongDistanceCommute: Employees commuting 20+ miles

**Career Progression:**
- CareerGrowth: Years at company relative to promotion timing
- PromotionFrequency: How often promotions occur
- JobLevelMismatch: Experienced employees in junior roles

## Data Source

Uses the IBM HR Analytics Attrition Dataset available on Kaggle, containing 1,470 employee records with 35 features including demographics, job details, satisfaction scores, and attrition status.
