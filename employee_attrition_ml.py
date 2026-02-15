import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class EmployeeAttritionML:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load the IBM HR Analytics dataset"""
        print("Loading IBM HR Analytics Attrition Dataset...")
        
        # Try to load from local file first, then download if needed
        if os.path.exists('WA_Fn-UseC_-HR-Employee-Attrition.csv'):
            self.df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
        else:
            import kagglehub
            path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            self.df = pd.read_csv(os.path.join(path, csv_files[0]))
        
        print(f"Dataset loaded: {self.df.shape[0]} employees, {self.df.shape[1]} features")
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n=== DATA EXPLORATION ===")
        
        # Basic info
        print(f"Dataset shape: {self.df.shape}")
        print(f"Target distribution:")
        print(self.df['Attrition'].value_counts(normalize=True))
        
        # Missing values
        print(f"\nMissing values: {self.df.isnull().sum().sum()}")
        
        # Data types
        print(f"\nCategorical columns: {len(self.df.select_dtypes(include=['object']).columns)}")
        print(f"Numerical columns: {len(self.df.select_dtypes(include=['int64', 'float64']).columns)}")
        
        # Attrition by key factors
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Attrition by Department
        sns.countplot(data=self.df, x='Department', hue='Attrition', ax=axes[0,0])
        axes[0,0].set_title('Attrition by Department')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Attrition by Job Satisfaction
        sns.countplot(data=self.df, x='JobSatisfaction', hue='Attrition', ax=axes[0,1])
        axes[0,1].set_title('Attrition by Job Satisfaction')
        
        # Attrition by Overtime
        sns.countplot(data=self.df, x='OverTime', hue='Attrition', ax=axes[1,0])
        axes[1,0].set_title('Attrition by Overtime')
        
        # Age distribution by Attrition
        sns.histplot(data=self.df, x='Age', hue='Attrition', multiple='stack', ax=axes[1,1])
        axes[1,1].set_title('Age Distribution by Attrition')
        
        plt.tight_layout()
        plt.savefig('attrition_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def feature_engineering(self):
        """Create advanced features for better prediction"""
        print("\n=== FEATURE ENGINEERING ===")
        
        df = self.df.copy()
        
        # Target variable
        df['Attrition_Target'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        
        # 1. Experience-based features
        df['TenurePerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
        df['YearsWithoutPromotion'] = df['YearsSinceLastPromotion']
        df['SalaryPerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
        df['AgeExperienceGap'] = df['Age'] - df['TotalWorkingYears'] - 18
        
        # 2. Satisfaction and engagement features
        df['WorkLifeScore'] = (df['WorkLifeBalance'] + df['JobSatisfaction'] + 
                              df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']) / 4
        df['LowEngagement'] = ((df['JobInvolvement'] <= 2) | (df['EnvironmentSatisfaction'] <= 2)).astype(int)
        df['TrainingEngagement'] = df['TrainingTimesLastYear'] * df['JobInvolvement']
        
        # 3. Career progression features
        df['CareerGrowth'] = df['YearsAtCompany'] / (df['YearsSinceLastPromotion'] + 1)
        df['PromotionFrequency'] = df['TotalWorkingYears'] / (df['YearsSinceLastPromotion'] + 1)
        df['ManagerTenureRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
        df['RoleStability'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
        
        # 4. Risk indicator features
        df['IsOverworked'] = ((df['OverTime'] == 'Yes') & (df['WorkLifeBalance'] <= 2)).astype(int)
        df['HighPerformerStuck'] = ((df['PerformanceRating'] >= 3) & (df['YearsSinceLastPromotion'] >= 3)).astype(int)
        df['JobHopperFlag'] = (df['NumCompaniesWorked'] >= 4).astype(int)
        df['LongDistanceCommute'] = (df['DistanceFromHome'] >= 20).astype(int)
        df['JobLevelMismatch'] = ((df['TotalWorkingYears'] >= 10) & (df['JobLevel'] <= 2)).astype(int)
        
        # 5. Financial features
        df['CompensationRatio'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] * 1000 + 1)
        df['IncomeToJobLevel'] = df['MonthlyIncome'] / (df['JobLevel'] * 1000 + 1)
        
        # 6. Categorical encoding
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('Attrition')  # Remove target
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_Encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        # Drop original categorical columns and unnecessary columns
        cols_to_drop = categorical_cols + ['Attrition', 'EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        self.df = df
        print(f"Features created: {df.shape[1]} total features")
        print(f"New engineered features: {len([col for col in df.columns if '_' in col and col not in ['Attrition_Target']])}")
        
        return df
    
    def prepare_data(self):
        """Prepare data for modeling"""
        print("\n=== DATA PREPARATION ===")
        
        # Separate features and target
        X = self.df.drop('Attrition_Target', axis=1)
        y = self.df['Attrition_Target']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.feature_names = X.columns.tolist()
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Attrition rate - Train: {self.y_train.mean():.2%}, Test: {self.y_test.mean():.2%}")
        
    def train_models(self):
        """Train and compare multiple models"""
        print("\n=== MODEL TRAINING ===")
        
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='roc_auc')
            
            # Fit model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = model.score(self.X_test_scaled, self.y_test)
            roc_auc = roc_auc_score(self.y_test, y_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  ROC AUC: {roc_auc:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        self.model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best ROC AUC: {results[best_model_name]['roc_auc']:.3f}")
        
        return results, best_model_name
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importances = np.abs(self.model.coef_[0])
        else:
            print("Model doesn't support feature importance analysis")
            return
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Top 20 features
        top_features = feature_importance.head(20)
        
        print("\nTop 20 Most Important Features:")
        for i, row in top_features.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        # Visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 20 Feature Importance for Employee Attrition Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance by category
        engineered_features = [f for f in top_features['feature'] if any(keyword in f for keyword in ['Score', 'Ratio', 'Flag', 'Growth', 'Frequency', 'Per'])]
        original_features = [f for f in top_features['feature'] if f not in engineered_features]
        
        print(f"\nEngineered features in top 20: {len(engineered_features)}")
        print(f"Original features in top 20: {len(original_features)}")
        
        return feature_importance
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION ===")
        
        y_pred = self.model.predict(self.X_test_scaled)
        y_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Stay', 'Leave']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Stay', 'Leave'], yticklabels=['Stay', 'Leave'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, y_proba)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Business metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nBusiness Metrics:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"False Positive Rate: {fp/(fp+tn):.3f}")
        print(f"False Negative Rate: {fn/(fn+tp):.3f}")
        
        return {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def save_model(self):
        """Save the trained model and components"""
        print("\n=== SAVING MODEL ===")
        
        model_info = {
            'model_name': type(self.model).__name__,
            'accuracy': self.model.score(self.X_test_scaled, self.y_test),
            'roc_auc': roc_auc_score(self.y_test, self.model.predict_proba(self.X_test_scaled)[:, 1]),
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'n_samples': len(self.X_train)
        }
        
        # Save components
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open('model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print("Model and components saved successfully!")
        print(f"Model: {model_info['model_name']}")
        print(f"Accuracy: {model_info['accuracy']:.2%}")
        print(f"ROC AUC: {model_info['roc_auc']:.4f}")
    
    def predict_employee(self, employee_data):
        """Make prediction for a single employee"""
        # This would be used by the existing prediction system
        df_new = pd.DataFrame([employee_data])
        
        # Apply same feature engineering
        # (This would need to be implemented based on the feature_engineering method)
        
        return None

def main():
    """Main execution function"""
    print("=" * 60)
    print("EMPLOYEE ATTRITION PREDICTION - COMPLETE ML PIPELINE")
    print("=" * 60)
    
    # Initialize ML pipeline
    ml_pipeline = EmployeeAttritionML()
    
    # Load and explore data
    ml_pipeline.load_data()
    ml_pipeline.explore_data()
    
    # Feature engineering
    ml_pipeline.feature_engineering()
    
    # Prepare data
    ml_pipeline.prepare_data()
    
    # Train models
    results, best_model = ml_pipeline.train_models()
    
    # Feature importance analysis
    feature_importance = ml_pipeline.analyze_feature_importance()
    
    # Evaluate model
    metrics = ml_pipeline.evaluate_model()
    
    # Save model
    ml_pipeline.save_model()
    
    print("\n" + "=" * 60)
    print("ML PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Files generated:")
    print("- best_model.pkl: Trained model")
    print("- scaler.pkl: Feature scaler")
    print("- model_info.pkl: Model metadata")
    print("- label_encoders.pkl: Categorical encoders")
    print("- feature_importance.png: Feature importance visualization")
    print("- confusion_matrix.png: Confusion matrix")
    print("- roc_curve.png: ROC curve")
    print("- attrition_exploration.png: Data exploration plots")
    
    return ml_pipeline, results, metrics

if __name__ == "__main__":
    ml_pipeline, results, metrics = main()
