import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
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
        self.X_train_balanced = None
        self.y_train_balanced = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoders = {}
        self.balancing_method = None
        self.class_weights = None
        
    def load_data(self):
        print("Loading IBM HR Analytics Attrition Dataset...")
        
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
        print("\n=== DATA EXPLORATION ===")
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Target distribution:")
        attrition_dist = self.df['Attrition'].value_counts(normalize=True)
        print(attrition_dist)
        
        imbalance_ratio = attrition_dist['No'] / attrition_dist['Yes']
        print(f"Imbalance ratio (No:Yes): {imbalance_ratio:.2f}:1")
        
        print(f"\nMissing values: {self.df.isnull().sum().sum()}")
        
        print(f"\nCategorical columns: {len(self.df.select_dtypes(include=['object']).columns)}")
        print(f"Numerical columns: {len(self.df.select_dtypes(include=['int64', 'float64']).columns)}")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        sns.countplot(data=self.df, x='Department', hue='Attrition', ax=axes[0,0])
        axes[0,0].set_title('Attrition by Department')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        sns.countplot(data=self.df, x='JobSatisfaction', hue='Attrition', ax=axes[0,1])
        axes[0,1].set_title('Attrition by Job Satisfaction')
        
        sns.countplot(data=self.df, x='OverTime', hue='Attrition', ax=axes[0,2])
        axes[0,2].set_title('Attrition by Overtime')
        
        sns.histplot(data=self.df, x='Age', hue='Attrition', multiple='stack', ax=axes[1,0])
        axes[1,0].set_title('Age Distribution by Attrition')
        
        sns.histplot(data=self.df, x='MonthlyIncome', hue='Attrition', multiple='stack', ax=axes[1,1])
        axes[1,1].set_title('Income Distribution by Attrition')
        
        attrition_counts = self.df['Attrition'].value_counts()
        axes[1,2].pie(attrition_counts.values, labels=attrition_counts.index, autopct='%1.1f%%')
        axes[1,2].set_title('Attrition Distribution')
        
        plt.tight_layout()
        plt.savefig('attrition_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def feature_engineering(self):
        print("\n=== FEATURE ENGINEERING ===")
        
        df = self.df.copy()
        
        df['Attrition_Target'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        
        df['TenurePerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
        df['YearsWithoutPromotion'] = df['YearsSinceLastPromotion']
        df['SalaryPerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
        df['AgeExperienceGap'] = df['Age'] - df['TotalWorkingYears'] - 18
        
        df['WorkLifeScore'] = (df['WorkLifeBalance'] + df['JobSatisfaction'] + 
                              df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']) / 4
        df['LowEngagement'] = ((df['JobInvolvement'] <= 2) | (df['EnvironmentSatisfaction'] <= 2)).astype(int)
        df['TrainingEngagement'] = df['TrainingTimesLastYear'] * df['JobInvolvement']
        
        df['CareerGrowth'] = df['YearsAtCompany'] / (df['YearsSinceLastPromotion'] + 1)
        df['PromotionFrequency'] = df['TotalWorkingYears'] / (df['YearsSinceLastPromotion'] + 1)
        df['ManagerTenureRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
        df['RoleStability'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
        
        df['IsOverworked'] = ((df['OverTime'] == 'Yes') & (df['WorkLifeBalance'] <= 2)).astype(int)
        df['HighPerformerStuck'] = ((df['PerformanceRating'] >= 3) & (df['YearsSinceLastPromotion'] >= 3)).astype(int)
        df['JobHopperFlag'] = (df['NumCompaniesWorked'] >= 4).astype(int)
        df['LongDistanceCommute'] = (df['DistanceFromHome'] >= 20).astype(int)
        df['JobLevelMismatch'] = ((df['TotalWorkingYears'] >= 10) & (df['JobLevel'] <= 2)).astype(int)
        
        df['CompensationRatio'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] * 1000 + 1)
        df['IncomeToJobLevel'] = df['MonthlyIncome'] / (df['JobLevel'] * 1000 + 1)
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('Attrition')
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_Encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        cols_to_drop = categorical_cols + ['Attrition', 'EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        self.df = df
        print(f"Features created: {df.shape[1]} total features")
        print(f"New engineered features: {len([col for col in df.columns if '_' in col and col not in ['Attrition_Target']])}")
        
        return df
    
    def prepare_data(self):
        print("\n=== DATA PREPARATION ===")
        
        X = self.df.drop('Attrition_Target', axis=1)
        y = self.df['Attrition_Target']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.feature_names = X.columns.tolist()
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Attrition rate - Train: {self.y_train.mean():.2%}, Test: {self.y_test.mean():.2%}")
        
        train_dist = pd.Series(self.y_train).value_counts(normalize=True)
        print(f"Training imbalance ratio: {train_dist[0]/train_dist[1]:.2f}:1")
        
        self.class_weights = {0: 1, 1: train_dist[0]/train_dist[1]}
        print(f"Class weights: {self.class_weights}")
        
    def handle_imbalance(self, method='smote'):
        print(f"\n=== HANDLING IMBALANCED DATA ({method.upper()}) ===")
        
        if method == 'smote':
            balancer = SMOTE(random_state=42)
        elif method == 'adasyn':
            balancer = ADASYN(random_state=42)
        elif method == 'smote_tomek':
            balancer = SMOTETomek(random_state=42)
        elif method == 'smote_enn':
            balancer = SMOTEENN(random_state=42)
        elif method == 'undersample':
            balancer = RandomUnderSampler(random_state=42)
        else:
            print("No balancing applied")
            return
        
        self.X_train_balanced, self.y_train_balanced = balancer.fit_resample(self.X_train_scaled, self.y_train)
        self.balancing_method = method
        
        print(f"Original training set shape: {self.X_train_scaled.shape}")
        print(f"Balanced training set shape: {self.X_train_balanced.shape}")
        
        balanced_dist = pd.Series(self.y_train_balanced).value_counts(normalize=True)
        print(f"Balanced distribution: {balanced_dist.to_dict()}")
        
    def train_models(self, use_balanced=True):
        print("\n=== MODEL TRAINING ===")
        
        models = {
            'Random Forest': RandomForestClassifier(
                random_state=42, 
                n_estimators=100,
                class_weight=self.class_weights if not use_balanced else None
            ),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight=self.class_weights if not use_balanced else None
            )
        }
        
        results = {}
        X_train_data = self.X_train_balanced if use_balanced else self.X_train_scaled
        y_train_data = self.y_train_balanced if use_balanced else self.y_train
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            cv_scores = cross_val_score(model, X_train_data, y_train_data, cv=cv, scoring='roc_auc')
            
            model.fit(X_train_data, y_train_data)
            
            y_pred = model.predict(self.X_test_scaled)
            y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            accuracy = model.score(self.X_test_scaled, self.y_test)
            roc_auc = roc_auc_score(self.y_test, y_proba)
            precision = average_precision_score(self.y_test, y_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'avg_precision': precision,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  ROC AUC: {roc_auc:.3f}")
            print(f"  Avg Precision: {precision:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        self.model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best ROC AUC: {results[best_model_name]['roc_auc']:.3f}")
        
        return results, best_model_name
    
    def compare_balancing_methods(self):
        print("\n=== COMPARING BALANCING METHODS ===")
        
        methods = ['none', 'smote', 'adasyn', 'smote_tomek', 'undersample']
        comparison_results = {}
        
        for method in methods:
            print(f"\nTesting method: {method}")
            
            if method != 'none':
                self.handle_imbalance(method)
                use_balanced = True
            else:
                use_balanced = False
            
            results, best_model = self.train_models(use_balanced)
            comparison_results[method] = {
                'best_model': best_model,
                'roc_auc': results[best_model]['roc_auc'],
                'accuracy': results[best_model]['accuracy'],
                'avg_precision': results[best_model]['avg_precision']
            }
        
        comparison_df = pd.DataFrame(comparison_results).T
        print("\nBalancing Methods Comparison:")
        print(comparison_df.round(3))
        
        best_method = comparison_df['roc_auc'].idxmax()
        print(f"\nBest balancing method: {best_method}")
        
        if best_method != 'none':
            self.handle_imbalance(best_method)
            results, best_model = self.train_models(True)
        else:
            results, best_model = self.train_models(False)
        
        plt.figure(figsize=(12, 6))
        comparison_df['roc_auc'].plot(kind='bar', color='skyblue')
        plt.title('ROC AUC by Balancing Method')
        plt.xlabel('Balancing Method')
        plt.ylabel('ROC AUC')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('balancing_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_results
    
    def analyze_feature_importance(self):
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            print("Model doesn't support feature importance analysis")
            return
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(20)
        
        print("\nTop 20 Most Important Features:")
        for i, row in top_features.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 20 Feature Importance for Employee Attrition Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        engineered_features = [f for f in top_features['feature'] if any(keyword in f for keyword in ['Score', 'Ratio', 'Flag', 'Growth', 'Frequency', 'Per'])]
        original_features = [f for f in top_features['feature'] if f not in engineered_features]
        
        print(f"\nEngineered features in top 20: {len(engineered_features)}")
        print(f"Original features in top 20: {len(original_features)}")
        
        return feature_importance
    
    def evaluate_model(self):
        print("\n=== MODEL EVALUATION ===")
        
        y_pred = self.model.predict(self.X_test_scaled)
        y_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Stay', 'Leave']))
        
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Stay', 'Leave'], yticklabels=['Stay', 'Leave'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        fpr, tpr, thresholds = roc_curve(self.y_test, y_proba)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        
        precision, recall, pr_thresholds = precision_recall_curve(self.y_test, y_proba)
        avg_precision = average_precision_score(self.y_test, y_proba)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(recall, precision, color='green', label=f'PR Curve (AP = {avg_precision:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        tn, fp, fn, tp = cm.ravel()
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nDetailed Metrics:")
        print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.3f}")
        print(f"Precision: {precision_val:.3f}")
        print(f"Recall (Sensitivity): {recall_val:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"False Positive Rate: {fp/(fp+tn):.3f}")
        print(f"False Negative Rate: {fn/(fn+tp):.3f}")
        
        return {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': precision_val,
            'recall': recall_val,
            'specificity': specificity,
            'f1': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision
        }
    
    def save_model(self):
        print("\n=== SAVING MODEL ===")
        
        model_info = {
            'model_name': type(self.model).__name__,
            'accuracy': self.model.score(self.X_test_scaled, self.y_test),
            'roc_auc': roc_auc_score(self.y_test, self.model.predict_proba(self.X_test_scaled)[:, 1]),
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'n_samples': len(self.X_train),
            'balancing_method': self.balancing_method,
            'class_weights': self.class_weights
        }
        
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
        print(f"Balancing Method: {model_info['balancing_method']}")
        print(f"Class Weights: {model_info['class_weights']}")
    
    def predict_employee(self, employee_data):
        df_new = pd.DataFrame([employee_data])
        return None

def main():
    print("=" * 60)
    print("EMPLOYEE ATTRITION PREDICTION - COMPLETE ML PIPELINE")
    print("WITH IMBALANCED DATASET HANDLING")
    print("=" * 60)
    
    ml_pipeline = EmployeeAttritionML()
    
    ml_pipeline.load_data()
    ml_pipeline.explore_data()
    
    ml_pipeline.feature_engineering()
    ml_pipeline.prepare_data()
    
    comparison_results = ml_pipeline.compare_balancing_methods()
    
    feature_importance = ml_pipeline.analyze_feature_importance()
    
    metrics = ml_pipeline.evaluate_model()
    
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
    print("- roc_pr_curves.png: ROC and Precision-Recall curves")
    print("- attrition_exploration.png: Data exploration plots")
    print("- balancing_comparison.png: Balancing methods comparison")
    
    return ml_pipeline, comparison_results, metrics

if __name__ == "__main__":
    ml_pipeline, comparison_results, metrics = main()
