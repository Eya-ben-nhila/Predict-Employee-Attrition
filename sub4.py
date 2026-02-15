import pandas as pd
import pickle
import numpy as np
import os

print("\nEmployee Attrition Prediction System")
print("Loading model and dataset...\n")

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

print(f"Model: {model_info['model_name']}")
print(f"Accuracy: {model_info['accuracy']:.2%}")
print(f"ROC AUC: {model_info['roc_auc']:.4f}\n")

import kagglehub
path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
df = pd.read_csv(os.path.join(path, csv_files[0]))

print(f"Dataset loaded: {df.shape[0]} employees\n")

df['Attrition_Actual'] = df['Attrition'].map({'Yes': 1, 'No': 0})

def make_prediction(employee_data, show_actual=False):
    
    if isinstance(employee_data, dict):
        df_new = pd.DataFrame([employee_data])
    else:
        df_new = employee_data.copy()
    
    actual_values = None
    if 'Attrition_Actual' in df_new.columns:
        actual_values = df_new['Attrition_Actual'].values
        df_new = df_new.drop(['Attrition', 'Attrition_Actual'], axis=1, errors='ignore')
    elif 'Attrition' in df_new.columns:
        df_new = df_new.drop('Attrition', axis=1, errors='ignore')
    
    df_new['TenurePerCompany'] = df_new['TotalWorkingYears'] / (df_new['NumCompaniesWorked'] + 1)
    df_new['YearsWithoutPromotion'] = df_new['YearsSinceLastPromotion']
    df_new['SalaryPerYear'] = df_new['MonthlyIncome'] / (df_new['TotalWorkingYears'] + 1)
    
    df_new['ExperienceLevel'] = pd.cut(df_new['TotalWorkingYears'], 
                                        bins=[0, 5, 10, 20, 50], 
                                        labels=['Entry', 'Mid', 'Senior', 'Expert'])
    
    df_new['IncomeLevel'] = pd.cut(df_new['MonthlyIncome'], 
                                    bins=[0, 3000, 6000, 10000, 100000], 
                                    labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    df_new['WorkLifeScore'] = (df_new['WorkLifeBalance'] + df_new['JobSatisfaction'] + 
                                df_new['EnvironmentSatisfaction'] + df_new['RelationshipSatisfaction']) / 4
    
    df_new['CareerGrowth'] = df_new['YearsAtCompany'] / (df_new['YearsSinceLastPromotion'] + 1)
    df_new['ManagerTenureRatio'] = df_new['YearsWithCurrManager'] / (df_new['YearsAtCompany'] + 1)
    df_new['RoleStability'] = df_new['YearsInCurrentRole'] / (df_new['YearsAtCompany'] + 1)
    
    df_new['IsOverworked'] = ((df_new['OverTime'] == 'Yes') & (df_new['WorkLifeBalance'] <= 2)).astype(int)
    df_new['IsUnderpaid'] = 0
    df_new['HighPerformerStuck'] = ((df_new['PerformanceRating'] >= 3) & (df_new['YearsSinceLastPromotion'] >= 3)).astype(int)
    df_new['JobHopperFlag'] = (df_new['NumCompaniesWorked'] >= 4).astype(int)
    df_new['LongDistanceCommute'] = (df_new['DistanceFromHome'] >= 20).astype(int)
    df_new['LowEngagement'] = ((df_new['JobInvolvement'] <= 2) | (df_new['EnvironmentSatisfaction'] <= 2)).astype(int)
    df_new['CompensationRatio'] = df_new['MonthlyIncome'] / (df_new['TotalWorkingYears'] * 1000 + 1)
    df_new['TrainingEngagement'] = df_new['TrainingTimesLastYear'] * df_new['JobInvolvement']
    df_new['AgeExperienceGap'] = df_new['Age'] - df_new['TotalWorkingYears'] - 18
    df_new['PromotionFrequency'] = df_new['TotalWorkingYears'] / (df_new['YearsSinceLastPromotion'] + 1)
    df_new['JobLevelMismatch'] = ((df_new['TotalWorkingYears'] >= 10) & (df_new['JobLevel'] <= 2)).astype(int)
    
    categorical_cols = df_new.select_dtypes(include=['object', 'category']).columns.tolist()
    df_encoded = pd.get_dummies(df_new, columns=categorical_cols, drop_first=True)
    
    for col in model_info['feature_names']:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[model_info['feature_names']]
    
    df_scaled = scaler.transform(df_encoded)
    
    prediction = model.predict(df_scaled)
    probability = model.predict_proba(df_scaled)
    
    return prediction, probability, actual_values

print("Select testing option:")
print("1. Test random employee from dataset")
print("2. Test specific employee by ID")
print("3. Test multiple random employees")
print("4. Test all employees who left (high risk)")
print("5. Test all employees who stayed (low risk)")
print("6. Compare predictions vs actual for sample")

choice = input("\nChoice (1-6): ")

if choice == '1':
    random_idx = np.random.randint(0, len(df))
    employee = df.iloc[random_idx:random_idx+1].copy()
    
    print(f"\nTesting employee at index {random_idx}")
    print("\nEmployee details:")
    print(f"Age: {employee['Age'].values[0]}")
    print(f"Department: {employee['Department'].values[0]}")
    print(f"Job Role: {employee['JobRole'].values[0]}")
    print(f"Monthly Income: ${employee['MonthlyIncome'].values[0]}")
    print(f"Years at Company: {employee['YearsAtCompany'].values[0]}")
    print(f"Overtime: {employee['OverTime'].values[0]}")
    print(f"Actual Status: {employee['Attrition'].values[0]}")
    
    prediction, probability, actual = make_prediction(employee, show_actual=True)
    
    print("\nPrediction:")
    print(f"Model predicts: {'Will leave' if prediction[0] == 1 else 'Will stay'}")
    print(f"Leave probability: {probability[0][1]:.1%}")
    print(f"Actual outcome: {'Left' if actual[0] == 1 else 'Stayed'}")
    print(f"Prediction: {'Correct' if prediction[0] == actual[0] else 'Incorrect'}")

elif choice == '2':
    emp_id = int(input("\nEnter employee index (0 to {}): ".format(len(df)-1)))
    
    if emp_id < 0 or emp_id >= len(df):
        print("Invalid index")
    else:
        employee = df.iloc[emp_id:emp_id+1].copy()
        
        print(f"\nEmployee {emp_id} details:")
        print(f"Age: {employee['Age'].values[0]}")
        print(f"Department: {employee['Department'].values[0]}")
        print(f"Job Role: {employee['JobRole'].values[0]}")
        print(f"Monthly Income: ${employee['MonthlyIncome'].values[0]}")
        print(f"Years at Company: {employee['YearsAtCompany'].values[0]}")
        print(f"Job Satisfaction: {employee['JobSatisfaction'].values[0]}/4")
        print(f"Work Life Balance: {employee['WorkLifeBalance'].values[0]}/4")
        print(f"Overtime: {employee['OverTime'].values[0]}")
        print(f"Actual Status: {employee['Attrition'].values[0]}")
        
        prediction, probability, actual = make_prediction(employee, show_actual=True)
        
        print("\nPrediction:")
        print(f"Model predicts: {'Will leave' if prediction[0] == 1 else 'Will stay'}")
        print(f"Leave probability: {probability[0][1]:.1%}")
        print(f"Actual outcome: {'Left' if actual[0] == 1 else 'Stayed'}")
        print(f"Prediction: {'Correct' if prediction[0] == actual[0] else 'Incorrect'}")

elif choice == '3':
    n = int(input("\nHow many employees to test? "))
    random_indices = np.random.choice(len(df), size=min(n, len(df)), replace=False)
    employees = df.iloc[random_indices].copy()
    
    prediction, probability, actual = make_prediction(employees, show_actual=True)
    
    print(f"\nTesting {len(employees)} random employees")
    print("-" * 70)
    
    correct = 0
    for i in range(len(employees)):
        emp = employees.iloc[i]
        pred_status = 'Leave' if prediction[i] == 1 else 'Stay'
        actual_status = 'Left' if actual[i] == 1 else 'Stayed'
        is_correct = prediction[i] == actual[i]
        
        if is_correct:
            correct += 1
        
        print(f"\nEmployee {i+1}: {emp['JobRole']}, Age {emp['Age']}, Income ${emp['MonthlyIncome']}")
        print(f"  Predicted: {pred_status} ({probability[i][1]:.1%} chance to leave)")
        print(f"  Actual: {actual_status}")
        print(f"  Result: {'✓ Correct' if is_correct else '✗ Incorrect'}")
    
    accuracy = correct / len(employees) * 100
    print(f"\nAccuracy: {correct}/{len(employees)} = {accuracy:.1f}%")

elif choice == '4':
    left_employees = df[df['Attrition'] == 'Yes'].head(10).copy()
    
    print(f"\nTesting {len(left_employees)} employees who actually left")
    print("-" * 70)
    
    prediction, probability, actual = make_prediction(left_employees, show_actual=True)
    
    correct = 0
    for i in range(len(left_employees)):
        emp = left_employees.iloc[i]
        
        if prediction[i] == 1:
            correct += 1
        
        print(f"\nEmployee {i+1}: {emp['JobRole']}, Age {emp['Age']}")
        print(f"  Monthly Income: ${emp['MonthlyIncome']}")
        print(f"  Years at Company: {emp['YearsAtCompany']}")
        print(f"  Overtime: {emp['OverTime']}")
        print(f"  Model prediction: {'Will leave' if prediction[i] == 1 else 'Will stay'} ({probability[i][1]:.1%})")
        print(f"  Result: {'✓ Correctly identified' if prediction[i] == 1 else '✗ Missed'}")
    
    recall = correct / len(left_employees) * 100
    print(f"\nRecall (detected leavers): {correct}/{len(left_employees)} = {recall:.1f}%")

elif choice == '5':
    stayed_employees = df[df['Attrition'] == 'No'].head(10).copy()
    
    print(f"\nTesting {len(stayed_employees)} employees who actually stayed")
    print("-" * 70)
    
    prediction, probability, actual = make_prediction(stayed_employees, show_actual=True)
    
    correct = 0
    for i in range(len(stayed_employees)):
        emp = stayed_employees.iloc[i]
        
        if prediction[i] == 0:
            correct += 1
        
        print(f"\nEmployee {i+1}: {emp['JobRole']}, Age {emp['Age']}")
        print(f"  Monthly Income: ${emp['MonthlyIncome']}")
        print(f"  Years at Company: {emp['YearsAtCompany']}")
        print(f"  Job Satisfaction: {emp['JobSatisfaction']}/4")
        print(f"  Model prediction: {'Will leave' if prediction[i] == 1 else 'Will stay'} ({probability[i][0]:.1%})")
        print(f"  Result: {'✓ Correctly identified' if prediction[i] == 0 else '✗ False alarm'}")
    
    specificity = correct / len(stayed_employees) * 100
    print(f"\nSpecificity (detected stayers): {correct}/{len(stayed_employees)} = {specificity:.1f}%")

elif choice == '6':
    n = int(input("\nHow many employees to sample? "))
    sample = df.sample(n=min(n, len(df))).copy()
    
    prediction, probability, actual = make_prediction(sample, show_actual=True)
    
    results_df = pd.DataFrame({
        'Age': sample['Age'].values,
        'Department': sample['Department'].values,
        'JobRole': sample['JobRole'].values,
        'MonthlyIncome': sample['MonthlyIncome'].values,
        'YearsAtCompany': sample['YearsAtCompany'].values,
        'Actual': ['Left' if a == 1 else 'Stayed' for a in actual],
        'Predicted': ['Leave' if p == 1 else 'Stay' for p in prediction],
        'Probability': [f"{prob[1]:.1%}" for prob in probability],
        'Correct': ['✓' if p == a else '✗' for p, a in zip(prediction, actual)]
    })
    
    print("\nComparison Table:")
    print(results_df.to_string(index=False))
    
    accuracy = (prediction == actual).sum() / len(actual) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    
    true_positive = ((prediction == 1) & (actual == 1)).sum()
    false_positive = ((prediction == 1) & (actual == 0)).sum()
    true_negative = ((prediction == 0) & (actual == 0)).sum()
    false_negative = ((prediction == 0) & (actual == 1)).sum()
    
    print(f"Correctly predicted leavers: {true_positive}")
    print(f"Correctly predicted stayers: {true_negative}")
    print(f"False alarms: {false_positive}")
    print(f"Missed leavers: {false_negative}")

else:
    print("Invalid choice")

print("\nDone")