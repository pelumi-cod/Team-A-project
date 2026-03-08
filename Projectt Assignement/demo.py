# Stroke Prediction using GradientBoostingClassifier with manual patient input
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# -----------------------
# 1️⃣ Create small dataset (for demo/testing)
# -----------------------
data = pd.DataFrame({
    'gender':[1,0,1],
    'age':[60,45,70],
    'hypertension':[0,1,1],
    'heart_disease':[1,0,1],
    'ever_married':[1,1,1],
    'work_type':[0,1,2],
    'Residence_type':[1,0,1],
    'avg_glucose_level':[105.5,95.2,130.6],
    'bmi':[27.8,24.1,30.5],
    'smoking_status':[2,0,1],
    'stroke':[1,0,1]
})

# -----------------------
# 2️⃣ Define features and target
# -----------------------
X = data.drop('stroke', axis=1)
y = data['stroke']

# -----------------------
# 3️⃣ Train Gradient Boosting model
# -----------------------
model = GradientBoostingClassifier()
model.fit(X, y)

# -----------------------
# 4️⃣ Function to predict stroke for manual input
# -----------------------
def predict_stroke():
    print("Enter patient details:")
    try:
        gender = int(input("Gender (0=Female, 1=Male): "))
        age = int(input("Age: "))
        hypertension = int(input("Hypertension (0=No, 1=Yes): "))
        heart_disease = int(input("Heart Disease (0=No, 1=Yes): "))
        ever_married = int(input("Ever Married (0=No, 1=Yes): "))
        work_type = int(input("Work Type (0=Private,1=Self-employed,2=Govt_job,3=children,4=Never_worked): "))
        Residence_type = int(input("Residence Type (0=Rural,1=Urban): "))
        avg_glucose_level = float(input("Average Glucose Level: "))
        bmi = float(input("BMI: "))
        smoking_status = int(input("Smoking Status (0=never smoked,1=formerly smoked,2=smokes,3=Unknown): "))
    except EOFError:
        print("Input error! Using default values.")
        gender, age, hypertension, heart_disease = 1, 60, 0, 0
        ever_married, work_type, Residence_type = 1, 0, 1
        avg_glucose_level, bmi, smoking_status = 100.0, 25.0, 0

    patient_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    df = pd.DataFrame([patient_data])
    prediction = model.predict(df)[0]
    print("Prediction: ", "This Patient is liable to have Stroke" if prediction == 1 else "No Stroke")

# -----------------------
# 5️⃣ Run the interactive function
# -----------------------
predict_stroke()
