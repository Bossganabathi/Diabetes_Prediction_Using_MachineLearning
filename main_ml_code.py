import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1. Data Loading (Assuming you have a 'diabetes.csv' file)
# The Pima Indians Diabetes dataset is commonly used. It includes:
# Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, 
# DiabetesPedigreeFunction, Age, and Outcome (0 or 1).

# For demonstration, we'll create a dummy DataFrame structure. 
# In a real project, you would use: df = pd.read_csv('diabetes.csv')
data = {
    'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10], 
    'Glucose': [148, 85, 183, 89, 137, 116, 78, 115], 
    'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0], # Note: 0 is an unrealistic value/missing data
    'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0], 
    'Insulin': [0, 0, 0, 94, 168, 0, 88, 0], 
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3], 
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134],
    'Age': [50, 31, 32, 21, 33, 30, 26, 29], 
    'Outcome': [1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# The 'Outcome' column is our target variable (y).
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 2. Data Preprocessing
# Columns like Glucose, BloodPressure, SkinThickness, Insulin, and BMI 
# should not be zero. We'll replace 0s in these columns with the mean of the column.

# Identify columns with meaningful zero values (which indicate missing data)
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
X[cols_to_impute] = X[cols_to_impute].replace(0, np.nan)

# Use SimpleImputer to fill NaN values with the mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale the features (important for Logistic Regression and distance-based models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 3. Splitting the Dataset
# Split the data into 80% for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model Training (Logistic Regression)
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# 5. Model Prediction
y_pred = model.predict(X_test)

# 6. Model Evaluation
print("### Model Evaluation ###")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example of a new prediction (with dummy input)
new_input = np.array([1, 85, 66, 29, 0, 26.6, 0.351, 31]).reshape(1, -1)
# 1. Impute (if necessary, though we recommend cleaning all new data)
new_input[0, 4] = imputer.named_transformers_['insulin'].mean_ # Example: impute insulin
# 2. Scale
new_input_scaled = scaler.transform(new_input)
# 3. Predict
new_prediction = model.predict(new_input_scaled)

print("\n### New Prediction ###")
if new_prediction[0] == 0:
    print("Predicted as: Non-Diabetic")
else:
    print("Predicted as: Diabetic")