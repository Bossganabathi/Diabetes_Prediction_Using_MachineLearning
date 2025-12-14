import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data():
    data = {
        'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10],
        'Glucose': [148, 85, 183, 89, 137, 116, 78, 115],
        'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0],
        'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0],
        'Insulin': [0, 0, 0, 94, 168, 0, 88, 0],
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134],
        'Age': [50, 31, 32, 21, 33, 30, 26, 29],
        'Outcome': [1, 0, 1, 0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


def preprocess_data(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    X[cols_to_impute] = X[cols_to_impute].replace(0, np.nan)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y, imputer, scaler


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def predict(model, scaler, imputer, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data[:, 4] = np.nan  # insulin example
    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)
    return model.predict(input_data)
