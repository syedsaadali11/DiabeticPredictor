import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
df = pd.read_csv('D:\diabeetic\diabetes_predictor\diabetes_predictor\diabetes_012_health_indicators_BRFSS2015.csv')
X = df.drop('Diabetes_012', axis=1).values
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, 'D:/diabeetic/diabetes_predictor/predictor/models/scaler.pkl')