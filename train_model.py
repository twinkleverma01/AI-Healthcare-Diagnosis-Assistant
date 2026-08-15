import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import joblib


data = pd.read_csv("dataset/disease_dataset.csv")


symptom_columns = ["symptom1", "symptom2", "symptom3", "symptom4"]

data["symptoms"] = data[symptom_columns].values.tolist()


mlb = MultiLabelBinarizer()
X = mlb.fit_transform(data["symptoms"])


y = data["disease"]


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


joblib.dump(model, "model/disease_model.pkl")
joblib.dump(mlb, "model/symptom_encoder.pkl")

print("Model trained successfully!")
print("Model saved in the model folder.") 