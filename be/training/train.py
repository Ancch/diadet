import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("../data/diabetes_dataset.csv")

df = df.drop(columns=["PatientID"])

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = X.columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42)
}

best_model = None
best_acc = 0

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        best_model = pipe

print("Best accuracy:", best_acc)

joblib.dump(best_model, "../model/diabetes_pipeline.pkl")
print("Model saved.")
