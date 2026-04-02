# Decision Tree Classifier for Diabetes

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1 & 2. Load dataset & assign columns
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, header=None, names=columns)

# 3. Check for missing/unrealistic zero values
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# 4 & 5. Define features (X) and target (y), split dataset (80/20, random_state=42)
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6 & 7. Train standard Decision Tree & Evaluate
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

print("--- Standard Decision Tree Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Train restricted Decision Tree (max_depth=3) & Compare
dt_restricted = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_restricted.fit(X_train, y_train)
y_pred_restricted = dt_restricted.predict(X_test)

print("\n--- Restricted Decision Tree (max_depth=3) Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_restricted):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_restricted))
print("Classification Report:\n", classification_report(y_test, y_pred_restricted))

# Extract Feature Importance
print("\nFeature Importances (Restricted Tree):")
for feature, importance in zip(X.columns, dt_restricted.feature_importances_):
    if importance > 0:
        print(f"{feature}: {importance:.4f}")
