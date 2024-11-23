# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Travel details dataset.csv'
data = pd.read_csv(file_path)

# Data Cleaning
# Drop rows with missing values
data = data.dropna()

# Convert numeric columns
data['Accommodation cost'] = pd.to_numeric(data['Accommodation cost'], errors='coerce')
data['Transportation cost'] = pd.to_numeric(data['Transportation cost'], errors='coerce')

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Destination', 'Traveler gender', 'Traveler nationality',
                       'Accommodation type', 'Transportation type']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature selection
features = ['Duration (days)', 'Traveler age', 'Traveler gender', 'Accommodation cost']
target = 'Accommodation type'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[['Duration (days)', 'Traveler age', 'Accommodation cost']] = scaler.fit_transform(
    X_train[['Duration (days)', 'Traveler age', 'Accommodation cost']])
X_test[['Duration (days)', 'Traveler age', 'Accommodation cost']] = scaler.transform(
    X_test[['Duration (days)', 'Traveler age', 'Accommodation cost']])

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
classification_report_text = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Additional Metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Output Results
print("\nClassification Report:\n", classification_report_text)
print("\nConfusion Matrix:\n", confusion)
print("Model Accuracy:", accuracy)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Decode test results
y_test_decoded = label_encoders[target].inverse_transform(y_test)
y_pred_decoded = label_encoders[target].inverse_transform(y_pred)

# Create a DataFrame for test results
test_results_decoded = pd.DataFrame({
    'Actual': y_test_decoded,
    'Predicted': y_pred_decoded
}).reset_index(drop=True)

# Print the first few rows of decoded test results
print("\nSample Test Results:")
print(test_results_decoded.head(10))

# Feature Importance
feature_importances = model.feature_importances_
plt.figure(figsize=(8, 6))
sns.barplot(x=features, y=feature_importances, palette="viridis")
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()
