import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('dataset\cleaned_dataset.csv')

# Select the required columns (10 columns)
required_columns = [
    'loan_amnt', 'int_rate', 'annual_inc', 'dti',
    'fico_range_low', 'open_acc', 'revol_bal',
    'revol_util', 'delinq_2yrs', 'loan_status'  # Target variable
]

# Create a new DataFrame with only the required columns
df = data[required_columns]

# Preprocess the data
df.fillna(df.median(), inplace=True)  # Fill missing values with median

# Split the data into features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Initialize variables to keep track of the best model and score
best_model = None
best_score = 0

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)  # Train the model
    y_pred = model.predict(X_test_scaled)  # Make predictions
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy

    print(f"{model_name} Accuracy: {accuracy:.4f}")  # Print accuracy

    # Check if this model is the best one
    if accuracy > best_score:
        best_score = accuracy
        best_model = model

# Print the best model's details
print(f"Best Model: {best_model} with Accuracy: {best_score:.4f}")

# Save the best model with a new name
joblib.dump(best_model, 'final_best_model.pkl')