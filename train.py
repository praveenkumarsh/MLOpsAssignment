import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import psutil
import time

# Load dataset from CSV
data = pd.read_csv('heart_disease.csv')
df = pd.DataFrame(data)

# Splitting the dataset into features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining categorical and numerical columns
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
numerical_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# Creating preprocessing pipelines
categorical_transformer = OneHotEncoder(drop="first", handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Creating a pipeline with preprocessing and model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])


# Creating a pipeline with preprocessing and model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Starting MLflow run
with mlflow.start_run():
    # Log system metrics before training
    cpu_start = psutil.cpu_percent()
    mem_start = psutil.virtual_memory().percent

    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predicting the test set results
    y_pred = model.predict(X_test)

    # Logging metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_metric('training_time', training_time)
    mlflow.log_metric('cpu_usage_start', cpu_start)
    mlflow.log_metric('mem_usage_start', mem_start)
    mlflow.log_text(report, 'classification_report.txt')
    mlflow.sklearn.log_model(model, 'model')

    # Log system metrics after training
    cpu_end = psutil.cpu_percent()
    mem_end = psutil.virtual_memory().percent
    mlflow.log_metric('cpu_usage_end', cpu_end)
    mlflow.log_metric('mem_usage_end', mem_end)

    # Save the model pipeline
    joblib.dump(model, 'model.joblib')
