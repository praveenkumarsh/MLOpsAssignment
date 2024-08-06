import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, f1_score)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Defining categorical and numerical columns
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina",
                    "ST_Slope"]
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
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    "classifier__n_estimators": [50, 100, 150, 200],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10]
}

# Setup GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy',
                           n_jobs=-1, verbose=1)

# Starting MLflow run
with mlflow.start_run():
    # Log system metrics before training
    cpu_start = psutil.cpu_percent()
    mem_start = psutil.virtual_memory().percent

    # Train the model with GridSearchCV
    start_time = time.time()
    model = grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Predicting the test set results with the best model
    y_pred = best_model.predict(X_test)

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
    mlflow.sklearn.log_model(best_model, 'model')

    # Log system metrics after training
    cpu_end = psutil.cpu_percent()
    mem_end = psutil.virtual_memory().percent
    mlflow.log_metric('cpu_usage_end', cpu_end)
    mlflow.log_metric('mem_usage_end', mem_end)

    # Save the best model pipeline
    joblib.dump(best_model, 'model.joblib')

    # Log the best parameters
    mlflow.log_params(grid_search.best_params_)

    signature = infer_signature(X_test, y_pred)
    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="sk-learn-heart-disease-reg-model",
    )
