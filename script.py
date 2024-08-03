from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model pipeline
model = joblib.load('model.joblib')


@app.route('/', methods=['GET'])
def hello_world_stash():
    return jsonify(message="Application is up and running")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json

        # Create DataFrame from JSON data
        df = pd.DataFrame(data)

        # Ensure columns match the training columns
        num_features = model.named_steps['preprocessor'].transformers_[0][2]
        cat_features = model.named_steps['preprocessor'].transformers_[1][2]
        expected_columns = num_features + cat_features
        if not all(col in df.columns for col in expected_columns):
            return jsonify({"error": "Incorrect input columns"}), 400

        # Predict using the model pipeline
        predictions = model.predict(df)

        # Return predictions as JSON
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
