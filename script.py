from flask import Flask, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Define a route for the default URL, which serves as an endpoint
@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify(message="Hello, World!")

# Run the application
if __name__ == '__main__':
    app.run(debug=True, port=5000)
