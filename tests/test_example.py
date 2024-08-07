import unittest
import json
from script import app


class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_hello_world_stash(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        message = "Application is up and running"
        self.assertEqual(response.json, {"message": message})

    def test_predict_correct_data(self):
        correct_data = [
            {
                "Age": 45, "RestingBP": 130, "Cholesterol": 250, "MaxHR": 150,
                "Oldpeak": 1.5,
                "Sex": "M", "ChestPainType": "TYP", "RestingECG": "NORM",
                "ExerciseAngina": "N", "ST_Slope": "Flat"
            }
        ]
        response = self.app.post('/predict', data=json.dumps(correct_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(isinstance(response.json, list))

    def test_predict_incorrect_data(self):
        incorrect_data = [
            {
                "Age": 45, "RestingBP": 130, "Cholesterol": 250, "MaxHR": 150,
                "Oldpeak": 1.5,
                "Sex": "M", "ChestPainType": "TYP", "RestingECG": "NORM",
                "ExerciseAngina": "N"
                # Missing ST_Slope
            }
        ]
        response = self.app.post('/predict', data=json.dumps(incorrect_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "Incorrect input columns"})

    def test_predict_missing_data(self):
        missing_data = []
        response = self.app.post('/predict', data=json.dumps(missing_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)  # Update to expect 400
        self.assertTrue("error" in response.json)


if __name__ == '__main__':
    unittest.main()
