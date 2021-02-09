To Launch the webapp

1. Open Terminal
2. Go to the WebApp directory
3. export FLASK_APP=main.py
4. python -m flask run

To get a prediction, make a GET or POST request on localhost:5000/predict_class with a json body of the following format:
{"input_example": "الوطن العربي فيه واحد و عشرين دولة، تخيل!"}