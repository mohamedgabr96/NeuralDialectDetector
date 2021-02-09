from flask import Flask, request, render_template
import sys
import json
import requests
app = Flask(__name__)


country_predict_API = "http://b4e3314b-aea0-40fd-9c63-83b1242402e1.eastus.azurecontainer.io/score"

# Functions for the WebApp
@app.route('/', methods=["GET"])
def my_form():
    return render_template('first_page.html')


@app.route('/', methods=['POST'])
def my_form_post():
    input_text = request.form['text']
    results = requests.post(country_predict_API, json.dumps({'input': input_text}), headers={"Content-Type":"application/json", "Accept":'application/json'})
    results_final = results.json()
    return render_template('second_page.html', sentence_in=input_text, predict_country=results_final["country_prediction"])


# APIs
# One function for country level, and another for the provincial
@app.route("/predict_class", methods=['GET', 'POST'])
def predict_class():
    content = request.json
    results = requests.post(country_predict_API, json.dumps({'input': content["input_example"]}), headers={"Content-Type":"application/json", "Accept":'application/json'})
    results_final = results.json()
    return str([]) + " " + str(results_final["country_prediction"])
