import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from pris import X_test, y_test, plea_list

model = joblib.load('model')

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    plea_orc_list = plea_list
    return render_template('index.html', plea_orc_list=plea_orc_list)

@app.route('/predict', methods=['POST'])
def predict():
    plea_orc_list = plea_list
    input_plea = request.form['plea_orc']
    input_race = request.form['race']
    if input_race == 'white':
        input_race = '4'
    elif input_race =='black':
        input_race = '1'
    elif input_race == 'hispanic':
        input_race = '2'
    elif input_race == 'asian':
        input_race = '0'
    elif input_race == 'other':
        input_race = '3'
    input_prior = request.form['prior_charges']

    user_input = [[input_plea, input_race, input_prior]]

    output = model.predict(user_input)
    score = model.score(X_test, y_test)
    return render_template('index.html', prediction_text='1 for yes 0 for no: {}'.format(output), prediction_score='There is a {:.0%} chance'.format(score), plea_orc_list=plea_orc_list)

if __name__ == '__main__':
    app.run(debug=True)
