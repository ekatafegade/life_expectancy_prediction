import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

infile = open('model.pkl', 'rb')
model = pickle.load(infile)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if prediction[0] == 0:
        return render_template('index.html', prediction_text='Person is safe'.format(output))
    else:
        return render_template('index.html', prediction_text='Person has chances of getting Heart Attack'.format(output))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9090)
