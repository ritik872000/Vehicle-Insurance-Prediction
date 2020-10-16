import numpy as np
from flask import Flask, render_template, request, url_for
import pickle

app = Flask('__name__', template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=["GET","POST"])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features)
    if prediction == 1:
        result = 'YAY! User will buy the insurance.'
    else:
        result = 'OOPS! User wont buy the insurance.'

    return render_template('index.html', prediction_text='Prediction: {}'.format(result))


if __name__ == '__main__':
    app.run()
