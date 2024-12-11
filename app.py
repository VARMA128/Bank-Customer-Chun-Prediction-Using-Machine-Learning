import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')  # Ensure the model is saved as 'model.pkl'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction[0]
    if output == 1:
        return render_template('index.html', prediction_text='CUSTOMER WILL EXIT')
    else:
        return render_template('index.html', prediction_text='CUSTOMER WILL NOT EXIT')

if __name__ == "__main__":
    app.run(host="localhost", port=6067)
