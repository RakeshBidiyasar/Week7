import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the dataset and model
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", "rb"))

@app.route('/')
def index():
    # Send location list to HTML form
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Collect form input
        location = request.form.get('location')
        bhk = int(request.form.get('bhk'))
        bath = int(request.form.get('bath'))
        sqft = float(request.form.get('total_sqft'))

        # Prepare DataFrame (no 'size' column now)
        input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                                columns=['location', 'total_sqft', 'bath', 'bhk'])

        # Predict price
        prediction = pipe.predict(input_df)[0] * 25000
        return str(round(prediction, 2))

    except Exception as e:
        return f"Error in prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True, port=5001)
