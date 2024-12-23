from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    size = float(request.form['size'])
    bedrooms = int(request.form['bedrooms'])
    
    # Make prediction
    prediction = model.predict(np.array([[size, bedrooms]]))
    
    # Return prediction result
    return jsonify({'predicted_price': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True)
