from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(request.form[feature]) for feature in ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    return render_template('index.html', prediction=f'Predicted House Price: ${prediction * 100000:.2f}')  # California housing target is in 100k

if __name__ == '__main__':
    app.run(debug=True)