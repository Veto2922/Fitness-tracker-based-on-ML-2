from flask import Flask, request, jsonify
import sys

sys.path.append('src/models/predict_model.py')
from src.models.predict_model import Tracker

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/add')
def hello():
    return "you are in add!"

@app.route('/predict', methods=['POST'])
def predict():
    
    acc = request.files['acc']
    gyr = request.files['gyr']
    
    acc_path = acc.filename
    gyr_path = gyr.filename
    
  
    acc.save(acc_path)
    gyr.save(gyr_path)
        
    Track = Tracker(acc_path, gyr_path)
    print(Track.model())
    return "Prediction is done"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
