
from flask import Flask
from flask import request
from sklearn.linear_model import Perceptron
import pickle
import numpy as np

# Create a flask
app = Flask(__name__)

# Create an API end point
@app.route("/api/v1.0/predict", methods=['GET'])
def make_prediction():
    sl = request.args.get("sl", 0, type=float)
    sw = request.args.get("sw", 0, type=float)
    pl = request.args.get("pl", 0, type=float)
    pw = request.args.get("pw", 0, type=float)
    data = np.array([sl, sw, pl, pw]).reshape(1, -1)
    
    with open("model_app.pkl", "rb") as fh:    
        model = pickle.load(fh)
    
    pred = model.predict(data)[0]
    
    return {"prediction": pred, "features": {"sl": sl, "sw": sw, "pl": pl, "pw": pw}}

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5050)
