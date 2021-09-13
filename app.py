import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = pickle.NONE
    if prediction==0:
        output = "NOT Be Promoted"
    elif prediction==1:
        output = "Be Promoted"

    return render_template("index.html", prediction_text = "This Employee should  {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)