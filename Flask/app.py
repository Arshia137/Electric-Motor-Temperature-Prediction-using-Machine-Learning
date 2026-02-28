import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("model.save")
transform = joblib.load("transform.save")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [
            float(request.form['u_q']),
            float(request.form['coolant']),
            float(request.form['u_d']),
            float(request.form['motor_speed']),
            float(request.form['i_d']),
            float(request.form['i_q']),
            float(request.form['ambient']),
            float(request.form['torque'])
        ]

        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = transform.transform(input_array)
        prediction = model.predict(input_scaled)

        return render_template(
            "index.html",
            prediction_text=f"Predicted PM Temperature: {round(prediction[0],2)} °C"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)