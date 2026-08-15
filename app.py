from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model/disease_model.pkl")
encoder = joblib.load("model/symptom_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get selected symptoms from checkboxes
    symptoms = request.form.getlist("symptoms")

    if not symptoms:
        return render_template(
            "index.html",
            error="Please select at least one symptom."
        )

    # Convert symptoms into model features
    input_data = encoder.transform([symptoms])

    # Predict disease
    prediction = model.predict(input_data)

    disease = prediction[0]

    return render_template(
        "index.html",
        prediction=disease,
        symptoms=", ".join(symptoms)
    )


if __name__ == "__main__":
    app.run(debug=True)