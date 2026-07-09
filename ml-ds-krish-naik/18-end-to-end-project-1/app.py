from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import os

# Configure the templates folder as 'template' to match the project directory
app = Flask(__name__, template_folder="template")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        # GET request renders the dashboard and predict forms
        return render_template("index.html")

    # POST request processes the form data and runs predictions
    try:
        # Get form data from request
        input_data = request.form.to_dict()
        input_df = pd.DataFrame([input_data])

        # Cast score values to float/int, while leaving categoricals as strings
        if "reading_score" in input_df:
            input_df["reading_score"] = pd.to_numeric(
                input_df["reading_score"], errors="coerce"
            )
        if "writing_score" in input_df:
            input_df["writing_score"] = pd.to_numeric(
                input_df["writing_score"], errors="coerce"
            )

        # Locate artifacts (model and preprocessor)
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

        # Fallback to current directory if not found in artifacts
        if not os.path.exists(model_path):
            model_path = "model.pkl"
        if not os.path.exists(preprocessor_path):
            preprocessor_path = "preprocessor.pkl"

        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            return render_template(
                "index.html",
                prediction_text="Error: Preprocessor or model pkl file not found. Please train the model first.",
            )

        # Load models
        model = joblib.load(model_path)
        scaler = joblib.load(preprocessor_path)

        # Scale and preprocess features
        scaled_data = scaler.transform(input_df)

        # Run prediction
        prediction = model.predict(scaled_data)

        # Format predicted math score
        predicted_score = round(prediction[0], 2)

        return render_template("index.html", prediction_text=f"{predicted_score}")
    except Exception as e:
        return render_template(
            "index.html", prediction_text=f"Error occurred during prediction: {str(e)}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
